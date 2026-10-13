"""Estimate OOLONG-Pairs promotion noise from repeated frozen-H0 runs.

Each input is a split directory containing ``round_00`` (for example
``.../baseline/heldout``). Runs are paired by their persisted attempt number.
The report recommends a conservative absolute null-delta quantile for the
overall macro-F1 improvement threshold and for the per-length regression
allowance. Freeze the chosen values in the experiment TOML before optimization.
"""

import argparse
import json
import math
import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import NormalDist
from typing import Any

from shrlm.environments.diagnostics import SET_QUALITY
from shrlm.optimization.driver import load_round
from shrlm.optimization.validation import EVAL_ROUND_INDEX


def quality_value(entry: dict[str, Any]) -> float:
    """Return persisted F1, using only verifier-registered terminal values."""
    measurement = entry["verdict"].get("quality")
    if measurement is not None:
        if measurement.get("definition_id") != SET_QUALITY.identifier:
            raise ValueError(
                f"expected {SET_QUALITY.identifier!r}, got {measurement.get('definition_id')!r}"
            )
        return float(measurement["value"])
    cause = entry.get("cause")
    if cause not in SET_QUALITY.terminal_values:
        raise ValueError(
            f"run {entry.get('run_id')!r} has no F1 and cause {cause!r} has no terminal value"
        )
    return float(SET_QUALITY.terminal_values[cause])


def upper_quantile(values: list[float], quantile: float) -> float:
    """Conservative nearest-rank quantile, requiring finite non-empty input."""
    if not values:
        raise ValueError("cannot estimate a quantile from no deltas")
    if not 0 < quantile <= 1:
        raise ValueError(f"quantile must be in (0, 1], got {quantile}")
    ordered = sorted(values)
    return ordered[math.ceil(quantile * len(ordered)) - 1]


def _load_by_length_instance_attempt(
    split_dirs: list[Path],
) -> dict[str, dict[str, dict[int, float]]]:
    """F1 for every (context length, instance, attempt), instance identity kept.

    ``calibrate()`` immediately collapses the instance axis (it only ever
    needs attempt-level means); ``bootstrap_calibrate()`` and
    ``parametric_calibrate()`` need instances kept separate so they can
    resample instances independently of attempts.
    """
    values: dict[str, dict[str, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    for split_dir in split_dirs:
        runs, _verdicts, _envelope, entries = load_round(split_dir, EVAL_ROUND_INDEX)
        for entry, (instance, _completion) in zip(entries, runs, strict=True):
            if "context_len" not in instance:
                raise ValueError(f"instance {instance.get('id')!r} has no context_len")
            length = str(instance["context_len"])
            instance_id = str(entry["instance_id"])
            values[length][instance_id][int(entry["attempt"])] = quality_value(entry)
    return values


def calibrate(split_dirs: list[Path], quantile: float = 0.95) -> dict[str, Any]:
    """Compute paired attempt-level null deltas across context lengths."""
    by_length = _load_by_length_instance_attempt(split_dirs)
    values: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for length, by_instance in by_length.items():
        for by_attempt in by_instance.values():
            for attempt, value in by_attempt.items():
                values[attempt][length].append(value)

    attempts = sorted(values)
    if len(attempts) < 2:
        raise ValueError("tau calibration needs at least two attempts; use three or more")
    lengths = sorted(values[attempts[0]], key=int)
    for attempt in attempts:
        if sorted(values[attempt], key=int) != lengths:
            raise ValueError("every attempt must cover the same context lengths")

    means = {
        attempt: {
            length: sum(values[attempt][length]) / len(values[attempt][length])
            for length in lengths
        }
        for attempt in attempts
    }
    macro = {attempt: sum(means[attempt].values()) / len(lengths) for attempt in attempts}
    pairs = list(combinations(attempts, 2))
    macro_deltas = [abs(macro[right] - macro[left]) for left, right in pairs]
    by_length_deltas = {
        length: [abs(means[right][length] - means[left][length]) for left, right in pairs]
        for length in lengths
    }
    return {
        "schema": "shrlm-promotion-tau-calibration/v1",
        "quality_definition": SET_QUALITY.to_dict(),
        "quantile": quantile,
        "attempts": attempts,
        "attempt_macro_f1": {str(key): value for key, value in macro.items()},
        "attempt_f1_by_context_length": {str(attempt): means[attempt] for attempt in attempts},
        "absolute_pairwise_macro_deltas": macro_deltas,
        "absolute_pairwise_deltas_by_context_length": by_length_deltas,
        "recommended": {
            "tau_improvement": upper_quantile(macro_deltas, quantile),
            # Per context length: noise is not uniform (observed macro deltas on
            # a frozen harness can differ by over 30x between lengths), so a
            # single collapsed-to-worst-case margin is either far too loose at a
            # quiet length or still too tight at a noisy one. PromotionConfig's
            # per-length regression check consumes this mapping directly.
            "tau_regression": {
                length: upper_quantile(deltas, quantile)
                for length, deltas in by_length_deltas.items()
            },
        },
        "warning": (
            "Three attempts yield only three non-independent pairwise deltas; treat this "
            "as a pilot noise floor, freeze the result, and do not tune it on candidates."
        ),
    }


def _instance_matrices(
    split_dirs: list[Path],
) -> tuple[list[str], dict[str, list[list[float]]]]:
    """Per-length ``n_instances x n_attempts`` F1 matrices, shape-checked.

    Every length must share the same instance count and attempt count (the
    balanced design both ``bootstrap_calibrate()`` and ``parametric_calibrate()``
    assume); raises if a length's instances disagree on how many attempts they
    have, or if lengths disagree on shape with each other.
    """
    by_length = _load_by_length_instance_attempt(split_dirs)
    if not by_length:
        raise ValueError("cannot estimate a quantile from no data")
    lengths = sorted(by_length, key=int)

    matrices: dict[str, list[list[float]]] = {}
    for length in lengths:
        by_instance = by_length[length]
        instance_ids = sorted(by_instance)
        rows = []
        n_attempts = None
        for instance_id in instance_ids:
            attempts = sorted(by_instance[instance_id])
            if n_attempts is None:
                n_attempts = len(attempts)
            elif len(attempts) != n_attempts:
                raise ValueError(
                    f"context length {length}: every instance must share the same attempt count"
                )
            rows.append([by_instance[instance_id][attempt] for attempt in attempts])
        matrices[length] = rows

    shapes = {(len(matrices[length]), len(matrices[length][0])) for length in lengths}
    if len(shapes) != 1:
        raise ValueError(
            f"every context length must share the same instance/attempt shape, got {shapes}"
        )
    return lengths, matrices


def bootstrap_calibrate(
    split_dirs: list[Path],
    quantile: float = 0.95,
    n_replicates: int = 5000,
    seed: int = 0,
) -> dict[str, Any]:
    """Estimate per-length and macro null-delta quantiles by nested bootstrap.

    ``calibrate()``'s pilot report differences already attempt-averaged means
    across only as many attempts as were run (e.g. 3 attempts -> 3 pairwise
    deltas total) -- a different, variance-reduced quantity from what
    ``score_candidate`` actually compares: a flat pooled mean over every
    instance x attempt at that length (``shrlm.optimization.validation``'s
    ``by_context_length[length]["mean"]``). This resamples instances AND
    attempts with replacement -- independently per instance, a nested/cluster
    bootstrap -- to synthesize the sampling distribution of
    "pooled_mean(resample B) - pooled_mean(resample A)" under the null that
    both resamples are the same frozen harness, then takes a quantile of that
    much larger synthetic sample instead of a literal 3-point "percentile".

    Reliability caveat, always reported: the between-instance variance
    component is estimated from only as many real instances as were measured.
    Resampling with replacement recombines those instances; it cannot surface
    the difficulty of an instance that was never measured.
    """
    lengths, matrices = _instance_matrices(split_dirs)
    n_instances = len(matrices[lengths[0]])
    n_attempts = len(matrices[lengths[0]][0])
    rng = random.Random(seed)

    def resample_mean(matrix: list[list[float]]) -> float:
        total = 0.0
        for _ in range(n_instances):
            row = matrix[rng.randrange(n_instances)]
            for _ in range(n_attempts):
                total += row[rng.randrange(n_attempts)]
        return total / (n_instances * n_attempts)

    by_length_deltas: dict[str, list[float]] = {length: [] for length in lengths}
    macro_deltas: list[float] = []
    for _ in range(n_replicates):
        means_a: dict[str, float] = {}
        means_b: dict[str, float] = {}
        for length in lengths:
            means_a[length] = resample_mean(matrices[length])
            means_b[length] = resample_mean(matrices[length])
            by_length_deltas[length].append(abs(means_b[length] - means_a[length]))
        macro_deltas.append(
            abs(sum(means_b.values()) / len(lengths) - sum(means_a.values()) / len(lengths))
        )

    return {
        "schema": "shrlm-promotion-tau-calibration/v2-bootstrap",
        "quality_definition": SET_QUALITY.to_dict(),
        "quantile": quantile,
        "n_instances": n_instances,
        "n_attempts": n_attempts,
        "n_replicates": n_replicates,
        "seed": seed,
        "recommended": {
            "tau_improvement": upper_quantile(macro_deltas, quantile),
            "tau_regression": {
                length: upper_quantile(deltas, quantile)
                for length, deltas in by_length_deltas.items()
            },
        },
        "reliability": (
            f"Between-instance variance is estimated from only {n_instances} real "
            "instances per length; the bootstrap recombines them with replacement "
            "and cannot discover the difficulty of an instance that was never "
            "measured. Treat this as a large improvement over a 3-sample "
            "nearest-rank estimate, not as a guarantee of the true population "
            "quantile."
        ),
    }


def parametric_calibrate(split_dirs: list[Path], quantile: float = 0.95) -> dict[str, Any]:
    """Closed-form cross-check for ``bootstrap_calibrate()``.

    Per context length, estimates the within-instance variance (mean of each
    instance's own sample variance across its attempts) and the
    between-instance variance (method-of-moments one-way-ANOVA estimate from
    the per-instance attempt-means), propagates both via the CLT to the
    standard error of the pooled mean over ``n_instances x n_attempts`` runs,
    then scales by sqrt(2) (the promotion check differences two INDEPENDENT
    pooled means, baseline vs candidate) and the normal quantile's z-score to
    get a parametric estimate of the same quantity ``bootstrap_calibrate()``
    estimates empirically. Same small-n caveat: the between-instance term is
    itself estimated from only as many instances as were measured, and can be
    negative before clipping (handled by flooring at zero).
    """
    lengths, matrices = _instance_matrices(split_dirs)
    z = NormalDist().inv_cdf((1 + quantile) / 2)

    def sample_variance(values: list[float]) -> float:
        n = len(values)
        mean = sum(values) / n
        return sum((value - mean) ** 2 for value in values) / (n - 1)

    by_length: dict[str, dict[str, float]] = {}
    for length in lengths:
        rows = matrices[length]
        n_instances = len(rows)
        n_attempts = len(rows[0])
        instance_means = [sum(row) / n_attempts for row in rows]
        within_var = sum(sample_variance(row) for row in rows) / n_instances
        between_var = max(0.0, sample_variance(instance_means) - within_var / n_attempts)
        pooled_mean_var = between_var / n_instances + within_var / (n_instances * n_attempts)
        by_length[length] = {
            "n_instances": n_instances,
            "n_attempts": n_attempts,
            "within_instance_variance": within_var,
            "between_instance_variance": between_var,
            "pooled_mean_variance": pooled_mean_var,
            "tau_regression": z * math.sqrt(2 * pooled_mean_var),
        }

    macro_pooled_mean_var = (
        sum(item["pooled_mean_variance"] for item in by_length.values()) / len(lengths) ** 2
    )
    return {
        "schema": "shrlm-promotion-tau-calibration/v2-parametric",
        "quality_definition": SET_QUALITY.to_dict(),
        "quantile": quantile,
        "by_context_length": by_length,
        "recommended": {
            "tau_improvement": z * math.sqrt(2 * macro_pooled_mean_var),
            "tau_regression": {
                length: item["tau_regression"] for length, item in by_length.items()
            },
        },
        "reliability": (
            "Between-instance variance is a method-of-moments ANOVA estimate from "
            f"only {next(iter(by_length.values()))['n_instances']} real instances per "
            "length -- noisy with this little data, intended as an order-of-magnitude "
            "cross-check against bootstrap_calibrate(), not a replacement for it."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-dir",
        action="append",
        required=True,
        type=Path,
        help="split directory containing round_00; repeat for 8k, 16k, and 32k",
    )
    parser.add_argument("--quantile", type=float, default=0.95)
    args = parser.parse_args()
    print(json.dumps(calibrate(args.split_dir, args.quantile), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
