"""Recalibrate OOLONG-Pairs promotion tau from data already on disk. Read-only:
loads persisted runs, spends nothing.

``examples/calibrate_oolong_pairs_tau.py``'s pilot report (``calibrate()``,
schema ``v1``) has a real statistical problem: it averages F1 across the 4
calibration instances within each attempt BEFORE computing deltas (a
different, variance-reduced quantity from what ``score_candidate`` actually
compares -- a flat pooled mean over every instance x attempt at one length),
and with only 3 attempts that leaves exactly 3 pairwise deltas, so its "95th
percentile" is literally just the largest of 3 numbers. The module's own
docstring already called this a pilot noise floor, not something to trust.

This script instead runs ``bootstrap_calibrate()``/``parametric_calibrate()``
(``shrlm/experiment/calibrate_promotion_tau.py``) -- a nested bootstrap over
thousands of resamples of the real per-instance-per-attempt data, with a
closed-form cross-check -- against two independent sources, reported
side by side and never pooled together (they cover different, non-comparable
task-id populations):

  --heldout-split-dir   the REAL held-out instances score_candidate scores
                         against (task_ids actually used in promotion).
  --pilot-split-dir     the original calibration pilot's instances (repeat
                         for 8k/16k/32k) -- a different, harder task-id slice;
                         printed only as a secondary cross-check.

A third, cruder sanity figure comes from the live experiment's own mining
rounds (held-in instances, m=2 attempts -- too thin to bootstrap meaningfully,
so this is just "is the result in the same decade", not a calibration input).

Nothing here writes the experiment TOML. Freeze the numbers you trust into
``configs/experiment_oolong_pairs_blended_DeepSeekV4Flash.toml`` by hand, with
a comment citing the source, replicate count, and quantile.
"""

import argparse
import json
from pathlib import Path

from shrlm.environments.diagnostics import SET_QUALITY
from shrlm.experiment.calibrate_promotion_tau import bootstrap_calibrate, parametric_calibrate


def quality_value(entry: dict) -> float:
    measurement = entry["verdict"].get("quality")
    if measurement is not None:
        return float(measurement["value"])
    cause = entry.get("cause")
    if cause not in SET_QUALITY.terminal_values:
        raise ValueError(f"run {entry.get('run_id')!r} has no F1 and no terminal value")
    return float(SET_QUALITY.terminal_values[cause])


def mining_sanity_figures(mining_round_dirs: list[Path]) -> dict[str, float]:
    """Crude max-pairwise-delta per length from held-in mining data.

    Not a calibration input (m=2 attempts per instance -- too thin to
    bootstrap) -- just an order-of-magnitude check that the recalibrated
    numbers aren't wildly off the scale of real held-in noise.
    """
    by_length: dict[str, list[float]] = {}
    for round_dir in mining_round_dirs:
        instances = {
            json.loads(line)["id"]: json.loads(line)
            for line in (round_dir / "instances.jsonl").read_text().splitlines()
        }
        by_instance_attempt: dict[tuple[str, str], dict[int, float]] = {}
        for line in (round_dir / "runs.jsonl").read_text().splitlines():
            entry = json.loads(line)
            instance = instances[entry["instance_id"]]
            length = str(instance["context_len"])
            key = (length, entry["instance_id"])
            by_instance_attempt.setdefault(key, {})[int(entry["attempt"])] = quality_value(entry)
        for (length, _instance_id), by_attempt in by_instance_attempt.items():
            values = list(by_attempt.values())
            if len(values) >= 2:
                by_length.setdefault(length, []).append(max(values) - min(values))
    return {length: max(deltas) for length, deltas in by_length.items() if deltas}


def print_report(label: str, report: dict) -> None:
    print(f"\n--- {label} ---")
    print(f"n_instances={report.get('n_instances')} n_attempts={report.get('n_attempts')}")
    tau = report["recommended"]
    for length in sorted(tau["tau_regression"], key=int):
        print(f"  tau_regression[{length}] = {tau['tau_regression'][length]:.6f}")
    print(f"  tau_improvement        = {tau['tau_improvement']:.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--heldout-split-dir",
        action="append",
        required=True,
        type=Path,
        help="split directory containing round_00 with the real held-out instances",
    )
    parser.add_argument(
        "--pilot-split-dir",
        action="append",
        default=[],
        type=Path,
        help="pilot calibration split directory containing round_00; repeat for 8k/16k/32k",
    )
    parser.add_argument(
        "--mining-round-dir",
        action="append",
        default=[],
        type=Path,
        help="mining round directory (e.g. opt/round_01/mining/round_01); repeat per round",
    )
    parser.add_argument("--quantile", type=float, default=0.95)
    parser.add_argument("--n-replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    heldout_bootstrap = bootstrap_calibrate(
        args.heldout_split_dir, args.quantile, args.n_replicates, args.seed
    )
    heldout_parametric = parametric_calibrate(args.heldout_split_dir, args.quantile)
    print_report("held-out, bootstrap", heldout_bootstrap)
    print_report("held-out, parametric cross-check", heldout_parametric)

    if args.pilot_split_dir:
        pilot_bootstrap = bootstrap_calibrate(
            args.pilot_split_dir, args.quantile, args.n_replicates, args.seed
        )
        pilot_parametric = parametric_calibrate(args.pilot_split_dir, args.quantile)
        print_report("pilot (different task-ids; cross-check only)", pilot_bootstrap)
        print_report("pilot, parametric cross-check", pilot_parametric)

    if args.mining_round_dir:
        sanity = mining_sanity_figures(args.mining_round_dir)
        print("\n--- held-in mining, crude max-pairwise-delta (order-of-magnitude only) ---")
        for length in sorted(sanity, key=int):
            print(f"  max_delta[{length}] = {sanity[length]:.6f}")

    print("\n--- sanity ---")
    for length, tau in heldout_bootstrap["recommended"]["tau_regression"].items():
        verdict = "OK" if 0.0 < tau <= 1.0 else "SUSPECT"
        print(f"  {length}: tau_regression={tau:.6f} ({verdict})")
    print(
        "\nNothing written. Freeze the held-out bootstrap numbers above into "
        "[promotion] in the experiment TOML by hand if they look sane."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
