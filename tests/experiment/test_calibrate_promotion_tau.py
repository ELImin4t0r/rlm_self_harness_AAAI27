from pathlib import Path

import pytest

from shrlm.experiment import calibrate_promotion_tau
from shrlm.experiment.calibrate_promotion_tau import (
    bootstrap_calibrate,
    parametric_calibrate,
    upper_quantile,
)


def test_upper_quantile_uses_conservative_nearest_rank() -> None:
    assert upper_quantile([0.01, 0.02, 0.03], 0.95) == 0.03
    assert upper_quantile([0.01, 0.02, 0.03, 0.04], 0.50) == 0.02


def test_upper_quantile_rejects_unusable_inputs() -> None:
    with pytest.raises(ValueError, match="no deltas"):
        upper_quantile([], 0.95)
    with pytest.raises(ValueError, match="quantile"):
        upper_quantile([0.1], 0.0)


def fake_loader(values: dict[str, dict[str, dict[int, float]]]):
    """Patch the loader boundary so bootstrap/parametric calibrate run on
    known synthetic data instead of a persisted round on disk."""

    def _loader(split_dirs: list[Path]) -> dict[str, dict[str, dict[int, float]]]:
        return values

    return _loader


# Four instances, three attempts, two lengths -- matches the pilot's real
# shape (4 instances x 3 attempts), small enough to reason about by hand.
IDENTICAL_VALUES = {
    "8192": {f"inst{i}": {1: 0.9, 2: 0.9, 3: 0.9} for i in range(4)},
    "16384": {f"inst{i}": {1: 0.8, 2: 0.8, 3: 0.8} for i in range(4)},
}

VARYING_VALUES = {
    "8192": {
        "inst0": {1: 0.5, 2: 0.9, 3: 0.7},
        "inst1": {1: 0.6, 2: 0.8, 3: 0.6},
        "inst2": {1: 0.4, 2: 1.0, 3: 0.8},
        "inst3": {1: 0.5, 2: 0.7, 3: 0.9},
    },
    "16384": {
        "inst0": {1: 0.95, 2: 0.96, 3: 0.94},
        "inst1": {1: 0.93, 2: 0.95, 3: 0.97},
        "inst2": {1: 0.96, 2: 0.94, 3: 0.95},
        "inst3": {1: 0.94, 2: 0.96, 3: 0.93},
    },
}


class TestBootstrapCalibrate:
    def test_identical_values_collapse_to_a_zero_margin(self, monkeypatch) -> None:
        monkeypatch.setattr(
            calibrate_promotion_tau,
            "_load_by_length_instance_attempt",
            fake_loader(IDENTICAL_VALUES),
        )
        report = bootstrap_calibrate([Path("unused")], n_replicates=200, seed=0)
        assert report["recommended"]["tau_regression"]["8192"] == 0.0
        assert report["recommended"]["tau_regression"]["16384"] == 0.0
        assert report["recommended"]["tau_improvement"] == 0.0

    def test_varying_values_produce_a_nonzero_margin_bounded_by_the_data_range(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            calibrate_promotion_tau,
            "_load_by_length_instance_attempt",
            fake_loader(VARYING_VALUES),
        )
        report = bootstrap_calibrate([Path("unused")], n_replicates=2000, seed=0)
        tau = report["recommended"]["tau_regression"]
        # 8192 has real spread (0.4-1.0); 16384 is nearly flat (0.93-0.97).
        assert 0.0 < tau["8192"] <= 1.0
        assert 0.0 < tau["16384"] <= 1.0
        assert tau["8192"] > tau["16384"], "the noisier length must get the bigger margin"
        assert report["recommended"]["tau_improvement"] > 0.0

    def test_reports_shape_and_reliability_caveat(self, monkeypatch) -> None:
        monkeypatch.setattr(
            calibrate_promotion_tau,
            "_load_by_length_instance_attempt",
            fake_loader(VARYING_VALUES),
        )
        report = bootstrap_calibrate([Path("unused")], n_replicates=50, seed=0)
        assert report["n_instances"] == 4
        assert report["n_attempts"] == 3
        assert report["n_replicates"] == 50
        assert report["seed"] == 0
        assert "4 real" in report["reliability"]

    def test_is_reproducible_for_a_fixed_seed(self, monkeypatch) -> None:
        monkeypatch.setattr(
            calibrate_promotion_tau,
            "_load_by_length_instance_attempt",
            fake_loader(VARYING_VALUES),
        )
        first = bootstrap_calibrate([Path("unused")], n_replicates=500, seed=7)
        second = bootstrap_calibrate([Path("unused")], n_replicates=500, seed=7)
        assert first["recommended"] == second["recommended"]


class TestParametricCalibrate:
    def test_identical_values_collapse_to_a_zero_margin(self, monkeypatch) -> None:
        monkeypatch.setattr(
            calibrate_promotion_tau,
            "_load_by_length_instance_attempt",
            fake_loader(IDENTICAL_VALUES),
        )
        report = parametric_calibrate([Path("unused")])
        assert report["recommended"]["tau_regression"]["8192"] == pytest.approx(0.0)
        assert report["recommended"]["tau_regression"]["16384"] == pytest.approx(0.0)
        assert report["recommended"]["tau_improvement"] == pytest.approx(0.0)

    def test_agrees_with_bootstrap_within_the_same_order_of_magnitude(self, monkeypatch) -> None:
        monkeypatch.setattr(
            calibrate_promotion_tau,
            "_load_by_length_instance_attempt",
            fake_loader(VARYING_VALUES),
        )
        bootstrap = bootstrap_calibrate([Path("unused")], n_replicates=4000, seed=0)
        parametric = parametric_calibrate([Path("unused")])
        for length in ("8192", "16384"):
            b = bootstrap["recommended"]["tau_regression"][length]
            p = parametric["recommended"]["tau_regression"][length]
            assert p / 4 <= b <= p * 4, f"length {length}: bootstrap={b} parametric={p}"

    def test_negative_between_instance_variance_is_floored_at_zero(self, monkeypatch) -> None:
        """Within-instance noise alone can exceed the observed between-instance
        spread; the method-of-moments estimate must not go negative."""
        flat_means_high_within_noise = {
            "8192": {
                "inst0": {1: 0.1, 2: 0.9, 3: 0.5},
                "inst1": {1: 0.2, 2: 0.8, 3: 0.5},
                "inst2": {1: 0.0, 2: 1.0, 3: 0.5},
                "inst3": {1: 0.3, 2: 0.7, 3: 0.5},
            },
        }
        monkeypatch.setattr(
            calibrate_promotion_tau,
            "_load_by_length_instance_attempt",
            fake_loader(flat_means_high_within_noise),
        )
        report = parametric_calibrate([Path("unused")])
        assert report["by_context_length"]["8192"]["between_instance_variance"] >= 0.0
        assert report["recommended"]["tau_regression"]["8192"] >= 0.0


class TestInstanceMatricesShapeValidation:
    def test_mismatched_attempt_counts_within_a_length_raise(self, monkeypatch) -> None:
        bad = {
            "8192": {
                "inst0": {1: 0.5, 2: 0.6, 3: 0.7},
                "inst1": {1: 0.5, 2: 0.6},
            },
        }
        monkeypatch.setattr(
            calibrate_promotion_tau, "_load_by_length_instance_attempt", fake_loader(bad)
        )
        with pytest.raises(ValueError, match="same attempt count"):
            bootstrap_calibrate([Path("unused")])

    def test_mismatched_shapes_across_lengths_raise(self, monkeypatch) -> None:
        bad = {
            "8192": {f"inst{i}": {1: 0.5, 2: 0.6, 3: 0.7} for i in range(4)},
            "16384": {f"inst{i}": {1: 0.5, 2: 0.6, 3: 0.7} for i in range(3)},
        }
        monkeypatch.setattr(
            calibrate_promotion_tau, "_load_by_length_instance_attempt", fake_loader(bad)
        )
        with pytest.raises(ValueError, match="same instance/attempt shape"):
            bootstrap_calibrate([Path("unused")])

    def test_no_data_raises(self, monkeypatch) -> None:
        monkeypatch.setattr(
            calibrate_promotion_tau, "_load_by_length_instance_attempt", fake_loader({})
        )
        with pytest.raises(ValueError, match="no data"):
            bootstrap_calibrate([Path("unused")])
