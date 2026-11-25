"""GPU dispatch is opt-in. Hardware parity tests run with RUST_EPHEM_GPU=1."""

import os
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

import rust_ephem as re
from rust_ephem import _gpu


def test_gpu_dispatch_for_large_geometric_masks(
    ground_ephemeris: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Include duplicate and reordered epochs. Cross the dispatch threshold
    # without requiring a GPU on the ordinary CPU-only test runners.
    indices = list(range(7, -1, -1)) * 128
    ra = np.linspace(0, 360, 4096, endpoint=False).tolist()
    dec = np.linspace(-90, 90, 4096).tolist()
    calls = []

    hardware = os.environ.get("RUST_EPHEM_GPU") == "1"
    hardware_mask = _gpu.cosine_mask

    def record(
        targets: NDArray[np.float64], geometry: NDArray[np.float64]
    ) -> NDArray[np.bool_]:
        calls.append(targets.shape)
        if hardware:
            return hardware_mask(targets, geometry)
        dot = targets[:, 0, None] * geometry[:, 0]
        dot += targets[:, 1, None] * geometry[:, 1]
        dot += targets[:, 2, None] * geometry[:, 2]
        return np.asarray(
            (dot > geometry[:, 3]) | (dot < geometry[:, 4]), dtype=np.bool_
        )

    monkeypatch.setattr(_gpu, "cosine_mask", record)

    for rule in [
        re.SunConstraint(min_angle=40, max_angle=140),
        re.MoonConstraint(min_angle=25),
        re.MoonLimbConstraint(min_angle=30, max_angle=150),
        re.EarthLimbConstraint(min_angle=10, max_angle=75),
    ]:
        monkeypatch.delenv("RUST_EPHEM_GPU", raising=False)
        expected = rule.in_constraint_batch(ground_ephemeris, ra, dec, indices=indices)
        calls.clear()
        monkeypatch.setenv("RUST_EPHEM_GPU", "1")
        actual = rule.in_constraint_batch(ground_ephemeris, ra, dec, indices=indices)
        assert calls
        np.testing.assert_array_equal(actual, expected)
        calls.clear()
        rule.in_constraint_batch(ground_ephemeris, ra[:1], dec[:1], indices=indices)
        assert not calls


@pytest.mark.skipif(
    os.environ.get("RUST_EPHEM_GPU") != "1", reason="requires WebGPU hardware"
)
def test_shader_resolves_rounded_thresholds() -> None:
    # These f64 directions round to exactly the same f32 vector. Include a
    # partial final word and cases at, above, and below both inclusive bounds.
    x = np.array([0.5, np.nextafter(0.5, 0.0), np.nextafter(0.5, 1.0)])
    targets = np.column_stack((x, np.sqrt(1 - x * x), np.zeros_like(x)))
    epochs = np.array([[1, 0, 0, 0.5, -16], [1, 0, 0, 16, 0.5]], dtype=float)
    np.testing.assert_array_equal(
        _gpu.cosine_mask(targets, epochs),
        [[False, False], [False, True], [True, False]],
    )


@pytest.mark.skipif(
    os.environ.get("RUST_EPHEM_GPU") != "1", reason="requires WebGPU hardware"
)
def test_gpu_tiles_preserve_target_and_epoch_order() -> None:
    rng = np.random.default_rng(37)
    targets = rng.normal(size=(129, 3))
    targets /= np.linalg.norm(targets, axis=1, keepdims=True)
    directions = rng.normal(size=(65_539, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    epochs = np.column_stack(
        (
            directions,
            rng.uniform(0, 1, len(directions)),
            rng.uniform(-1, 0, len(directions)),
        )
    )
    mask = _gpu.cosine_mask(targets, epochs)
    for row, target in zip(mask, targets):
        cosine = (
            directions[:, 0] * target[0]
            + directions[:, 1] * target[1]
            + directions[:, 2] * target[2]
        )
        np.testing.assert_array_equal(
            row, (cosine > epochs[:, 3]) | (cosine < epochs[:, 4])
        )
