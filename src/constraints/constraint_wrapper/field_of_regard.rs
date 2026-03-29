use crate::constraints::core::ConstraintEvaluator;
use crate::ephemeris::ephemeris_common::EphemerisBase;
use crate::ephemeris::GroundEphemeris;
use crate::ephemeris::OEMEphemeris;
use crate::ephemeris::SPICEEphemeris;
use crate::ephemeris::TLEEphemeris;
use ndarray::Array2;
use pyo3::prelude::*;
use std::f64::consts::PI;
use std::sync::{Arc, OnceLock, RwLock};

/// Default number of roll-angle samples for free-roll field-of-regard sweeps.
/// 72 samples gives 5° resolution, sufficient for Fibonacci-sphere accuracy.
pub(super) const DEFAULT_N_ROLL_SAMPLES: usize = 72;

/// Default number of Fibonacci-sphere sky samples for field-of-regard integration.
pub(super) const DEFAULT_N_POINTS: usize = 20_000;

struct SkySamples {
    unit_vectors: Array2<f64>,
}

const MAX_FIBONACCI_CACHE_ENTRIES: usize = 32;
const MAX_CACHEABLE_FIBONACCI_POINTS: usize = 200_000;

type SkySampleCache = std::collections::HashMap<usize, Arc<SkySamples>>;

static FIBONACCI_SAMPLES_CACHE: OnceLock<RwLock<SkySampleCache>> = OnceLock::new();

fn fibonacci_samples_cache() -> &'static RwLock<SkySampleCache> {
    FIBONACCI_SAMPLES_CACHE.get_or_init(|| RwLock::new(std::collections::HashMap::new()))
}

fn fibonacci_sphere_unit_vectors(n_points: usize) -> SkySamples {
    if n_points == 0 {
        return SkySamples {
            unit_vectors: Array2::zeros((0, 3)),
        };
    }

    // Golden angle in radians
    let golden_angle = PI * (3.0 - 5.0_f64.sqrt());
    let (sin_golden, cos_golden) = golden_angle.sin_cos();
    let mut sin_phi = 0.0;
    let mut cos_phi = 1.0;
    let n = n_points as f64;
    let mut unit_vectors = Array2::<f64>::zeros((n_points, 3));

    for i in 0..n_points {
        let k = i as f64 + 0.5;
        let z = 1.0 - (2.0 * k) / n;
        let radius_xy = (1.0 - z * z).sqrt();

        let x = radius_xy * cos_phi;
        let y = radius_xy * sin_phi;

        unit_vectors[[i, 0]] = x;
        unit_vectors[[i, 1]] = y;
        unit_vectors[[i, 2]] = z;

        // Advance phi by golden_angle using a cheap rotation recurrence.
        if i + 1 < n_points {
            let next_cos = cos_phi * cos_golden - sin_phi * sin_golden;
            let next_sin = sin_phi * cos_golden + cos_phi * sin_golden;
            cos_phi = next_cos;
            sin_phi = next_sin;
        }
    }

    SkySamples { unit_vectors }
}

fn cached_fibonacci_sphere_radec(n_points: usize) -> Arc<SkySamples> {
    // Very large sample grids are expensive to retain globally; skip cache for these.
    if n_points > MAX_CACHEABLE_FIBONACCI_POINTS {
        return Arc::new(fibonacci_sphere_unit_vectors(n_points));
    }

    {
        let cache = fibonacci_samples_cache()
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        if let Some(samples) = cache.get(&n_points) {
            return Arc::clone(samples);
        }
    }

    let mut cache = fibonacci_samples_cache()
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    if let Some(samples) = cache.get(&n_points) {
        return Arc::clone(samples);
    }

    if cache.len() >= MAX_FIBONACCI_CACHE_ENTRIES {
        if let Some(key_to_remove) = cache.keys().next().copied() {
            cache.remove(&key_to_remove);
        }
    }

    let new_samples = Arc::new(fibonacci_sphere_unit_vectors(n_points));
    cache.insert(n_points, Arc::clone(&new_samples));
    new_samples
}

#[allow(dead_code)]
pub(super) fn clear_fibonacci_samples_cache() {
    let mut cache = fibonacci_samples_cache()
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    cache.clear();
}

#[allow(clippy::too_many_arguments)]
pub(super) fn instantaneous_field_of_regard_impl<F>(
    py: Python,
    ephemeris: Py<PyAny>,
    time: Option<&Bound<PyAny>>,
    index: Option<usize>,
    n_points: usize,
    n_roll_samples: usize,
    evaluator: &dyn ConstraintEvaluator,
    parse_times_to_indices: F,
) -> PyResult<f64>
where
    F: FnOnce(&Bound<PyAny>, &Bound<PyAny>) -> PyResult<Vec<usize>>,
{
    if n_points == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "n_points must be greater than 0",
        ));
    }
    if n_roll_samples == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "n_roll_samples must be greater than 0",
        ));
    }

    let has_time = time.is_some();
    let has_index = index.is_some();
    if has_time == has_index {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Specify exactly one of 'time' or 'index'",
        ));
    }

    let bound = ephemeris.bind(py);
    let eval_index = if let Some(t) = time {
        let parsed = parse_times_to_indices(bound, t)?;
        if parsed.len() != 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "'time' must be a single datetime for instantaneous evaluation",
            ));
        }
        parsed[0]
    } else {
        index.expect("index checked above")
    };

    if has_index {
        let n_times = if let Ok(ephem) = bound.extract::<PyRef<TLEEphemeris>>() {
            ephem.get_times()?.len()
        } else if let Ok(ephem) = bound.extract::<PyRef<SPICEEphemeris>>() {
            ephem.get_times()?.len()
        } else if let Ok(ephem) = bound.extract::<PyRef<GroundEphemeris>>() {
            ephem.get_times()?.len()
        } else if let Ok(ephem) = bound.extract::<PyRef<OEMEphemeris>>() {
            ephem.get_times()?.len()
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported ephemeris type. Expected TLEEphemeris, SPICEEphemeris, GroundEphemeris, or OEMEphemeris",
            ));
        };

        if eval_index >= n_times {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "index {} out of range for ephemeris with {} timestamps",
                eval_index, n_times
            )));
        }
    }

    let sky_samples = cached_fibonacci_sphere_radec(n_points);

    let evaluate_batch = |ephem: &dyn EphemerisBase| -> PyResult<Vec<bool>> {
        evaluator.field_of_regard_violated_batch(
            ephem,
            &sky_samples.unit_vectors,
            eval_index,
            n_roll_samples,
        )
    };

    let violated: Vec<bool> = if let Ok(ephem) = bound.extract::<PyRef<TLEEphemeris>>() {
        evaluate_batch(&*ephem as &dyn EphemerisBase)?
    } else if let Ok(ephem) = bound.extract::<PyRef<SPICEEphemeris>>() {
        evaluate_batch(&*ephem as &dyn EphemerisBase)?
    } else if let Ok(ephem) = bound.extract::<PyRef<GroundEphemeris>>() {
        evaluate_batch(&*ephem as &dyn EphemerisBase)?
    } else if let Ok(ephem) = bound.extract::<PyRef<OEMEphemeris>>() {
        evaluate_batch(&*ephem as &dyn EphemerisBase)?
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "Unsupported ephemeris type. Expected TLEEphemeris, SPICEEphemeris, GroundEphemeris, or OEMEphemeris",
        ));
    };

    let visible_count = violated.iter().filter(|&&is_violated| !is_violated).count();
    let visible_fraction = visible_count as f64 / n_points as f64;

    Ok(4.0 * PI * visible_fraction)
}
