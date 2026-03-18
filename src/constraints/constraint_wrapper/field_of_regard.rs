use crate::constraints::core::ConstraintEvaluator;
use crate::ephemeris::ephemeris_common::EphemerisBase;
use crate::ephemeris::GroundEphemeris;
use crate::ephemeris::OEMEphemeris;
use crate::ephemeris::SPICEEphemeris;
use crate::ephemeris::TLEEphemeris;
use pyo3::prelude::*;
use std::f64::consts::PI;
use std::sync::{Arc, OnceLock, RwLock};

#[derive(Clone)]
struct SkySamples {
    ras: Vec<f64>,
    decs: Vec<f64>,
}

type SkySampleCache = std::collections::HashMap<usize, Arc<SkySamples>>;
static FIBONACCI_SAMPLES_CACHE: OnceLock<RwLock<SkySampleCache>> = OnceLock::new();

fn fibonacci_samples_cache() -> &'static RwLock<SkySampleCache> {
    FIBONACCI_SAMPLES_CACHE.get_or_init(|| RwLock::new(std::collections::HashMap::new()))
}

fn fibonacci_sphere_radec(n_points: usize) -> SkySamples {
    let mut ras = Vec::with_capacity(n_points);
    let mut decs = Vec::with_capacity(n_points);

    if n_points == 0 {
        return SkySamples { ras, decs };
    }

    // Golden angle in radians
    let golden_angle = PI * (3.0 - 5.0_f64.sqrt());
    let (sin_golden, cos_golden) = golden_angle.sin_cos();
    let mut sin_phi = 0.0;
    let mut cos_phi = 1.0;
    let n = n_points as f64;

    for i in 0..n_points {
        let k = i as f64 + 0.5;
        let z = 1.0 - (2.0 * k) / n;
        let radius_xy = (1.0 - z * z).sqrt();

        let x = radius_xy * cos_phi;
        let y = radius_xy * sin_phi;

        let mut ra_deg = y.atan2(x).to_degrees();
        if ra_deg < 0.0 {
            ra_deg += 360.0;
        }
        let dec_deg = z.clamp(-1.0, 1.0).asin().to_degrees();

        ras.push(ra_deg);
        decs.push(dec_deg);

        // Advance phi by golden_angle using a cheap rotation recurrence.
        if i + 1 < n_points {
            let next_cos = cos_phi * cos_golden - sin_phi * sin_golden;
            let next_sin = sin_phi * cos_golden + cos_phi * sin_golden;
            cos_phi = next_cos;
            sin_phi = next_sin;
        }
    }

    SkySamples { ras, decs }
}

fn cached_fibonacci_sphere_radec(n_points: usize) -> Arc<SkySamples> {
    {
        let cache = fibonacci_samples_cache()
            .read()
            .expect("fibonacci sample cache lock poisoned");
        if let Some(samples) = cache.get(&n_points) {
            return Arc::clone(samples);
        }
    }

    let new_samples = Arc::new(fibonacci_sphere_radec(n_points));
    let mut cache = fibonacci_samples_cache()
        .write()
        .expect("fibonacci sample cache lock poisoned");
    Arc::clone(
        cache
            .entry(n_points)
            .or_insert_with(|| Arc::clone(&new_samples)),
    )
}

pub(super) fn instantaneous_field_of_regard_impl<F>(
    py: Python,
    ephemeris: Py<PyAny>,
    time: Option<&Bound<PyAny>>,
    index: Option<usize>,
    n_points: usize,
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

    let violated = if let Ok(ephem) = bound.extract::<PyRef<TLEEphemeris>>() {
        evaluator.in_constraint_batch(
            &*ephem as &dyn EphemerisBase,
            &sky_samples.ras,
            &sky_samples.decs,
            Some(&[eval_index]),
        )?
    } else if let Ok(ephem) = bound.extract::<PyRef<SPICEEphemeris>>() {
        evaluator.in_constraint_batch(
            &*ephem as &dyn EphemerisBase,
            &sky_samples.ras,
            &sky_samples.decs,
            Some(&[eval_index]),
        )?
    } else if let Ok(ephem) = bound.extract::<PyRef<GroundEphemeris>>() {
        evaluator.in_constraint_batch(
            &*ephem as &dyn EphemerisBase,
            &sky_samples.ras,
            &sky_samples.decs,
            Some(&[eval_index]),
        )?
    } else if let Ok(ephem) = bound.extract::<PyRef<OEMEphemeris>>() {
        evaluator.in_constraint_batch(
            &*ephem as &dyn EphemerisBase,
            &sky_samples.ras,
            &sky_samples.decs,
            Some(&[eval_index]),
        )?
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "Unsupported ephemeris type. Expected TLEEphemeris, SPICEEphemeris, GroundEphemeris, or OEMEphemeris",
        ));
    };

    let visible_count = violated
        .column(0)
        .iter()
        .filter(|&&is_violated| !is_violated)
        .count();
    let visible_fraction = visible_count as f64 / n_points as f64;

    Ok(4.0 * PI * visible_fraction)
}
