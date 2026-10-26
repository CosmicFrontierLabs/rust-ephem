//! Single-waypoint routing at one ephemeris sample. Geometry stays in Rust;
//! every mask uses the same evaluator and roll context as batch evaluation.
use crate::constraints::core::{ConstraintEvaluator, EvaluationRoll};
use crate::constraints::diagonal::SampleEphemeris;
use crate::ephemeris::ephemeris_common::EphemerisBase;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

type Vec3 = [f64; 3];
const NO_ROTATION: f64 = 1e-6;

pub(super) fn sample_index(
    ephemeris: &dyn EphemerisBase,
    time: Option<&Bound<'_, PyAny>>,
    index: Option<usize>,
) -> PyResult<usize> {
    let times = ephemeris
        .data()
        .times
        .as_ref()
        .ok_or_else(|| PyValueError::new_err("ephemeris has no timestamps"))?;
    match (time, index) {
        (None, Some(i)) if i < times.len() => Ok(i),
        (None, Some(_)) => Err(PyValueError::new_err("index out of bounds")),
        (Some(time), None) => {
            let instant = if time.is_instance_of::<pyo3::types::PyDateTime>() {
                let utc_time;
                let time = if !time.call_method0("utcoffset")?.is_none() {
                    let utc = time
                        .py()
                        .import("datetime")?
                        .getattr("timezone")?
                        .getattr("utc")?;
                    utc_time = time.call_method1("astimezone", (utc,))?;
                    &utc_time
                } else {
                    time
                };
                crate::utils::time_utils::python_datetime_to_utc(time)?
            } else {
                let seconds: f64 = time.extract().map_err(|_| {
                    PyValueError::new_err("time must be a datetime or Unix timestamp")
                })?;
                if !seconds.is_finite() {
                    return Err(PyValueError::new_err("time must be finite"));
                }
                chrono::DateTime::from_timestamp(
                    seconds.floor() as i64,
                    ((seconds - seconds.floor()) * 1e9) as u32,
                )
                .ok_or_else(|| PyValueError::new_err("time is out of range"))?
            };
            times
                .partition_point(|t| *t <= instant)
                .checked_sub(1)
                .ok_or_else(|| PyValueError::new_err("time precedes ephemeris"))
        }
        _ => Err(PyValueError::new_err(
            "provide exactly one of time or index",
        )),
    }
}

/// Result of sampled single-waypoint routing. A missing waypoint with
/// `violated=True` means routing failed, not that the direct slew is safe.
#[pyclass(frozen, get_all)]
#[derive(Debug, Default)]
pub struct RouteResult {
    pub violated: bool,
    pub first_violation: Option<f64>,
    pub waypoint: Option<(f64, f64)>,
    pub detour_deg: Option<f64>,
}

#[pymethods]
impl RouteResult {
    fn __repr__(&self) -> String {
        let optional_float =
            |value: Option<f64>| value.map_or_else(|| "None".to_owned(), |value| value.to_string());
        let waypoint = self
            .waypoint
            .map_or_else(|| "None".to_owned(), |(ra, dec)| format!("({ra}, {dec})"));
        format!(
            "RouteResult(violated={}, first_violation={}, waypoint={}, detour_deg={})",
            if self.violated { "True" } else { "False" },
            optional_float(self.first_violation),
            waypoint,
            optional_float(self.detour_deg)
        )
    }
}

fn dot(a: Vec3, b: Vec3) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize(v: Vec3) -> Vec3 {
    let norm = dot(v, v).sqrt();
    v.map(|x| x / norm)
}

fn angle(a: Vec3, b: Vec3) -> f64 {
    dot(a, b).clamp(-1.0, 1.0).acos()
}

fn normal(a: Vec3, b: Vec3) -> Vec3 {
    let n = cross(a, b);
    if dot(n, n).sqrt() >= 1e-12 {
        return normalize(n);
    }
    let mut axis = [0.0; 3];
    let mut smallest = 0;
    for i in 1..3 {
        if a[i].abs() < a[smallest].abs() {
            smallest = i;
        }
    }
    axis[smallest] = 1.0;
    normalize(cross(a, axis))
}

fn radec(v: Vec3) -> (f64, f64) {
    let v = normalize(v);
    (
        v[1].atan2(v[0]).to_degrees().rem_euclid(360.0),
        v[2].clamp(-1.0, 1.0).asin().to_degrees(),
    )
}

struct Arc {
    a: Vec3,
    b: Vec3,
    theta: f64,
    n: Vec3,
}

impl Arc {
    fn new(a: Vec3, b: Vec3) -> Self {
        Self {
            a,
            b,
            theta: angle(a, b),
            n: normal(a, b),
        }
    }

    fn point(&self, t: f64) -> Vec3 {
        if t == 0.0 {
            return self.a;
        }
        if t == 1.0 {
            return self.b;
        }
        if self.theta < 0.01 {
            return normalize(std::array::from_fn(|i| {
                self.a[i] + t * (self.b[i] - self.a[i])
            }));
        }
        if self.theta.cos() < 0.0 && self.theta.sin().abs() < 1e-7 {
            // The reference's sin(theta) denominator is undefined at antipodes.
            // Use the prescribed plane, including its deterministic axis fallback.
            let tangent = cross(self.n, self.a);
            return normalize(std::array::from_fn(|i| {
                (t * self.theta).cos() * self.a[i] + (t * self.theta).sin() * tangent[i]
            }));
        }
        std::array::from_fn(|i| {
            (((1.0 - t) * self.theta).sin() * self.a[i] + (t * self.theta).sin() * self.b[i])
                / self.theta.sin()
        })
    }

    fn points(&self, samples: usize, endpoints: bool) -> Vec<Vec3> {
        let range = if endpoints {
            0..samples + 1
        } else {
            1..samples
        };
        range
            .map(|i| self.point(i as f64 / samples as f64))
            .collect()
    }
}

fn validate<F>(a: Vec3, b: Vec3, w: Vec3, samples: usize, check: &mut F) -> PyResult<bool>
where
    F: FnMut(&[Vec3]) -> PyResult<Vec<bool>>,
{
    if check(&[w])?[0] {
        return Ok(false);
    }
    for arc in [Arc::new(a, w), Arc::new(w, b)] {
        if arc.theta > NO_ROTATION
            && samples > 1
            && check(&arc.points(samples, false))?
                .iter()
                .any(|&blocked| blocked)
        {
            return Ok(false);
        }
    }
    Ok(true)
}

fn candidates<F>(
    arc: &Arc,
    c: Vec3,
    direction: Vec3,
    offset: f64,
    samples: usize,
    check: &mut F,
) -> PyResult<Option<(Vec3, f64)>>
where
    F: FnMut(&[Vec3]) -> PyResult<Vec<bool>>,
{
    let mut candidates = [1.0, -1.0].map(|sign| {
        let w = normalize(std::array::from_fn(|i| {
            offset.cos() * c[i] + sign * offset.sin() * direction[i]
        }));
        let length = angle(arc.a, w).to_degrees() + angle(w, arc.b).to_degrees();
        (w, length)
    });
    if candidates[1].1 < candidates[0].1 {
        candidates.swap(0, 1);
    }
    for (w, length) in candidates {
        if validate(arc.a, arc.b, w, samples, check)? {
            return Ok(Some((w, length)));
        }
    }
    Ok(None)
}

fn search<F>(
    a: Vec3,
    b: Vec3,
    samples: usize,
    margin_deg: f64,
    grow_margin: bool,
    target_dependent: bool,
    mut check: F,
) -> PyResult<RouteResult>
where
    F: FnMut(&[Vec3]) -> PyResult<Vec<bool>>,
{
    if !target_dependent {
        let blocked = check(&[a])?[0];
        return Ok(RouteResult {
            violated: blocked,
            first_violation: blocked.then_some(0.0),
            ..RouteResult::default()
        });
    }
    let arc = Arc::new(a, b);
    let points = arc.points(samples, true);
    let flags = check(&points)?;
    let Some(first) = flags.iter().position(|&blocked| blocked) else {
        return Ok(RouteResult::default());
    };
    let mut result = RouteResult {
        violated: true,
        first_violation: Some(first as f64 / samples as f64),
        ..RouteResult::default()
    };
    if flags[0] || flags[samples] || arc.theta < NO_ROTATION {
        return Ok(result);
    }
    let c = normalize(points[first]);
    // Preserve coast-sim's original (in-plane) direction in faithful mode.
    let mut found = candidates(
        &arc,
        c,
        normalize(cross(arc.n, c)),
        margin_deg.to_radians(),
        samples,
        &mut check,
    )?;
    if found.is_none() && grow_margin {
        // A real detour must leave the original plane. Increase the offset
        // geometrically, bounding work even for extremely small margins. Include
        // 90 degrees: for a single <=90-degree cone one of +/-n clears both legs
        // when both endpoints are outside. Other geometries may remain blocked.
        let mut offset = margin_deg;
        loop {
            found = candidates(&arc, c, arc.n, offset.to_radians(), samples, &mut check)?;
            if found.is_some() || offset >= 90.0 {
                break;
            }
            offset = (2.0 * offset).min(90.0);
        }
    }
    if let Some((w, length)) = found {
        result.waypoint = Some(radec(w));
        result.detour_deg = Some(length);
    }
    Ok(result)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn route_around(
    evaluator: &dyn ConstraintEvaluator,
    ephemeris: &dyn EphemerisBase,
    start_ra: f64,
    start_dec: f64,
    end_ra: f64,
    end_dec: f64,
    index: usize,
    target_roll: Option<f64>,
    n_roll_samples: usize,
    samples: usize,
    margin_deg: f64,
    grow_margin: bool,
) -> PyResult<RouteResult> {
    if samples == 0 || samples == usize::MAX {
        return Err(PyValueError::new_err(
            "samples must be positive and less than usize::MAX",
        ));
    }
    if !margin_deg.is_finite() || margin_deg <= 0.0 || margin_deg > 90.0 {
        return Err(PyValueError::new_err(
            "margin_deg must be finite and in (0, 90]",
        ));
    }
    if n_roll_samples == 0 {
        return Err(PyValueError::new_err("n_roll_samples must be positive"));
    }
    if target_roll.is_some_and(|r| !r.is_finite()) {
        return Err(PyValueError::new_err("target_roll must be finite"));
    }
    for (ra, dec) in [(start_ra, start_dec), (end_ra, end_dec)] {
        if !ra.is_finite() || !dec.is_finite() || !(-90.0..=90.0).contains(&dec) {
            return Err(PyValueError::new_err(
                "pointings must be finite with declination in [-90, 90]",
            ));
        }
    }
    let a = normalize(crate::utils::vector_math::radec_to_unit_vector(
        start_ra, start_dec,
    ));
    let b = normalize(crate::utils::vector_math::radec_to_unit_vector(
        end_ra, end_dec,
    ));
    // Snapshot just this sample, so per-arc geometry work is independent of the
    // ephemeris duration. Reuse it throughout the search, including roll sweeps.
    let sample = SampleEphemeris::new(ephemeris, index);
    search(
        a,
        b,
        samples,
        margin_deg,
        grow_margin,
        evaluator.is_target_dependent(),
        |points| {
            let (ras, decs): (Vec<_>, Vec<_>) = points.iter().map(|&v| radec(v)).unzip();
            let flags = match target_roll {
                Some(roll) => evaluator.in_constraint_batch_at_roll(
                    &sample,
                    &ras,
                    &decs,
                    Some(&[0]),
                    EvaluationRoll::Fixed(roll),
                )?,
                None => evaluator.in_constraint_batch_constrained_at_every_roll(
                    &sample,
                    &ras,
                    &decs,
                    Some(&[0]),
                    n_roll_samples,
                )?,
            };
            Ok(flags.column(0).to_vec())
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn time_only_constraints_evaluate_once_without_arc_sampling() {
        for blocked in [false, true] {
            let mut calls = 0;
            let result = search(
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                500,
                5.0,
                true,
                false,
                |points| {
                    calls += 1;
                    assert_eq!(points.len(), 1);
                    Ok(vec![blocked])
                },
            )
            .unwrap();
            assert_eq!(calls, 1);
            assert_eq!(result.violated, blocked);
            assert_eq!(result.first_violation, blocked.then_some(0.0));
            assert!(result.waypoint.is_none());
        }
    }

    #[test]
    fn antipodal_interpolation_uses_the_prescribed_plane() {
        let arc = Arc::new([1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]);
        assert_eq!(arc.n, [0.0, 0.0, 1.0]);
        let mid = arc.point(0.5);
        assert!(mid[0].abs() < 1e-15);
        assert!((mid[1] - 1.0).abs() < 1e-15);
        for v in arc.points(50, true) {
            assert!((dot(v, v) - 1.0).abs() < 1e-14);
        }
    }

    #[test]
    fn near_antipodes_and_small_arcs_remain_normalized() {
        for theta in [0.0_f64, 1e-8, 0.005, std::f64::consts::PI - 1e-9] {
            let arc = Arc::new([1.0, 0.0, 0.0], [theta.cos(), theta.sin(), 0.0]);
            for v in arc.points(50, true) {
                assert!((dot(v, v) - 1.0).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn a_failed_search_preserves_the_direct_crossing() {
        let result = search(
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            50,
            5.0,
            true,
            true,
            |points| {
                Ok(points
                    .iter()
                    .map(|v| v[0] > 0.1 && v[1] > 0.1 || v[2].abs() > 0.01)
                    .collect())
            },
        )
        .unwrap();
        assert!(result.violated);
        assert_eq!(result.first_violation, Some(0.08));
        assert!(result.waypoint.is_none());
        assert!(result.detour_deg.is_none());
    }
}
