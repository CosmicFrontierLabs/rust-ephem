//! Apparent solar-system positions for a receiving observer.
//!
//! Work in barycentric coordinates when solving light time: retarding only
//! the geocentric vector omits the Earth's motion and is especially wrong for
//! the Moon. Apply observer aberration only after solving the light path.

use chrono::{DateTime, Utc};
use ndarray::Array2;
use sofars::astro::{ab, ld, ldsun};

use crate::utils::celestial::planetary_almanac;
use crate::utils::config::{AU_TO_KM, EARTH_NAIF_ID, SUN_NAIF_ID};
use crate::utils::naif_ids::parse_body_identifier;
use crate::utils::time_utils::chrono_to_epoch;

const LIGHT_SPEED_KM_S: f64 = 299_792.458;
const LIGHT_TIME_TOLERANCE_S: f64 = 1e-8;
const MAX_LIGHT_TIME_ITERATIONS: usize = 12;
// Match Astropy's finite-distance solar deflection, including its limiter
// near the solar centre (astropy.coordinates.builtin_frames.utils.atciqz).
const SOLAR_DEFLECTION_LIMIT: f64 = 1e-6;
const MIN_SUN_SOURCE_DISTANCE_AU: f64 = 1e-10;

fn norm(v: &[f64; 3]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// The apparent-place correction of a distant target for one observer state:
/// maps a catalogue ICRS direction to the proper direction in which the moving
/// observer receives its light (solar light deflection, then aberration from
/// the observer's barycentric velocity). The counterpart, for targets, of
/// what `apparent_body_positions` applies to solar-system bodies.
#[derive(Clone, Copy, Debug)]
pub struct ObserverAberration {
    beta: [f64; 3],
    lorentz_inverse: f64,
    sun_to_observer: [f64; 3],
    sun_distance_au: f64,
}

impl ObserverAberration {
    /// Correct a catalogue unit vector (GCRS/ICRS axes).
    pub fn apply(&self, direction: [f64; 3]) -> [f64; 3] {
        let natural = ldsun(direction, self.sun_to_observer, self.sun_distance_au);
        ab(
            &natural,
            &self.beta,
            self.sun_distance_au,
            self.lorentz_inverse,
        )
    }

    /// Correct a catalogue (RA, Dec) in degrees; RA is returned in [0, 360).
    pub fn apply_radec_deg(&self, ra_deg: f64, dec_deg: f64) -> (f64, f64) {
        let (ra, dec) = (ra_deg.to_radians(), dec_deg.to_radians());
        let p = self.apply([dec.cos() * ra.cos(), dec.cos() * ra.sin(), dec.sin()]);
        (
            p[1].atan2(p[0]).to_degrees().rem_euclid(360.0),
            p[2].atan2(p[0].hypot(p[1])).to_degrees(),
        )
    }
}

/// Spacing of the SPICE samples of the Sun's barycentric velocity. It changes
/// by under a millimetre per second in a day, so linear interpolation between
/// samples is good to a microarcsecond of aberration.
const SUN_VELOCITY_KNOT_SECONDS: f64 = 86_400.0;

/// Barycentric velocity of the Earth (km/s), one row per time: the one
/// quantity the apparent place of the Earth, Sun and Moon needs beyond the
/// geocentric states an ephemeris already holds.
///
/// It is the Sun's barycentric velocity minus the Sun's geocentric velocity
/// (`sun`, (N, 6) geocentric states). The first is a few metres per second and
/// nearly constant, so it is read from SPICE once a day across the span of
/// `times` (which may be in any order) rather than once per time.
pub fn earth_barycentric_velocities(
    times: &[DateTime<Utc>],
    sun: &Array2<f64>,
    spice_kernel: Option<&str>,
) -> Result<Array2<f64>, String> {
    use anise::prelude::Frame;

    if sun.dim() != (times.len(), 6) {
        return Err("Apparent evaluation needs one (N, 6) Sun state per time".into());
    }
    let mut out = Array2::zeros((times.len(), 3));
    let (Some(first), Some(last)) = (times.iter().min(), times.iter().max()) else {
        return Ok(out);
    };
    let almanac = planetary_almanac(spice_kernel)?;
    let barycentre = Frame::from_ephem_j2000(0);
    let span = (*last - *first).num_microseconds().unwrap_or(i64::MAX) as f64 / 1e6;
    let intervals = (span / SUN_VELOCITY_KNOT_SECONDS).ceil().max(1.0) as usize;
    let step = span / intervals as f64;
    let knots = (0..=intervals)
        .map(|k| {
            let dt = *first + chrono::Duration::microseconds((k as f64 * step * 1e6) as i64);
            let epoch = chrono_to_epoch(&dt).to_time_scale(hifitime::TimeScale::TAI);
            almanac
                .translate_geometric(Frame::from_ephem_j2000(SUN_NAIF_ID), barycentre, epoch)
                .map(|state| state.velocity_km_s)
                .map_err(|e| {
                    format!(
                        "SPICE cannot compute the Sun's barycentric velocity at {dt}: {e}. \
                         Apparent evaluation needs a planetary kernel covering the times."
                    )
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    for (i, time) in times.iter().enumerate() {
        let elapsed = (*time - *first).num_microseconds().unwrap_or(0) as f64 / 1e6;
        let x = if step > 0.0 { elapsed / step } else { 0.0 };
        let k = (x.floor() as usize).min(intervals - 1);
        let f = x - k as f64;
        for j in 0..3 {
            let sun_barycentric = knots[k][j] * (1.0 - f) + knots[k + 1][j] * f;
            out[[i, j]] = sun_barycentric - sun[[i, j + 3]];
        }
    }
    Ok(out)
}

/// Apparent places from geocentric states (GCRS axes, km and km/s, one row
/// per time) for constraint evaluation, without a SPICE query per body.
///
/// `observer` and `sun` are (N, 6); `earth_velocity` is (N, 3) barycentric
/// (`earth_barycentric_velocities`). Returns, for each of `bodies` ((N, 6)
/// geocentric states; `None` is the Earth itself), the (N, 3) observer-centred
/// apparent positions, and the correction of each row for distant targets.
///
/// The light path is solved to first order in the light time, using the body's
/// barycentric velocity at reception. The neglected acceleration term is a few
/// centimetres for the Moon and less for the Earth and Sun, far below a
/// microarcsecond; deflection and aberration are those of
/// `apparent_body_positions`.
pub fn apparent_from_states(
    observer: &Array2<f64>,
    sun: &Array2<f64>,
    earth_velocity: &Array2<f64>,
    bodies: &[Option<&Array2<f64>>],
) -> Result<(Vec<Array2<f64>>, Vec<ObserverAberration>), String> {
    let n = observer.nrows();
    if observer.ncols() != 6 || sun.dim() != (n, 6) || earth_velocity.dim() != (n, 3) {
        return Err("Apparent evaluation needs (N, 6) observer and Sun states".into());
    }
    if bodies.iter().flatten().any(|body| body.dim() != (n, 6)) {
        return Err("Apparent evaluation needs (N, 6) body states".into());
    }
    if !observer.iter().all(|x| x.is_finite()) {
        return Err("Apparent evaluation requires finite observer states".into());
    }
    let mut positions = vec![Array2::zeros((n, 3)); bodies.len()];
    let mut corrections = Vec::with_capacity(n);
    for i in 0..n {
        let beta: [f64; 3] = std::array::from_fn(|j| {
            (earth_velocity[[i, j]] + observer[[i, j + 3]]) / LIGHT_SPEED_KM_S
        });
        let beta_squared = beta.iter().map(|x| x * x).sum::<f64>();
        if beta_squared >= 1.0 {
            return Err("Observer barycentric speed must be less than the speed of light".into());
        }
        let sun_to_observer: [f64; 3] = std::array::from_fn(|j| observer[[i, j]] - sun[[i, j]]);
        let sun_distance = norm(&sun_to_observer);
        if sun_distance == 0.0 {
            return Err("Apparent coordinates are undefined at the solar centre".into());
        }
        let correction = ObserverAberration {
            beta,
            lorentz_inverse: (1.0 - beta_squared).sqrt(),
            sun_to_observer: sun_to_observer.map(|x| x / sun_distance),
            sun_distance_au: sun_distance / AU_TO_KM,
        };
        for (body, out) in bodies.iter().zip(positions.iter_mut()) {
            // Geocentric position and barycentric velocity of the body at reception.
            let (position, velocity): ([f64; 3], [f64; 3]) = match body {
                Some(states) => (
                    std::array::from_fn(|j| states[[i, j]]),
                    std::array::from_fn(|j| earth_velocity[[i, j]] + states[[i, j + 3]]),
                ),
                None => ([0.0; 3], std::array::from_fn(|j| earth_velocity[[i, j]])),
            };
            let geometric: [f64; 3] = std::array::from_fn(|j| position[j] - observer[[i, j]]);
            let mut relative = geometric;
            for _ in 0..3 {
                let light_time = norm(&relative) / LIGHT_SPEED_KM_S;
                relative = std::array::from_fn(|j| geometric[j] - velocity[j] * light_time);
            }
            let distance = norm(&relative);
            if distance == 0.0 {
                return Err("Apparent direction is undefined for a body at the observer".into());
            }
            let direction = relative.map(|x| x / distance);
            let sun_to_source: [f64; 3] = std::array::from_fn(|j| sun_to_observer[j] + relative[j]);
            let sun_source_distance = norm(&sun_to_source);
            let source_direction = if sun_source_distance > MIN_SUN_SOURCE_DISTANCE_AU * AU_TO_KM {
                sun_to_source.map(|x| x / sun_source_distance)
            } else {
                direction
            };
            let natural = ld(
                1.0,
                direction,
                source_direction,
                correction.sun_to_observer,
                correction.sun_distance_au,
                SOLAR_DEFLECTION_LIMIT,
            );
            let proper = ab(
                &natural,
                &correction.beta,
                correction.sun_distance_au,
                correction.lorentz_inverse,
            );
            for j in 0..3 {
                out[[i, j]] = proper[j] * distance;
            }
        }
        corrections.push(correction);
    }
    Ok((positions, corrections))
}

/// Observer-centred apparent positions in GCRS axes (km), without velocities.
/// Uses converged reception light time, finite-distance solar deflection,
/// and relativistic aberration from the observer's barycentric velocity.
pub fn apparent_body_positions(
    times: &[DateTime<Utc>],
    observer: &Array2<f64>,
    body: &str,
    spice_kernel: Option<&str>,
) -> Result<Array2<f64>, String> {
    use anise::prelude::Frame;

    if observer.nrows() != times.len() || observer.ncols() != 6 {
        return Err("Apparent positions require one six-component observer state per time".into());
    }
    if !observer.iter().all(|x| x.is_finite()) {
        return Err("Apparent positions require finite observer states".into());
    }
    let target_id = parse_body_identifier(body)
        .ok_or_else(|| format!("Unknown body identifier '{body}' for apparent coordinates"))?;
    let almanac = planetary_almanac(spice_kernel)?;
    let barycentre = Frame::from_ephem_j2000(0);
    let mut out = Array2::zeros((times.len(), 3));

    for (i, dt) in times.iter().enumerate() {
        // hifitime subtracts durations in the epoch's stored time scale.
        // Use continuous TAI so light time remains physical elapsed time
        // when reception and emission straddle a UTC leap second.
        let reception = chrono_to_epoch(dt).to_time_scale(hifitime::TimeScale::TAI);
        let state = |id, epoch| {
            almanac
                .translate_geometric(Frame::from_ephem_j2000(id), barycentre, epoch)
                .map_err(|e| {
                    format!(
                        "SPICE cannot compute apparent body {body} at {dt}: {e}. \
                             The kernel must cover the reception and emission times and \
                             contain the target, Earth, and Sun relative to the barycentre."
                    )
                })
        };
        let earth = state(EARTH_NAIF_ID, reception)?;
        let sun = state(SUN_NAIF_ID, reception)?;
        let observer_position: [f64; 3] =
            std::array::from_fn(|j| earth.radius_km[j] + observer[[i, j]]);
        let beta: [f64; 3] = std::array::from_fn(|j| {
            (earth.velocity_km_s[j] + observer[[i, j + 3]]) / LIGHT_SPEED_KM_S
        });
        let beta_squared = beta.iter().map(|x| x * x).sum::<f64>();
        if beta_squared >= 1.0 {
            return Err("Observer barycentric speed must be less than the speed of light".into());
        }

        let mut light_time = 0.0;
        let mut relative = [0.0; 3];
        let mut converged = false;
        for _ in 0..MAX_LIGHT_TIME_ITERATIONS {
            let emission = reception - hifitime::Duration::from_seconds(light_time);
            let target = state(target_id, emission)?;
            relative = std::array::from_fn(|j| target.radius_km[j] - observer_position[j]);
            let next = norm(&relative) / LIGHT_SPEED_KM_S;
            if (next - light_time).abs() < LIGHT_TIME_TOLERANCE_S {
                converged = true;
                break;
            }
            light_time = next;
        }
        if !converged {
            return Err(format!(
                "Light time failed to converge for body '{body}' at {dt}"
            ));
        }
        let distance = norm(&relative);
        // Barycentric additions/subtractions can leave a few ulps when a
        // body's own ephemeris is used as the observer. Such a residual
        // must not be normalized into an arbitrary apparent direction.
        if distance <= 4.0 * f64::EPSILON * norm(&observer_position) {
            return Err(format!(
                "Apparent direction is undefined for body '{body}' at the observer"
            ));
        }
        let direction = relative.map(|x| x / distance);
        let sun_to_observer: [f64; 3] =
            std::array::from_fn(|j| observer_position[j] - sun.radius_km[j]);
        let sun_distance = norm(&sun_to_observer);
        if sun_distance == 0.0 {
            return Err("Apparent coordinates are undefined at the solar centre".into());
        }
        let sun_direction = sun_to_observer.map(|x| x / sun_distance);
        let sun_to_source: [f64; 3] = std::array::from_fn(|j| sun_to_observer[j] + relative[j]);
        let sun_source_distance = norm(&sun_to_source);
        let source_direction = if sun_source_distance > MIN_SUN_SOURCE_DISTANCE_AU * AU_TO_KM {
            sun_to_source.map(|x| x / sun_source_distance)
        } else {
            direction
        };
        let natural = ld(
            1.0,
            direction,
            source_direction,
            sun_direction,
            sun_distance / AU_TO_KM,
            SOLAR_DEFLECTION_LIMIT,
        );
        let proper = ab(
            &natural,
            &beta,
            sun_distance / AU_TO_KM,
            (1.0 - beta_squared).sqrt(),
        );
        for j in 0..3 {
            out[[i, j]] = proper[j] * distance;
        }
    }
    Ok(out)
}
