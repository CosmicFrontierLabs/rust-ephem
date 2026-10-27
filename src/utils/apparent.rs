//! Apparent solar-system positions for a receiving observer.
//!
//! Work in barycentric coordinates when solving light time: retarding only
//! the geocentric vector omits the Earth's motion and is especially wrong for
//! the Moon. Apply observer aberration only after solving the light path.

use chrono::{DateTime, Utc};
use ndarray::Array2;
use sofars::astro::{ab, ld};

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
        let reception = chrono_to_epoch(dt);
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
