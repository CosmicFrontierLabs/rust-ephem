use chrono::{DateTime, Datelike, Timelike, Utc};
use erfa::earth::position_velocity_00;
use erfa::prenut::precession_matrix_06;
use erfa::vectors_and_matrices::mat_mul_pvec;
use lru::LruCache;
use ndarray::{s, Array1, Array2};
use once_cell::sync::Lazy;
use sofars::astro::atco13;
use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::Mutex;

use crate::ephemeris::ephemeris_common::EphemerisBase;
use crate::utils::conversions::{convert_frames, Frame};
use crate::utils::geo::ecef_to_geodetic_deg;
use crate::utils::math_utils::transpose_matrix;
use crate::utils::time_utils::{datetime_to_jd_tt, datetime_to_jd_utc};
use crate::utils::{eop_provider, ut1_provider};
use crate::{is_planetary_ephemeris_initialized, utils::config::*};

// Frame cache: Maps NAIF ID to pre-computed Frame object
static FRAME_CACHE: Lazy<Mutex<HashMap<i32, anise::prelude::Frame>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

// Position cache: Maps (body_id, center_id, epoch_ns) to cached position/velocity vector
// Using an LRU cache with max 1000 entries (each entry is ~64 bytes, ~64 KB total)
// This provides ~1-2 orders of magnitude speedup for repeated queries
type PositionCacheKey = (i32, i32, i64);
type PositionCacheValue = [f64; 6]; // [x, y, z, vx, vy, vz]

static POSITION_CACHE: Lazy<Mutex<LruCache<PositionCacheKey, PositionCacheValue>>> =
    Lazy::new(|| Mutex::new(LruCache::new(NonZeroUsize::new(2000).unwrap())));

/// Get a position from cache if available
#[inline]
fn get_cached_position(body_id: i32, center_id: i32, epoch_ns: i64) -> Option<[f64; 6]> {
    let mut cache = POSITION_CACHE.lock().unwrap();
    cache.get(&(body_id, center_id, epoch_ns)).copied()
}

/// Store a position in the cache
#[inline]
fn cache_position(body_id: i32, center_id: i32, epoch_ns: i64, pos: [f64; 6]) {
    let mut cache = POSITION_CACHE.lock().unwrap();
    cache.put((body_id, center_id, epoch_ns), pos);
}

/// Get or create a cached Frame object for a given NAIF ID
/// This avoids repeated Frame::from_ephem_j2000 calls which have overhead
#[inline]
fn get_cached_frame(naif_id: i32) -> anise::prelude::Frame {
    use anise::prelude::Frame as AniseFrame;

    let mut cache = FRAME_CACHE.lock().unwrap();
    if let Some(&frame) = cache.get(&naif_id) {
        frame
    } else {
        let frame = AniseFrame::from_ephem_j2000(naif_id);
        cache.insert(naif_id, frame);
        frame
    }
}

/// Calculate Sun positions for multiple timestamps
/// Returns Array2 with shape (N, 6) containing [x, y, z, vx, vy, vz] for each timestamp
pub fn calculate_sun_positions_erfa(times: &[DateTime<Utc>]) -> Array2<f64> {
    let n = times.len();
    let mut out = Array2::<f64>::zeros((n, 6));

    // uses AU_TO_KM, AU_PER_DAY_TO_KM_PER_SEC from config

    for (i, dt) in times.iter().enumerate() {
        let (jd_tt1, jd_tt2) = datetime_to_jd_tt(dt);

        let (_warning, pvh, _pvb) = position_velocity_00(jd_tt1, jd_tt2);

        // Sun position is negative of Earth's heliocentric position
        let mut row = out.row_mut(i);
        row[0] = -pvh[0][0] * AU_TO_KM;
        row[1] = -pvh[0][1] * AU_TO_KM;
        row[2] = -pvh[0][2] * AU_TO_KM;
        row[3] = -pvh[1][0] * AU_PER_DAY_TO_KM_PER_SEC;
        row[4] = -pvh[1][1] * AU_PER_DAY_TO_KM_PER_SEC;
        row[5] = -pvh[1][2] * AU_PER_DAY_TO_KM_PER_SEC;
    }

    out
}

/// Calculate Moon positions for multiple timestamps
/// Returns Array2 with shape (N, 6) containing [x, y, z, vx, vy, vz] for each timestamp
pub fn calculate_moon_positions_meeus(times: &[DateTime<Utc>]) -> Array2<f64> {
    let n = times.len();
    let mut out = Array2::<f64>::zeros((n, 6));

    // uses GM_EARTH, JD_J2000, DAYS_PER_CENTURY from config

    for (i, dt) in times.iter().enumerate() {
        // Meeus formulae use dynamical time (TT)
        let (jd_tt1, jd_tt2) = datetime_to_jd_tt(dt);
        let jd = jd_tt1 + jd_tt2;

        // Julian centuries from J2000.0 in TT
        let t = (jd - JD_J2000) / DAYS_PER_CENTURY;
        let t_sq = t * t;
        let t_cb = t_sq * t;
        let t_qt = t_cb * t;

        // Meeus formulae for Moon's mean longitude, elongation, anomaly, etc.
        let l_prime = 218.3164477 + 481267.88123421 * t - 0.0015786 * t_sq + t_cb / 538841.0
            - t_qt / 65194000.0;
        let d = 297.8501921 + 445267.1114034 * t - 0.0018819 * t_sq + t_cb / 545868.0
            - t_qt / 113065000.0;
        let m = 357.5291092 + 35999.0502909 * t - 0.0001536 * t_sq + t_cb / 24490000.0;
        let m_prime = 134.9633964 + 477198.8675055 * t + 0.0087414 * t_sq + t_cb / 69699.0
            - t_qt / 14712000.0;
        let f = 93.2720950 + 483202.0175233 * t - 0.0036539 * t_sq - t_cb / 3526000.0
            + t_qt / 863310000.0;

        // Convert to radians
        let d_rad = d.to_radians();
        let m_rad = m.to_radians();
        let m_prime_rad = m_prime.to_radians();
        let f_rad = f.to_radians();

        // Pre-compute commonly used multiples
        let d_rad_2 = 2.0 * d_rad;
        let d_rad_4 = 4.0 * d_rad;
        let m_prime_rad_2 = 2.0 * m_prime_rad;
        let m_prime_rad_3 = 3.0 * m_prime_rad;
        let f_rad_2 = 2.0 * f_rad;
        let m_rad_2 = 2.0 * m_rad;

        // Cached trigonometric values
        let sin_m_prime = m_prime_rad.sin();
        let cos_m_prime = m_prime_rad.cos();
        let sin_d = d_rad.sin();
        let sin_d_2 = d_rad_2.sin();
        let cos_d_2 = d_rad_2.cos();
        let sin_m = m_rad.sin();
        let cos_m = m_rad.cos();
        let sin_f = f_rad.sin();
        let sin_f_2 = f_rad_2.sin();
        let cos_f_2 = f_rad_2.cos();

        // Calculate longitude (corrections in degrees according to Meeus Chapter 47)
        // Using significant terms (> 10000 micro-degrees or 0.01 degrees)
        let mut lon = l_prime
            + 6.288774 * sin_m_prime
            + 1.274027 * (d_rad_2 - m_prime_rad).sin()
            + 0.658314 * sin_d_2
            + 0.213618 * m_prime_rad_2.sin()
            - 0.185116 * sin_m
            - 0.114332 * sin_f_2
            + 0.058793 * (d_rad_2 - m_prime_rad_2).sin()
            + 0.057066 * (d_rad_2 - m_rad - m_prime_rad).sin()
            + 0.053322 * (d_rad_2 + m_prime_rad).sin()
            + 0.045758 * (d_rad_2 - m_rad).sin()
            - 0.040923 * (m_rad - m_prime_rad).sin()
            - 0.034720 * sin_d
            - 0.030383 * (m_rad + m_prime_rad).sin()
            + 0.015327 * (d_rad_2 - f_rad_2).sin()
            - 0.012528 * (m_prime_rad + f_rad_2).sin()
            + 0.010980 * (m_prime_rad - f_rad_2).sin()
            + 0.010675 * (d_rad_4 - m_prime_rad).sin()
            + 0.010034 * m_prime_rad_3.sin()
            + 0.008548 * (d_rad_4 - m_prime_rad_2).sin()
            - 0.007888 * (d_rad_2 + m_rad - m_prime_rad).sin()
            - 0.006766 * (d_rad_2 + m_rad).sin()
            - 0.005163 * (d_rad - m_prime_rad).sin()
            + 0.004987 * (d_rad + m_rad).sin()
            + 0.004036 * (d_rad_2 - m_rad + m_prime_rad).sin()
            + 0.003994 * (d_rad_2 + m_prime_rad_2).sin()
            + 0.003861 * d_rad_4.sin()
            + 0.003665 * (d_rad_2 - m_prime_rad_3).sin()
            - 0.002689 * (m_rad - m_prime_rad_2).sin()
            - 0.002602 * (d_rad_2 - m_prime_rad + f_rad_2).sin()
            + 0.002390 * (d_rad_2 - m_rad - m_prime_rad_2).sin()
            - 0.002348 * (d_rad + m_prime_rad).sin()
            + 0.002236 * (d_rad_2 - m_rad_2).sin()
            - 0.002120 * (m_rad + m_prime_rad_2).sin()
            - 0.002069 * m_rad_2.sin();

        // Additional correction terms from Meeus (A1, A2, A3)
        let a1 = (119.75 + 131.849 * t).to_radians();
        let a2 = (53.09 + 479264.290 * t).to_radians();
        let a3 = (313.45 + 481266.484 * t).to_radians();
        lon += 0.003958 * a1.sin();
        lon += 0.001962 * (l_prime.to_radians() - f_rad).sin();
        lon += 0.000318 * a2.sin();

        // Calculate latitude (corrections in degrees according to Meeus Chapter 47)
        let mut lat = 5.128122 * sin_f
            + 0.280602 * (m_prime_rad + f_rad).sin()
            + 0.277693 * (m_prime_rad - f_rad).sin()
            + 0.173237 * (d_rad_2 - f_rad).sin()
            + 0.055413 * (d_rad_2 - m_prime_rad + f_rad).sin()
            + 0.046271 * (d_rad_2 - m_prime_rad - f_rad).sin()
            + 0.032573 * (d_rad_2 + f_rad).sin()
            + 0.017198 * (m_prime_rad_2 + f_rad).sin()
            + 0.009266 * (d_rad_2 + m_prime_rad - f_rad).sin()
            + 0.008822 * (m_prime_rad_2 - f_rad).sin()
            + 0.008216 * (d_rad_2 - m_rad - f_rad).sin()
            + 0.004324 * (d_rad_2 - m_prime_rad_2 - f_rad).sin()
            + 0.004200 * (d_rad_2 + m_prime_rad + f_rad).sin()
            - 0.003359 * (d_rad_2 + m_rad - f_rad).sin();

        // Additional latitude correction terms from Meeus
        lat -= 0.002235 * l_prime.to_radians().sin();
        lat += 0.000382 * a3.sin();
        lat += 0.000175 * (a1 - f_rad).sin();
        lat += 0.000175 * (a1 + f_rad).sin();
        lat += 0.000127 * (l_prime.to_radians() - m_prime_rad).sin();
        lat -= 0.000115 * (l_prime.to_radians() + m_prime_rad).sin();

        // Calculate distance (corrections in km according to Meeus Chapter 47)
        let dist = 385000.56
            - 20905.355 * cos_m_prime
            - 3699.111 * (d_rad_2 - m_prime_rad).cos()
            - 2955.968 * cos_d_2
            - 569.925 * m_prime_rad_2.cos()
            + 48.888 * cos_m
            - 3.149 * cos_f_2
            + 246.158 * (d_rad_2 - m_prime_rad_2).cos()
            - 152.138 * (d_rad_2 - m_rad - m_prime_rad).cos()
            - 170.733 * (d_rad_2 + m_prime_rad).cos()
            - 204.586 * (d_rad_2 - m_rad).cos()
            - 129.620 * (m_rad - m_prime_rad).cos()
            + 108.743 * d_rad.cos()
            + 104.755 * (m_rad + m_prime_rad).cos()
            + 10.321 * (d_rad_2 - f_rad_2).cos();

        // Convert to radians for final calculation
        let lon_rad = lon.to_radians();
        let lat_rad = lat.to_radians();

        // Pre-compute sin/cos for final transformation
        let sin_lat = lat_rad.sin();
        let cos_lat = lat_rad.cos();
        let sin_lon = lon_rad.sin();
        let cos_lon = lon_rad.cos();

        // Convert spherical to Cartesian (ecliptic coordinates)
        let x_ecl = dist * cos_lat * cos_lon;
        let y_ecl = dist * cos_lat * sin_lon;
        let z_ecl = dist * sin_lat;

        // Mean obliquity of the ecliptic (J2000.0)
        let epsilon = (23.439291 - 0.0130042 * t).to_radians();
        let cos_epsilon = epsilon.cos();
        let sin_epsilon = epsilon.sin();

        // Rotate from ecliptic to equatorial coordinates (mean equator and equinox of date)
        let x_eq_date = x_ecl;
        let y_eq_date = y_ecl * cos_epsilon - z_ecl * sin_epsilon;
        let z_eq_date = y_ecl * sin_epsilon + z_ecl * cos_epsilon;

        // Approximate velocity using orbital mechanics
        let vel_mag = (GM_EARTH / dist).sqrt();

        // Velocity direction in ecliptic coordinates
        let r_ecl_mag = (x_ecl * x_ecl + y_ecl * y_ecl + z_ecl * z_ecl).sqrt();
        let vx_ecl = vel_mag * (-y_ecl / r_ecl_mag);
        let vy_ecl = vel_mag * (x_ecl / r_ecl_mag);
        let vz_ecl = 0.0;

        // Rotate velocity from ecliptic to equatorial coordinates (mean equator and equinox of date)
        let vx_eq_date = vx_ecl;
        let vy_eq_date = vy_ecl * cos_epsilon - vz_ecl * sin_epsilon;
        let vz_eq_date = vy_ecl * sin_epsilon + vz_ecl * cos_epsilon;

        // Transform from mean equatorial of date to GCRS (J2000) using precession matrix
        let (jd_tt1, jd_tt2) = datetime_to_jd_tt(dt);

        // Get the precession matrix from J2000 to date
        let prec_matrix = precession_matrix_06(jd_tt1, jd_tt2);

        // Transpose to get date to J2000 (for orthogonal matrices, transpose = inverse)
        let prec_matrix_t = transpose_matrix(prec_matrix);

        // Apply precession transformation to get J2000/GCRS coordinates
        let pos_date = [x_eq_date, y_eq_date, z_eq_date];
        let vel_date = [vx_eq_date, vy_eq_date, vz_eq_date];

        let gcrs_pos = mat_mul_pvec(prec_matrix_t, pos_date);
        let gcrs_vel = mat_mul_pvec(prec_matrix_t, vel_date);

        // Store results
        let mut row = out.row_mut(i);
        row[0] = gcrs_pos[0];
        row[1] = gcrs_pos[1];
        row[2] = gcrs_pos[2];
        row[3] = gcrs_vel[0];
        row[4] = gcrs_vel[1];
        row[5] = gcrs_vel[2];
    }

    out
}

/// Calculate high-precision positions for any solar system body using SPICE/ANISE ephemeris
///
/// This function uses JPL's high-precision ephemeris data to calculate positions
/// with sub-arcsecond accuracy. Requires a SPICE kernel file (e.g., de440s.bsp).
///
/// # Arguments
/// * `times` - Vector of timestamps for which to calculate positions
/// * `target_id` - NAIF ID of the target body (e.g., 301 for Moon, 10 for Sun)
/// * `center_id` - NAIF ID of the center body (e.g., 399 for Earth, 0 for Solar System Barycenter)
///
/// Panics if the ephemeris has no GCRS position data
/// // Moon relative to Earth
/// let moon_positions = calculate_body_positions_spice(&times, 301, 399);
/// // Sun relative to Earth
/// let sun_positions = calculate_body_positions_spice(&times, 10, 399);
/// ```
/// Calculate high-precision positions for any solar system body using SPICE/ANISE ephemeris
///
/// This function uses JPL's high-precision ephemeris data to calculate positions
/// with sub-arcsecond accuracy. Requires a SPICE kernel file (e.g., de440s.bsp).
///
/// # Arguments
/// * `times` - Vector of timestamps for which to calculate positions
/// * `target_id` - NAIF ID of the target body (e.g., 301 for Moon, 10 for Sun)
/// * `center_id` - NAIF ID of the center body (e.g., 399 for Earth, 0 for Solar System Barycenter)
///
/// Panics if the ephemeris has no GCRS position data
/// // Moon relative to Earth
/// let moon_positions = calculate_body_positions_spice(&times, 301, 399);
/// // Sun relative to Earth
/// let sun_positions = calculate_body_positions_spice(&times, 10, 399);
/// ```
pub fn calculate_body_positions_spice(
    times: &[DateTime<Utc>],
    target_id: i32,
    center_id: i32,
) -> Array2<f64> {
    // Import ANISE types
    use anise::prelude::*;
    use hifitime::Epoch as HifiEpoch;

    // Prefer a centrally-initialized planetary almanac if available
    use crate::ephemeris::spice_manager;

    let maybe_ephemeris = spice_manager::get_planetary_ephemeris();

    let almanac = if let Some(almanac) = maybe_ephemeris {
        almanac
    } else {
        // Fallback: try to load from the best available default cache path (prefer full DE440)
        let path = if let Some(p) = spice_manager::best_available_planetary_path() {
            p
        } else {
            DEFAULT_DE440S_PATH.as_path().to_path_buf()
        };
        if !path.exists() {
            panic!(
                "SPK file not found at '{}'. Cannot compute high-precision body positions. \
                 To resolve this, initialize the planetary ephemeris using init_planetary_ephemeris() \
                 or ensure the SPK file exists at the specified path.",
                path.display()
            );
        }

        let path_str = path.to_str().expect("Invalid UTF-8 in path");
        let spk = SPK::load(path_str)
            .unwrap_or_else(|e| panic!("Failed to load SPK file '{}': {:?}", path.display(), e));

        Arc::new(Almanac::default().with_spk(spk))
    };

    // Prepare output array
    let n = times.len();
    let mut out = Array2::<f64>::zeros((n, 6));

    // Cache frames outside the loop to avoid repeated lookups
    let target_frame = get_cached_frame(target_id);
    let center_frame = get_cached_frame(center_id);

    // Use rayon for parallelization on large ephemerides (threshold: 50+ timestamps)
    if n > 50 {
        use rayon::prelude::*;

        // Process in parallel chunks
        let chunk_size = (n / rayon::current_num_threads()).max(1);
        let results: Vec<Vec<[f64; 6]>> = times
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(chunk_idx, chunk_times)| {
                chunk_times
                    .iter()
                    .enumerate()
                    .map(|(local_i, dt)| {
                        let _global_i = chunk_idx * chunk_size + local_i;

                        // Convert DateTime<Utc> to hifitime Epoch
                        let epoch = HifiEpoch::from_gregorian_utc(
                            dt.year(),
                            dt.month() as u8,
                            dt.day() as u8,
                            dt.hour() as u8,
                            dt.minute() as u8,
                            dt.second() as u8,
                            dt.timestamp_subsec_nanos(),
                        );

                        // Use epoch nanoseconds as cache key
                        let epoch_ns = dt.timestamp_nanos_opt().unwrap_or(0);

                        // Check position cache first
                        if let Some(cached_pos) = get_cached_position(target_id, center_id, epoch_ns) {
                            return cached_pos;
                        }

                        let state = almanac
                            .translate_geometric(target_frame, center_frame, epoch)
                            .unwrap_or_else(|e| {
                                panic!(
                                    "Failed to get state for body {target_id} relative to {center_id} at {dt}: {e:?}. \
                                     Ensure SPK file contains ephemeris data for both bodies."
                                )
                            });

                        // Extract position and velocity
                        let pos_vel = [
                            state.radius_km.x,
                            state.radius_km.y,
                            state.radius_km.z,
                            state.velocity_km_s.x,
                            state.velocity_km_s.y,
                            state.velocity_km_s.z,
                        ];

                        // Cache for future queries
                        cache_position(target_id, center_id, epoch_ns, pos_vel);

                        pos_vel
                    })
                    .collect()
            })
            .collect();

        // Flatten and store results
        let mut row_idx = 0;
        for chunk_results in results {
            for pos_vel in chunk_results {
                let mut row = out.row_mut(row_idx);
                row[0] = pos_vel[0];
                row[1] = pos_vel[1];
                row[2] = pos_vel[2];
                row[3] = pos_vel[3];
                row[4] = pos_vel[4];
                row[5] = pos_vel[5];
                row_idx += 1;
            }
        }
    } else {
        // Use serial processing for small ephemerides
        for (i, dt) in times.iter().enumerate() {
            // Convert DateTime<Utc> to hifitime Epoch
            let epoch = HifiEpoch::from_gregorian_utc(
                dt.year(),
                dt.month() as u8,
                dt.day() as u8,
                dt.hour() as u8,
                dt.minute() as u8,
                dt.second() as u8,
                dt.timestamp_subsec_nanos(),
            );

            // Use epoch nanoseconds as cache key
            let epoch_ns = dt.timestamp_nanos_opt().unwrap_or(0);

            // Check position cache first
            if let Some(cached_pos) = get_cached_position(target_id, center_id, epoch_ns) {
                let mut row = out.row_mut(i);
                row[0] = cached_pos[0];
                row[1] = cached_pos[1];
                row[2] = cached_pos[2];
                row[3] = cached_pos[3];
                row[4] = cached_pos[4];
                row[5] = cached_pos[5];
                continue;
            }

            let state = almanac
                .translate_geometric(target_frame, center_frame, epoch)
                .unwrap_or_else(|e| {
                    panic!(
                        "Failed to get state for body {target_id} relative to {center_id} at {dt}: {e:?}. \
                         Ensure SPK file contains ephemeris data for both bodies."
                    )
                });

            // Extract position and velocity
            let pos_vel = [
                state.radius_km.x,
                state.radius_km.y,
                state.radius_km.z,
                state.velocity_km_s.x,
                state.velocity_km_s.y,
                state.velocity_km_s.z,
            ];

            // Cache for future queries
            cache_position(target_id, center_id, epoch_ns, pos_vel);

            let mut row = out.row_mut(i);
            row[0] = pos_vel[0];
            row[1] = pos_vel[1];
            row[2] = pos_vel[2];
            row[3] = pos_vel[3];
            row[4] = pos_vel[4];
            row[5] = pos_vel[5];
        }
    }

    out
}

/// Non-panicking variant: calculate body positions using SPICE, returning Result
pub fn calculate_body_positions_spice_result(
    times: &[DateTime<Utc>],
    target_id: i32,
    center_id: i32,
    spice_kernel: Option<&str>,
) -> Result<Array2<f64>, String> {
    use crate::ephemeris::spice_manager;
    use anise::prelude::*;
    use hifitime::Epoch as HifiEpoch;

    // If a kernel is specified, initialize from it (URL or path). Otherwise use cache/defaults.
    if let Some(spec) = spice_kernel {
        spice_manager::ensure_planetary_ephemeris_spec(spec)
            .map_err(|e| format!("Failed to initialize planetary ephemeris from '{spec}': {e}"))?;
    }

    let almanac = if let Some(almanac) = spice_manager::get_planetary_ephemeris() {
        almanac
    } else {
        let path = if let Some(p) = spice_manager::best_available_planetary_path() {
            p
        } else {
            DEFAULT_DE440S_PATH.as_path().to_path_buf()
        };
        if !path.exists() {
            return Err(format!(
                "SPK file not found at '{}'. Initialize planetary ephemeris with ensure_planetary_ephemeris().",
                path.display()
            ));
        }
        let path_str = path
            .to_str()
            .ok_or_else(|| "Invalid UTF-8 in SPK path".to_string())?;
        let spk = SPK::load(path_str)
            .map_err(|e| format!("Failed to load SPK file '{}': {:?}", path.display(), e))?;
        Arc::new(Almanac::default().with_spk(spk))
    };

    let n = times.len();
    let mut out = Array2::<f64>::zeros((n, 6));

    // Cache frames outside the loop to avoid repeated lookups
    let target_frame = get_cached_frame(target_id);
    let center_frame = get_cached_frame(center_id);

    for (i, dt) in times.iter().enumerate() {
        let epoch = HifiEpoch::from_gregorian_utc(
            dt.year(),
            dt.month() as u8,
            dt.day() as u8,
            dt.hour() as u8,
            dt.minute() as u8,
            dt.second() as u8,
            dt.timestamp_subsec_nanos(),
        );

        // Use epoch nanoseconds as cache key
        let epoch_ns = dt.timestamp_nanos_opt().unwrap_or(0);

        // Check position cache first
        if let Some(cached_pos) = get_cached_position(target_id, center_id, epoch_ns) {
            let mut row = out.row_mut(i);
            row[0] = cached_pos[0];
            row[1] = cached_pos[1];
            row[2] = cached_pos[2];
            row[3] = cached_pos[3];
            row[4] = cached_pos[4];
            row[5] = cached_pos[5];
            continue;
        }

        let state = almanac
            .translate_geometric(target_frame, center_frame, epoch)
            .map_err(|e| {
                format!(
                    "SPICE could not provide body {target_id} relative to {center_id} at {dt}: {e:?}. \
                     This often means the current kernel does not include that NAIF ID. \
                     If you're using de440s.bsp, try a planetary barycenter (e.g., 'Jupiter barycenter' or '5') \
                     or install a full kernel like de440.bsp."
                )
            })?;

        // Extract and cache position/velocity
        let pos_vel = [
            state.radius_km.x,
            state.radius_km.y,
            state.radius_km.z,
            state.velocity_km_s.x,
            state.velocity_km_s.y,
            state.velocity_km_s.z,
        ];

        cache_position(target_id, center_id, epoch_ns, pos_vel);

        let mut row = out.row_mut(i);
        row[0] = pos_vel[0];
        row[1] = pos_vel[1];
        row[2] = pos_vel[2];
        row[3] = pos_vel[3];
        row[4] = pos_vel[4];
        row[5] = pos_vel[5];
    }

    Ok(out)
}

pub fn radec_to_altaz(
    ra_deg: f64,
    dec_deg: f64,
    ephemeris: &dyn EphemerisBase,
    time_indices: Option<&[usize]>,
) -> Array2<f64> {
    // Get ephemeris data
    let times = ephemeris.get_times().expect("Ephemeris must have times");
    let gcrs_data = ephemeris
        .data()
        .gcrs
        .as_ref()
        .expect("Ephemeris must have GCRS data");

    // Filter times and GCRS data if indices provided
    let (times_filtered, gcrs_filtered) = if let Some(indices) = time_indices {
        let filtered_times: Vec<DateTime<Utc>> = indices.iter().map(|&i| times[i]).collect();
        let obs_filtered = gcrs_data.select(ndarray::Axis(0), indices);
        (filtered_times, obs_filtered)
    } else {
        (times.to_vec(), gcrs_data.clone())
    };

    let n_times = times_filtered.len();
    let mut result = Array2::<f64>::zeros((n_times, 2));

    // Convert observer positions from GCRS to ITRS (convert_frames expects 6 columns: pos + vel)
    let itrs_data = convert_frames(
        &gcrs_filtered,
        &times_filtered,
        Frame::GCRS,
        Frame::ITRS,
        true,
    );

    // Convert ITRS positions to geodetic coordinates (extract position columns only)
    let positions_slice = itrs_data.slice(s![.., 0..3]);
    let positions_array = positions_slice.to_owned();
    let (lats_deg, lons_deg, heights_km) = ecef_to_geodetic_deg(&positions_array);

    // Precompute target coordinates in radians
    let ra_rad = ra_deg.to_radians();
    let dec_rad = dec_deg.to_radians();

    for i in 0..n_times {
        let lat_rad = lats_deg[i].to_radians();
        let lon_rad = lons_deg[i].to_radians();
        let height_m = heights_km[i] * 1000.0;
        let time = &times_filtered[i];

        let (utc1, utc2) = datetime_to_jd_utc(time);
        let dut1 = ut1_provider::get_ut1_utc_offset(time);
        let (xp, yp) = eop_provider::get_polar_motion_rad(time);

        // Use SOFA apparent-place routine for full topocentric alt/az (pressure=0: no refraction)
        let (aob, zob, _hob, _dob, _rob, _eo) = atco13(
            ra_rad, dec_rad, 0.0, 0.0, 0.0, 0.0, utc1, utc2, dut1, lon_rad, lat_rad, height_m, xp,
            yp, 0.0, 0.0, 0.0, 0.55,
        )
        .expect("SOFA atco13 failed");

        let alt_deg = (std::f64::consts::FRAC_PI_2 - zob).to_degrees();
        let mut az_deg = aob.to_degrees();
        if az_deg < 0.0 {
            az_deg += 360.0;
        }

        result[[i, 0]] = alt_deg;
        result[[i, 1]] = az_deg;
    }

    result
}

/// Calculate airmass using Kasten empirical formula (fast approximation)
///
/// The Kasten formula is a simple empirical fit to airmass vs zenith angle:
/// airmass = 1 / (cos(z) + 0.50572 * (96.07995 - z)^(-1.6364))
///
/// This is ~1000x faster than computing full topocentric coordinates with SOFA.
/// Accuracy: ±0.02 airmass for zenith angles up to ~75°
///
/// # Arguments
/// * `altitude_deg` - Altitude angle in degrees (0° = horizon, 90° = zenith)
///
/// # Returns
/// Airmass value (1.0 at zenith, increases toward horizon)
///
/// # Reference
/// Kasten, F., & Young, A. T. (1989)
pub fn calculate_airmass_kasten(altitude_deg: f64) -> f64 {
    if altitude_deg <= 0.0 {
        return f64::INFINITY; // Target below horizon
    }

    let zenith_deg = 90.0 - altitude_deg;
    let cos_z = zenith_deg.to_radians().cos();

    // Kasten formula: AM = 1 / (cos(z) + 0.50572 * (96.07995 - z)^(-1.6364))
    let denominator = cos_z + 0.50572 * (96.07995 - zenith_deg).powf(-1.6364);
    1.0 / denominator
}

/// Calculate airmass for multiple targets and all times using matrix multiplication
///
/// This fully vectorized function calculates airmass for multiple RA/Dec targets across
/// all ephemeris times using matrix operations for maximum performance on large batches.
///
/// # Arguments
/// * `ras_deg` - Right ascensions in degrees (array of N targets)
/// * `decs_deg` - Declinations in degrees (array of N targets)
/// * `ephemeris` - Ephemeris containing observer positions and times
/// * `time_indices` - Optional indices into ephemeris times (default: all times)
///
/// # Returns
/// Array2<f64> with shape (N_targets, N_times) containing airmass values
///
/// # Performance
/// 50-100x faster than full topocentric SOFA calculation
/// Matrix multiplication enables vectorized calculation for large batches (1000s of targets)
pub fn calculate_airmass_batch_fast(
    ras_deg: &[f64],
    decs_deg: &[f64],
    ephemeris: &dyn EphemerisBase,
    time_indices: Option<&[usize]>,
) -> Array2<f64> {
    assert_eq!(
        ras_deg.len(),
        decs_deg.len(),
        "RA and Dec arrays must have same length"
    );

    let n_targets = ras_deg.len();

    // Get observer positions in GCRS frame
    let obs_positions = ephemeris
        .data()
        .gcrs
        .as_ref()
        .expect("Ephemeris must have GCRS data");

    // Filter data if indices provided
    let obs_filtered = if let Some(indices) = time_indices {
        obs_positions.select(ndarray::Axis(0), indices)
    } else {
        obs_positions.clone()
    };

    let n_times = obs_filtered.nrows();

    // Pre-compute observer latitudes and longitudes for all times
    let obs_x = obs_filtered.column(0);
    let obs_y = obs_filtered.column(1);
    let obs_z = obs_filtered.column(2);

    // Compute rho = sqrt(x² + y²) more efficiently using element-wise operations
    let rho: Array1<f64> = obs_x
        .iter()
        .zip(obs_y.iter())
        .map(|(x, y)| {
            let x2y2 = x * x + y * y;
            x2y2.sqrt()
        })
        .collect();

    // Vectorize lat/lon computation avoiding collect
    let obs_lats: Array1<f64> = Array1::from_shape_fn(n_times, |i| obs_z[i].atan2(rho[i]));
    let obs_lons: Array1<f64> = Array1::from_shape_fn(n_times, |i| obs_y[i].atan2(obs_x[i]));

    // Convert target RA/Dec to radians - avoid intermediate Vec
    let ras_rad: Array1<f64> = Array1::from_shape_fn(n_targets, |i| ras_deg[i].to_radians());
    let decs_rad: Array1<f64> = Array1::from_shape_fn(n_targets, |i| decs_deg[i].to_radians());

    // Pre-compute trig functions
    let sin_decs = decs_rad.mapv(f64::sin);
    let cos_decs = decs_rad.mapv(f64::cos);
    let sin_lats = obs_lats.mapv(f64::sin);
    let cos_lats = obs_lats.mapv(f64::cos);

    // Compute hour angles matrix: HA[j, i] = obs_lon[i] - ra_rad[j]
    // Avoid .to_owned() on broadcast by using from_shape_fn for direct computation
    let ha_matrix: Array2<f64> =
        Array2::from_shape_fn((n_targets, n_times), |(j, i)| obs_lons[i] - ras_rad[j]);

    // Reshape 1D arrays into column/row vectors for broadcasting
    let sin_dec_col = sin_decs.view().into_shape((n_targets, 1)).unwrap();
    let sin_lat_row = sin_lats.view().into_shape((1, n_times)).unwrap();
    let cos_dec_col = cos_decs.view().into_shape((n_targets, 1)).unwrap();
    let cos_lat_row = cos_lats.view().into_shape((1, n_times)).unwrap();

    // First term: sin(dec)[j] * sin(lat)[i]
    let first_term = &sin_dec_col * &sin_lat_row;

    // Second term: cos(dec)[j] * cos(lat)[i] * cos(HA[j, i])
    let cos_ha = ha_matrix.mapv(f64::cos);
    let second_term = &cos_dec_col * &cos_lat_row * &cos_ha;

    // Combine and apply Kasten formula
    (first_term + second_term).mapv(|sin_alt| {
        let alt_rad = sin_alt.clamp(-1.0, 1.0).asin();
        calculate_airmass_kasten(alt_rad.to_degrees())
    })
}

/// Calculate Sun altitudes for all ephemeris times (vectorized for daytime constraints)
///
/// This function calculates the Sun's altitude above the horizon for all times in the ephemeris,
/// which is needed for daytime constraint evaluation.
///
/// # Arguments
/// * `ephemeris` - Ephemeris object containing Sun positions and observer positions
/// * `time_indices` - Optional indices into ephemeris times (default: all times)
///
/// # Returns
/// Array1 containing Sun altitude in degrees for each time
///
/// # Performance
/// Calculate Sun altitude angles - FAST approximation version
/// This is suitable for daytime/twilight constraints where we don't need
/// topocentric accuracy (sub-degree is fine).
///
/// This version avoids expensive frame conversions and SOFA calls by using
/// a simpler geocentric approximation, which is typically within 0.5-1.0 degrees
/// of the true topocentric value for ground-based observers.
pub fn calculate_sun_altitudes_batch_fast(
    ephemeris: &dyn EphemerisBase,
    time_indices: Option<&[usize]>,
) -> Array1<f64> {
    // Get Sun positions in GCRS frame
    let sun_positions = ephemeris
        .data()
        .sun_gcrs
        .as_ref()
        .expect("Ephemeris must have Sun data");

    // Get observer positions in GCRS frame
    let obs_positions = ephemeris
        .data()
        .gcrs
        .as_ref()
        .expect("Ephemeris must have GCRS data");

    // Get times
    let times = ephemeris.get_times().expect("Ephemeris must have times");

    // Filter data if indices provided
    let (sun_filtered, obs_filtered, times_filtered) = if let Some(indices) = time_indices {
        let sun_filtered = sun_positions.select(ndarray::Axis(0), indices);
        let obs_filtered = obs_positions.select(ndarray::Axis(0), indices);
        let times_filtered: Vec<DateTime<Utc>> = indices.iter().map(|&i| times[i]).collect();
        (sun_filtered, obs_filtered, times_filtered)
    } else {
        (sun_positions.clone(), obs_positions.clone(), times.to_vec())
    };

    let n_times = times_filtered.len();
    let mut result = Array1::<f64>::zeros(n_times);

    // Use simple geocentric approximation:
    // 1. Calculate relative Sun position (Sun as seen from observer)
    // 2. Convert to altitude using observer's latitude/longitude from geocentric position
    // 3. This avoids expensive frame conversions and is good enough for twilight

    for i in 0..n_times {
        // Get positions
        let sun_x = sun_filtered[[i, 0]];
        let sun_y = sun_filtered[[i, 1]];
        let sun_z = sun_filtered[[i, 2]];

        let obs_x = obs_filtered[[i, 0]];
        let obs_y = obs_filtered[[i, 1]];
        let obs_z = obs_filtered[[i, 2]];

        // Relative Sun position from observer
        let rel_x = sun_x - obs_x;
        let rel_y = sun_y - obs_y;
        let rel_z = sun_z - obs_z;

        // Observer's geocentric latitude and longitude (from GCRS position)
        let obs_lat = obs_z.atan2((obs_x * obs_x + obs_y * obs_y).sqrt());
        let obs_lon = obs_y.atan2(obs_x);

        // Convert Sun's relative position to RA/Dec
        let sun_dist = (rel_x * rel_x + rel_y * rel_y + rel_z * rel_z).sqrt();
        if sun_dist == 0.0 {
            result[i] = -90.0; // Sun at observer position
            continue;
        }

        let sun_dec = (rel_z / sun_dist).asin();
        let sun_ra = rel_y.atan2(rel_x);

        // Calculate Hour Angle (HA = LST - RA)
        // For simplicity, approximate LST ≈ observer_longitude (geocentric)
        // In GCRS, the x-axis points to vernal equinox, so longitude ≈ atan2(y, x)
        let lst_approx = obs_lon;
        let ha = lst_approx - sun_ra;

        // Simple altitude formula (good enough for daytime constraint):
        // sin(alt) = sin(dec) * sin(lat) + cos(dec) * cos(lat) * cos(HA)
        let sin_alt = sun_dec.sin() * obs_lat.sin() + sun_dec.cos() * obs_lat.cos() * ha.cos();
        let alt_rad = sin_alt.asin();
        result[i] = alt_rad.to_degrees();
    }

    result
}

/// Convert CIRS RA/Dec to horizontal coordinates (altitude/azimuth)
///
/// # Arguments
/// * `lat_deg` - Observer latitude in degrees
/// * `lst_deg` - Local Sidereal Time in degrees
/// * `ra_deg` - Right ascension in CIRS frame (degrees)
/// * `dec_deg` - Declination in CIRS frame (degrees)
///
/// # Returns
/// (altitude_deg, azimuth_deg)
/// Compatibility stub for old calculate_moon_positions_spice function
///
/// This function maintains backward compatibility with code that used the old
/// `calculate_moon_positions_spice(times, spk_path)` signature. It now calls
/// the more general `calculate_body_positions_spice` with Moon-specific parameters.
///
/// # Arguments
/// * `times` - Vector of timestamps for which to calculate Moon positions
///
/// # Returns
/// Array2 with shape (N, 6) containing [x, y, z, vx, vy, vz] in GCRS frame for each timestamp
///
/// # Note
/// The ephemeris is now loaded globally via `spice_manager`. This function calls
/// `calculate_body_positions_spice(times, 301, 399)` where 301 is the NAIF ID for Moon
/// and 399 is the NAIF ID for Earth.
pub fn calculate_moon_positions(times: &[DateTime<Utc>]) -> Array2<f64> {
    // Moon NAIF ID: 301, Earth NAIF ID: 399
    if is_planetary_ephemeris_initialized() {
        calculate_body_positions_spice(times, MOON_NAIF_ID, EARTH_NAIF_ID)
    } else {
        calculate_moon_positions_meeus(times)
    }
}

pub fn calculate_sun_positions(times: &[DateTime<Utc>]) -> Array2<f64> {
    // Sun NAIF ID: 10, Earth NAIF ID: 399
    if is_planetary_ephemeris_initialized() {
        calculate_body_positions_spice(times, SUN_NAIF_ID, EARTH_NAIF_ID)
    } else {
        calculate_sun_positions_erfa(times)
    }
}

/// Calculate positions for any body identified by NAIF ID or name
///
/// This is a convenience wrapper around `calculate_body_positions_spice` that accepts
/// either a NAIF ID (integer) or a body name (string). This is analogous to astropy's
/// `get_body` function.
///
/// # Arguments
/// * `times` - Vector of timestamps for which to calculate positions
/// * `body_identifier` - NAIF ID or body name (e.g., "Jupiter", "mars", "301" for Moon, "Halley" for comet)
/// * `observer_id` - NAIF ID of the observer/center body (default: 399 for Earth)
///
/// # Returns
/// `Ok(Array2<f64>)` with shape (N, 6) containing [x, y, z, vx, vy, vz] in GCRS frame,
/// or `Err(String)` if the body identifier is not recognized
///
/// # Example
/// ```rust,ignore
/// use chrono::{DateTime, Utc};
/// let times = vec![DateTime::parse_from_rfc3339("2025-01-01T00:00:00Z").unwrap().into()];
///
/// // By name
/// let jupiter = calculate_body_by_id_or_name(&times, "Jupiter", 399).unwrap();
///
/// // By NAIF ID
/// let mars = calculate_body_by_id_or_name(&times, "499", 399).unwrap();
///
/// // By name (case insensitive)
/// let moon = calculate_body_by_id_or_name(&times, "moon", 399, None, false).unwrap();
///
/// // Comet by name (requires use_horizons=true)
/// let halley = calculate_body_by_id_or_name(&times, "Halley", 399, None, true).unwrap();
///
/// // Or use JPL Horizons as fallback when SPICE data unavailable
/// let asteroid = calculate_body_by_id_or_name(&times, "433", 399, None, true).unwrap();
/// ```
pub fn calculate_body_by_id_or_name(
    times: &[DateTime<Utc>],
    body_identifier: &str,
    observer_id: i32,
    spice_kernel: Option<&str>,
    use_horizons: bool,
) -> Result<Array2<f64>, String> {
    use crate::naif_ids::parse_body_identifier;
    use crate::utils::horizons::query_horizons_body;

    // First, try to parse as a known NAIF ID/name
    if let Some(target_id) = parse_body_identifier(body_identifier) {
        // Try SPICE first
        let spice_result =
            calculate_body_positions_spice_result(times, target_id, observer_id, spice_kernel);

        // If SPICE fails and use_horizons is enabled, try JPL Horizons
        if spice_result.is_err() && use_horizons {
            return query_and_convert_horizons(times, target_id, observer_id, &query_horizons_body);
        }

        return spice_result;
    }

    // If not a recognized NAIF ID/name and use_horizons is enabled, try as a comet or object name
    if use_horizons {
        // Treat as a comet/object name
        return query_and_convert_horizons_by_name(times, body_identifier, observer_id);
    }

    // Neither NAIF ID/name nor Horizons
    Err(format!(
        "Unknown body identifier: '{}'. Provide a valid NAIF ID, body name (e.g., 'Jupiter', 'Mars', '301' for Moon), or enable use_horizons=True for comet/asteroid names.",
        body_identifier
    ))
}

/// Helper function to query Horizons by ID and convert coordinates
#[allow(clippy::type_complexity)]
fn query_and_convert_horizons(
    times: &[DateTime<Utc>],
    target_id: i32,
    observer_id: i32,
    query_fn: &dyn Fn(&[DateTime<Utc>], i32) -> Result<Array2<f64>, String>,
) -> Result<Array2<f64>, String> {
    // Horizons fallback only works for Earth-centered observers (NAIF ID 399)
    // For satellite observers, use SPICE kernels instead
    if observer_id != EARTH_NAIF_ID {
        return Err(format!(
            "Horizons fallback only supports Earth-centered observer (NAIF ID {}). \
            For satellite/spacecraft observers, ensure SPICE kernels are available. Got observer_id={}",
            EARTH_NAIF_ID, observer_id
        ));
    }

    // Query Horizons for geocentric position (CENTER='@399' in the API)
    // The Horizons API already returns Earth-centered ICRF coordinates
    query_fn(times, target_id)
}

/// Helper function to query Horizons by name and convert coordinates
fn query_and_convert_horizons_by_name(
    times: &[DateTime<Utc>],
    body_name: &str,
    observer_id: i32,
) -> Result<Array2<f64>, String> {
    use crate::utils::horizons::query_horizons_body_by_name;

    // Horizons fallback only works for Earth-centered observers (NAIF ID 399)
    // For satellite observers, use SPICE kernels instead
    if observer_id != EARTH_NAIF_ID {
        return Err(format!(
            "Horizons fallback only supports Earth-centered observer (NAIF ID {}). \
            For satellite/spacecraft observers, ensure SPICE kernels are available. Got observer_id={}",
            EARTH_NAIF_ID, observer_id
        ));
    }

    // Query Horizons for geocentric position by name (CENTER='@399' in the API)
    // The Horizons API already returns Earth-centered ICRF coordinates
    query_horizons_body_by_name(times, body_name)
}

/// Calculate positions for multiple bodies in a single efficient pass
///
/// This function is optimized for querying multiple bodies at once, avoiding
/// redundant time conversions and using shared frame cache and almanac.
///
/// # Arguments
/// * `times` - Vector of timestamps for which to calculate positions
/// * `body_ids` - Array of NAIF IDs for bodies to query
/// * `observer_id` - NAIF ID of the observer/center body
/// * `spice_kernel` - Optional path/URL to SPICE kernel
///
/// # Returns
/// HashMap mapping NAIF ID to Array2<f64> with positions/velocities
#[allow(dead_code)]
pub fn calculate_multiple_body_positions_spice(
    times: &[DateTime<Utc>],
    body_ids: &[i32],
    observer_id: i32,
    spice_kernel: Option<&str>,
) -> Result<HashMap<i32, Array2<f64>>, String> {
    use crate::ephemeris::spice_manager;
    use anise::prelude::*;
    use hifitime::Epoch as HifiEpoch;

    // If a kernel is specified, initialize from it
    if let Some(spec) = spice_kernel {
        spice_manager::ensure_planetary_ephemeris_spec(spec)
            .map_err(|e| format!("Failed to initialize planetary ephemeris from '{spec}': {e}"))?;
    }

    let almanac = if let Some(almanac) = spice_manager::get_planetary_ephemeris() {
        almanac
    } else {
        let path = if let Some(p) = spice_manager::best_available_planetary_path() {
            p
        } else {
            DEFAULT_DE440S_PATH.as_path().to_path_buf()
        };
        if !path.exists() {
            return Err(format!(
                "SPK file not found at '{}'. Initialize planetary ephemeris with ensure_planetary_ephemeris().",
                path.display()
            ));
        }
        let path_str = path
            .to_str()
            .ok_or_else(|| "Invalid UTF-8 in SPK path".to_string())?;
        let spk = SPK::load(path_str)
            .map_err(|e| format!("Failed to load SPK file '{}': {:?}", path.display(), e))?;
        Arc::new(Almanac::default().with_spk(spk))
    };

    // Pre-compute center frame (used for all bodies)
    let center_frame = get_cached_frame(observer_id);

    // Process all timestamps once, computing positions for all bodies
    let n_times = times.len();
    let mut results: HashMap<i32, Array2<f64>> = HashMap::new();

    // Initialize output arrays for each body
    for &body_id in body_ids {
        results.insert(body_id, Array2::<f64>::zeros((n_times, 6)));
    }

    // Single pass through all times, computing all bodies at each time
    for (i, dt) in times.iter().enumerate() {
        let epoch = HifiEpoch::from_gregorian_utc(
            dt.year(),
            dt.month() as u8,
            dt.day() as u8,
            dt.hour() as u8,
            dt.minute() as u8,
            dt.second() as u8,
            dt.timestamp_subsec_nanos(),
        );

        for &body_id in body_ids {
            let target_frame = get_cached_frame(body_id);

            let state = almanac
                .translate_geometric(target_frame, center_frame, epoch)
                .map_err(|e| {
                    format!(
                        "SPICE could not provide body {body_id} relative to {observer_id} at {dt}: {e:?}"
                    )
                })?;

            let mut row = results.get_mut(&body_id).unwrap().row_mut(i);
            row[0] = state.radius_km.x;
            row[1] = state.radius_km.y;
            row[2] = state.radius_km.z;
            row[3] = state.velocity_km_s.x;
            row[4] = state.velocity_km_s.y;
            row[5] = state.velocity_km_s.z;
        }
    }

    Ok(results)
}
