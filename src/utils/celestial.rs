use chrono::{DateTime, Datelike, Timelike, Utc};
use erfa::earth::position_velocity_00;
use erfa::prenut::precession_matrix_06;
use erfa::vectors_and_matrices::mat_mul_pvec;
use ndarray::{Array2, s};
use std::sync::Arc;

use crate::ephemeris::ephemeris_common::EphemerisBase;
use crate::utils::conversions::{convert_frames, Frame};
use crate::utils::geo::ecef_to_geodetic_deg;
use crate::utils::math_utils::transpose_matrix;
use crate::utils::time_utils::{datetime_to_jd_tt, datetime_to_jd_utc};
use crate::utils::{eop_provider, ut1_provider};
use crate::{is_planetary_ephemeris_initialized, utils::config::*};

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

        // Create frames for target and center bodies using J2000 orientation (GCRS)
        let target_frame = Frame::from_ephem_j2000(target_id);
        let center_frame = Frame::from_ephem_j2000(center_id);

        let state = almanac
            .translate_geometric(target_frame, center_frame, epoch)
            .unwrap_or_else(|e| {
                panic!(
                    "Failed to get state for body {target_id} relative to {center_id} at {dt}: {e:?}. \
                     Ensure SPK file contains ephemeris data for both bodies."
                )
            });

        // Extract position and velocity
        // ANISE returns position in km and velocity in km/s
        let mut row = out.row_mut(i);
        row[0] = state.radius_km.x;
        row[1] = state.radius_km.y;
        row[2] = state.radius_km.z;
        row[3] = state.velocity_km_s.x;
        row[4] = state.velocity_km_s.y;
        row[5] = state.velocity_km_s.z;
    }

    out
}

/// Non-panicking variant: calculate body positions using SPICE, returning Result
pub fn calculate_body_positions_spice_result(
    times: &[DateTime<Utc>],
    target_id: i32,
    center_id: i32,
) -> Result<Array2<f64>, String> {
    use crate::ephemeris::spice_manager;
    use anise::prelude::*;
    use hifitime::Epoch as HifiEpoch;

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

        let target_frame = Frame::from_ephem_j2000(target_id);
        let center_frame = Frame::from_ephem_j2000(center_id);

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

        let mut row = out.row_mut(i);
        row[0] = state.radius_km.x;
        row[1] = state.radius_km.y;
        row[2] = state.radius_km.z;
        row[3] = state.velocity_km_s.x;
        row[4] = state.velocity_km_s.y;
        row[5] = state.velocity_km_s.z;
    }

    Ok(out)
}

/// Convert RA/Dec to Altitude/Azimuth for given ephemeris times
///
/// This function calculates the topocentric altitude and azimuth of a celestial target
/// as seen from the observer locations defined in the ephemeris.
///
/// # Arguments
/// * `ra_deg` - Right ascension in degrees
/// * `dec_deg` - Declination in degrees  
/// * `ephemeris` - Ephemeris containing observer positions and times
/// * `time_indices` - Optional indices into ephemeris times to evaluate (default: all times)
///
/// # Returns
/// Array2 with shape (N, 2) containing [altitude_deg, azimuth_deg] for each time
/// where N is the number of selected times
///
/// # Panics
/// Panics if the ephemeris has no GCRS position data

pub fn radec_to_altaz(
    ra_deg: f64,
    dec_deg: f64,
    ephemeris: &dyn EphemerisBase,
    time_indices: Option<&[usize]>,
) -> Array2<f64> {
    // Get ephemeris data
    let times = ephemeris.get_times().expect("Ephemeris must have times");
    let gcrs_data = ephemeris.data().gcrs.as_ref().expect("Ephemeris must have GCRS data");
    
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
    let itrs_data = convert_frames(&gcrs_filtered, &times_filtered, Frame::GCRS, Frame::ITRS, true);

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

        // Use ERFA apparent-place routine for full topocentric alt/az (pressure=0: no refraction)
        let mut aob = 0.0;
        let mut zob = 0.0;
        let mut hob = 0.0;
        let mut dob = 0.0;
        let mut rob = 0.0;
        let mut eo = 0.0;

        let status = unsafe {
            erfa_sys::eraAtco13(
                ra_rad,
                dec_rad,
                0.0,
                0.0,
                0.0,
                0.0,
                utc1,
                utc2,
                dut1,
                lon_rad,
                lat_rad,
                height_m,
                xp,
                yp,
                0.0,
                0.0,
                0.0,
                0.55,
                &mut aob,
                &mut zob,
                &mut hob,
                &mut dob,
                &mut rob,
                &mut eo,
            )
        };

        if status < 0 {
            panic!("ERFA eraAtco13 failed with status {status}");
        }

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
/// * `body_identifier` - NAIF ID or body name (e.g., "Jupiter", "mars", "301" for Moon)
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
/// let moon = calculate_body_by_id_or_name(&times, "moon", 399).unwrap();
/// ```
pub fn calculate_body_by_id_or_name(
    times: &[DateTime<Utc>],
    body_identifier: &str,
    observer_id: i32,
) -> Result<Array2<f64>, String> {
    use crate::naif_ids::parse_body_identifier;

    let target_id = parse_body_identifier(body_identifier)
        .ok_or_else(|| format!("Unknown body identifier: '{body_identifier}'. Provide a valid NAIF ID or body name (e.g., 'Jupiter', 'Mars', '301' for Moon)."))?;

    calculate_body_positions_spice_result(times, target_id, observer_id)
}
