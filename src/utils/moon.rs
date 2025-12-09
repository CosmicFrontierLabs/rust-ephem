//! Moon-related calculations and utilities

use chrono::{DateTime, Utc};

/// Calculate Moon illumination fraction as seen from the spacecraft
/// Uses the angular separation between Sun and Moon from the observer's perspective
///
/// # Arguments
/// * `time` - The time at which to calculate illumination
///
/// # Returns
/// Moon illumination fraction (0.0 = new moon, 1.0 = full moon)
pub fn calculate_moon_illumination(time: &DateTime<Utc>) -> f64 {
    // Get 3D positions of Sun and Moon relative to Earth (observer_id = 399)
    // For distant observers, this gives accurate phase angle
    let times = vec![*time];
    let sun_pos_result = crate::utils::celestial::calculate_body_by_id_or_name(&times, "Sun", 399);
    let moon_pos_result =
        crate::utils::celestial::calculate_body_by_id_or_name(&times, "Moon", 399);

    if sun_pos_result.is_err() || moon_pos_result.is_err() {
        return 0.5; // Default to half illumination on error
    }

    let sun_pos = sun_pos_result.unwrap();
    let moon_pos = moon_pos_result.unwrap();

    // Extract positions (first row, x,y,z)
    let sun_x = sun_pos[[0, 0]];
    let sun_y = sun_pos[[0, 1]];
    let sun_z = sun_pos[[0, 2]];

    let moon_x = moon_pos[[0, 0]];
    let moon_y = moon_pos[[0, 1]];
    let moon_z = moon_pos[[0, 2]];

    // Convert to RA/Dec
    let sun_r = (sun_x * sun_x + sun_y * sun_y + sun_z * sun_z).sqrt();
    let sun_ra = sun_y.atan2(sun_x).to_degrees();
    let sun_dec = (sun_z / sun_r).asin().to_degrees();

    let moon_r = (moon_x * moon_x + moon_y * moon_y + moon_z * moon_z).sqrt();
    let moon_ra = moon_y.atan2(moon_x).to_degrees();
    let moon_dec = (moon_z / moon_r).asin().to_degrees();

    // Calculate angular separation between Sun and Moon (phase angle)
    let ra_diff = (sun_ra - moon_ra).to_radians();
    let dec1 = sun_dec.to_radians();
    let dec2 = moon_dec.to_radians();

    let cos_d = dec1.sin() * dec2.sin() + dec1.cos() * dec2.cos() * ra_diff.cos();
    let angular_separation = cos_d.acos(); // in radians

    // Illumination fraction: (1 - cos(phase_angle)) / 2
    // where phase_angle is the angular separation between Sun and Moon
    (1.0 - angular_separation.cos()) / 2.0
}
