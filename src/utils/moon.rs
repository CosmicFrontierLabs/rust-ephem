//! Moon-related calculations and utilities

/// Calculate Moon illumination fraction as seen from the observer (spacecraft/satellite)
/// Uses the positions of Sun and Moon relative to the observer from the ephemeris
///
/// # Arguments
/// * `ephemeris` - The ephemeris object containing observer and celestial body positions
/// * `time_index` - Index into the ephemeris times array
///
/// # Returns
/// Moon illumination fraction (0.0 = new moon, 1.0 = full moon)
pub fn calculate_moon_illumination<
    E: crate::ephemeris::ephemeris_common::EphemerisBase + ?Sized,
>(
    ephemeris: &E,
    time_index: usize,
) -> f64 {
    // Get Sun and Moon positions relative to the observer
    let sun_positions = match ephemeris.get_sun_positions() {
        Ok(pos) => pos,
        Err(_) => return 0.5, // Default to half illumination on error
    };
    let moon_positions = match ephemeris.get_moon_positions() {
        Ok(pos) => pos,
        Err(_) => return 0.5, // Default to half illumination on error
    };

    // Extract positions for the specific time index
    let sun_x = sun_positions[[time_index, 0]];
    let sun_y = sun_positions[[time_index, 1]];
    let sun_z = sun_positions[[time_index, 2]];

    let moon_x = moon_positions[[time_index, 0]];
    let moon_y = moon_positions[[time_index, 1]];
    let moon_z = moon_positions[[time_index, 2]];

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
