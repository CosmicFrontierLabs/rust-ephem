//! Moon-related calculations and utilities

/// Calculate Moon illumination fraction from Sun/Moon vectors relative to observer.
///
/// # Arguments
/// * `sun_rel` - Sun position relative to observer (km)
/// * `moon_rel` - Moon position relative to observer (km)
///
/// # Returns
/// Moon illumination fraction (0.0 = new moon, 1.0 = full moon)
pub(crate) fn calculate_moon_illumination_from_vectors(
    sun_rel: [f64; 3],
    moon_rel: [f64; 3],
) -> f64 {
    let sun_r =
        (sun_rel[0] * sun_rel[0] + sun_rel[1] * sun_rel[1] + sun_rel[2] * sun_rel[2]).sqrt();
    let sun_ra = sun_rel[1].atan2(sun_rel[0]).to_degrees();
    let sun_dec = (sun_rel[2] / sun_r).asin().to_degrees();

    let moon_r =
        (moon_rel[0] * moon_rel[0] + moon_rel[1] * moon_rel[1] + moon_rel[2] * moon_rel[2]).sqrt();
    let moon_ra = moon_rel[1].atan2(moon_rel[0]).to_degrees();
    let moon_dec = (moon_rel[2] / moon_r).asin().to_degrees();

    let ra_diff = (sun_ra - moon_ra).to_radians();
    let dec1 = sun_dec.to_radians();
    let dec2 = moon_dec.to_radians();

    let cos_d = dec1.sin() * dec2.sin() + dec1.cos() * dec2.cos() * ra_diff.cos();
    let angular_separation = cos_d.acos();

    (1.0 - angular_separation.cos()) / 2.0
}
