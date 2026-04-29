/// Shared tangent-plane polygon geometry for instrument FoV checks.
///
/// Used by both the bright star avoidance and body proximity polygon modes.
use std::f64::consts::PI;

/// Number of roll samples when sweeping all roll angles (5° resolution).
pub const N_ROLL_SAMPLES: usize = 72;

/// Gnomonic (tangent-plane) projection of a sky point relative to a boresight.
///
/// Returns `(east_rad, north_rad)` in tangent-plane coordinates, or `None` if
/// the point is on or behind the tangent plane (cos_c ≤ 0).
pub fn gnomonic_project(
    target_ra_rad: f64,
    target_dec_rad: f64,
    point_ra_rad: f64,
    point_dec_rad: f64,
) -> Option<(f64, f64)> {
    let delta_ra = point_ra_rad - target_ra_rad;
    let (sin_tdec, cos_tdec) = target_dec_rad.sin_cos();
    let (sin_sdec, cos_sdec) = point_dec_rad.sin_cos();
    let (sin_dra, cos_dra) = delta_ra.sin_cos();

    let cos_c = sin_tdec * sin_sdec + cos_tdec * cos_sdec * cos_dra;
    if cos_c <= 0.0 {
        return None;
    }

    let east = cos_sdec * sin_dra / cos_c;
    let north = (cos_tdec * sin_sdec - sin_tdec * cos_sdec * cos_dra) / cos_c;
    Some((east, north))
}

/// Transform tangent-plane (east, north) offsets to instrument (u, v) at a given roll.
///
/// Convention: at roll = 0°, u = east and v = north.  Roll is the position angle
/// of instrument +v from north, measured east of north.
#[inline]
pub fn to_instrument(east: f64, north: f64, sin_roll: f64, cos_roll: f64) -> (f64, f64) {
    (
        east * cos_roll - north * sin_roll,
        east * sin_roll + north * cos_roll,
    )
}

/// Ray-casting point-in-polygon test for `(u_deg, v_deg)` against `vertices` (degrees).
pub fn point_in_polygon(u: f64, v: f64, vertices: &[[f64; 2]]) -> bool {
    let n = vertices.len();
    if n < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let ui = vertices[i][0];
        let vi = vertices[i][1];
        let uj = vertices[j][0];
        let vj = vertices[j][1];
        if ((vi > v) != (vj > v)) && (u < (uj - ui) * (v - vi) / (vj - vi) + ui) {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// True if the sky point `(point_ra_rad, point_dec_rad)` falls inside `vertices`
/// when the instrument is at the given roll.
pub fn point_in_polygon_at_roll(
    target_ra_rad: f64,
    target_dec_rad: f64,
    point_ra_rad: f64,
    point_dec_rad: f64,
    vertices: &[[f64; 2]],
    sin_roll: f64,
    cos_roll: f64,
) -> bool {
    if let Some((east, north)) =
        gnomonic_project(target_ra_rad, target_dec_rad, point_ra_rad, point_dec_rad)
    {
        let (u_rad, v_rad) = to_instrument(east, north, sin_roll, cos_roll);
        point_in_polygon(u_rad.to_degrees(), v_rad.to_degrees(), vertices)
    } else {
        false
    }
}

/// True if the sky point falls inside the polygon for the given roll configuration.
///
/// When `roll_rad` is `None` the function sweeps `N_ROLL_SAMPLES` evenly-spaced
/// roll angles and returns `true` only when **every** roll has the point inside the
/// polygon (i.e., no clear roll exists).
pub fn point_violates_polygon(
    target_ra_rad: f64,
    target_dec_rad: f64,
    point_ra_rad: f64,
    point_dec_rad: f64,
    vertices: &[[f64; 2]],
    roll_rad: Option<f64>,
) -> bool {
    if let Some(roll) = roll_rad {
        let (sin_roll, cos_roll) = roll.sin_cos();
        point_in_polygon_at_roll(
            target_ra_rad,
            target_dec_rad,
            point_ra_rad,
            point_dec_rad,
            vertices,
            sin_roll,
            cos_roll,
        )
    } else {
        let roll_step = 2.0 * PI / N_ROLL_SAMPLES as f64;
        for step in 0..N_ROLL_SAMPLES {
            let roll = step as f64 * roll_step;
            let (sin_roll, cos_roll) = roll.sin_cos();
            if !point_in_polygon_at_roll(
                target_ra_rad,
                target_dec_rad,
                point_ra_rad,
                point_dec_rad,
                vertices,
                sin_roll,
                cos_roll,
            ) {
                return false; // found a clear roll
            }
        }
        true // all rolls blocked
    }
}

/// Convert a unit vector `[x, y, z]` to `(ra_rad, dec_rad)`.
///
/// The vector is assumed to be in the same equatorial frame as the target RA/Dec
/// (e.g., GCRS / ICRS for observer-relative directions).
#[inline]
pub fn unit_to_radec(unit: &[f64; 3]) -> (f64, f64) {
    let dec_rad = unit[2].clamp(-1.0, 1.0).asin();
    let ra_rad = unit[1].atan2(unit[0]);
    (ra_rad, dec_rad)
}
