use ndarray::{Array1, Array2};
use std::f64::consts::PI;

/// Convert ECEF (ITRS) positions (Nx3 array) in kilometers to geodetic latitude (degrees), longitude (degrees), and height (km)
/// using WGS84 ellipsoid parameters (a, f).
pub fn ecef_to_geodetic_deg(positions: &Array2<f64>) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    let n = positions.nrows();
    let mut lats = Array1::<f64>::zeros(n);
    let mut lons = Array1::<f64>::zeros(n);
    let mut hs = Array1::<f64>::zeros(n);

    // WGS84 parameters in km
    let a = 6378.137_f64; // semi-major axis km
    let f = 1.0 / 298.257223563_f64;
    let b = a * (1.0 - f);
    let e_sq = f * (2.0 - f);
    let e2p = (a * a - b * b) / (b * b); // second eccentricity squared

    for (i, row) in positions.rows().into_iter().enumerate() {
        let x = row[0];
        let y = row[1];
        let z = row[2];

        let p = (x * x + y * y).sqrt();
        let lon = y.atan2(x);

        // Bowring's formula
        let theta = (z * a).atan2(p * b);
        let sin_theta = theta.sin();
        let cos_theta = theta.cos();
        let lat = (z + e2p * b * sin_theta.powi(3)).atan2(p - e_sq * a * cos_theta.powi(3));

        // Radius of curvature in prime vertical
        let sin_lat = lat.sin();
        let n_phi = a / (1.0 - e_sq * sin_lat * sin_lat).sqrt();
        let h = if p.abs() < 1e-6 {
            z.abs() - b
        } else {
            p / lat.cos() - n_phi
        };

        lats[i] = lat.to_degrees();
        lons[i] = lon.to_degrees();
        hs[i] = h; // km
    }

    (lats, lons, hs)
}

/// Convert degrees array to radians (Array1)
pub fn deg_to_rad_array(deg: &Array1<f64>) -> Array1<f64> {
    deg * (PI / 180.0)
}
