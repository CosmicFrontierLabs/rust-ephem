/// Unit tests for celestial calculation functions
/// Tests the core airmass calculation functions that were changed from altitude_to_airmass to Kasten formula
#[cfg(test)]
mod celestial_tests {
    use chrono::{DateTime, Utc};
    use ndarray::Array2;
    use pyo3::{Py, PyAny};
    use rust_ephem::ephemeris::ephemeris_common::{EphemerisBase, EphemerisData};
    use rust_ephem::utils::celestial::{calculate_airmass_batch_fast, calculate_airmass_kasten};

    // Mock ephemeris for testing - simplified version without PyO3 dependencies
    struct MockEphemeris {
        data: EphemerisData,
    }

    impl MockEphemeris {
        fn new(lat_deg: f64, lon_deg: f64, height_km: f64, times: Vec<DateTime<Utc>>) -> Self {
            let n_times = times.len();

            // Create GCRS positions for observer at given lat/lon/height
            // For simplicity, we'll use a basic approximation
            let mut gcrs_positions = Array2::<f64>::zeros((n_times, 3));

            // Convert lat/lon/height to approximate GCRS coordinates
            // This is a simplified conversion for testing purposes
            let lat_rad = lat_deg.to_radians();
            let lon_rad = lon_deg.to_radians();
            let earth_radius_km = 6371.0; // Approximate Earth radius
            let r = earth_radius_km + height_km;

            for i in 0..n_times {
                gcrs_positions[[i, 0]] = r * lat_rad.cos() * lon_rad.cos(); // X
                gcrs_positions[[i, 1]] = r * lat_rad.cos() * lon_rad.sin(); // Y
                gcrs_positions[[i, 2]] = r * lat_rad.sin(); // Z
            }

            let mut data = EphemerisData::new();
            data.gcrs = Some(gcrs_positions);
            data.times = Some(times);

            MockEphemeris { data }
        }
    }

    impl EphemerisBase for MockEphemeris {
        fn data(&self) -> &EphemerisData {
            &self.data
        }

        fn data_mut(&mut self) -> &mut EphemerisData {
            &mut self.data
        }

        fn get_itrs_data(&self) -> Option<&Array2<f64>> {
            None
        }

        fn get_itrs_skycoord_ref(&self) -> Option<&Py<PyAny>> {
            None
        }

        fn set_itrs_skycoord_cache(&self, _skycoord: Py<PyAny>) -> Result<(), Py<PyAny>> {
            Ok(())
        }

        fn radec_to_altaz(
            &self,
            _ra_deg: f64,
            _dec_deg: f64,
            _time_indices: Option<&[usize]>,
        ) -> Array2<f64> {
            // Not needed for airmass tests
            Array2::zeros((1, 2))
        }
    }

    mod test_calculate_airmass_kasten {
        use super::*;

        #[test]
        fn test_zenith_airmass() {
            // At zenith (90° altitude), airmass should be 1.0
            let airmass = calculate_airmass_kasten(90.0);
            assert!((airmass - 1.0).abs() < 1e-10);
        }

        #[test]
        fn test_horizon_airmass() {
            // At horizon (0° altitude), airmass should be finite but large
            let airmass = calculate_airmass_kasten(0.0);
            assert!(airmass.is_finite());
            assert!(airmass > 30.0); // Should be quite large near horizon
        }

        #[test]
        fn test_below_horizon_airmass() {
            // Below horizon, airmass should be infinite
            let airmass = calculate_airmass_kasten(-5.0);
            assert!(airmass.is_infinite());
            assert!(airmass > 0.0); // Positive infinity
        }

        #[test]
        fn test_various_altitudes() {
            // Test several altitudes and verify Kasten formula behavior
            let test_cases = vec![
                (90.0, 1.0),    // Zenith
                (60.0, 1.1547), // 30° zenith angle
                (45.0, 1.4139), // 45° zenith angle
                (30.0, 2.0000), // 60° zenith angle
                (15.0, 3.8637), // 75° zenith angle
                (5.0, 10.4009), // 85° zenith angle
            ];

            for (alt_deg, expected_approx) in test_cases {
                let airmass = calculate_airmass_kasten(alt_deg);
                assert!(airmass.is_finite());
                assert!(airmass > 1.0);

                // Allow some tolerance for the approximation
                let tolerance = 0.01;
                assert!(
                    (airmass - expected_approx).abs() < tolerance,
                    "Altitude {}°: expected ~{}, got {}",
                    alt_deg,
                    expected_approx,
                    airmass
                );
            }
        }

        #[test]
        fn test_monotonic_increase() {
            // Airmass should increase monotonically as altitude decreases
            let altitudes = vec![80.0, 60.0, 40.0, 20.0, 10.0, 5.0];

            let mut prev_airmass = 0.0;
            for alt in altitudes {
                let airmass = calculate_airmass_kasten(alt);
                assert!(airmass > prev_airmass);
                prev_airmass = airmass;
            }
        }
    }

    mod test_calculate_airmass_batch_fast {
        use super::*;

        #[test]
        fn test_single_target_single_time() {
            // Create mock ephemeris for a single time at equator
            let times = vec![DateTime::parse_from_rfc3339("2024-01-01T12:00:00Z")
                .unwrap()
                .with_timezone(&Utc)];
            let ephem = MockEphemeris::new(0.0, 0.0, 0.0, times);

            // Target at zenith (RA=0, Dec=0 for equatorial observer)
            let ras = vec![0.0];
            let decs = vec![0.0];

            let result = calculate_airmass_batch_fast(&ras, &decs, &ephem, None);

            // Should return 1x1 array
            assert_eq!(result.nrows(), 1);
            assert_eq!(result.ncols(), 1);

            // Zenith target should have airmass ≈ 1.0
            let airmass = result[[0, 0]];
            assert!(airmass > 0.99 && airmass < 1.01);
        }

        #[test]
        fn test_multiple_targets_single_time() {
            let times = vec![DateTime::parse_from_rfc3339("2024-01-01T12:00:00Z")
                .unwrap()
                .with_timezone(&Utc)];
            let ephem = MockEphemeris::new(0.0, 0.0, 0.0, times);

            // Multiple targets
            let ras = vec![0.0, 90.0, 180.0];
            let decs = vec![0.0, 0.0, 0.0];

            let result = calculate_airmass_batch_fast(&ras, &decs, &ephem, None);

            // Should return 3x1 array (3 targets, 1 time)
            assert_eq!(result.nrows(), 3);
            assert_eq!(result.ncols(), 1);

            // First target at zenith should have airmass ≈ 1.0
            assert!(result[[0, 0]] > 0.99 && result[[0, 0]] < 1.01);

            // Other targets should have higher airmass
            assert!(result[[1, 0]] > 1.0);
            assert!(result[[2, 0]] > 1.0);
        }

        #[test]
        fn test_single_target_multiple_times() {
            // Create multiple times
            let base_time = DateTime::parse_from_rfc3339("2024-01-01T12:00:00Z")
                .unwrap()
                .with_timezone(&Utc);
            let times = vec![
                base_time,
                base_time + chrono::Duration::hours(1),
                base_time + chrono::Duration::hours(2),
            ];
            let ephem = MockEphemeris::new(0.0, 0.0, 0.0, times);

            // Single target
            let ras = vec![0.0];
            let decs = vec![0.0];

            let result = calculate_airmass_batch_fast(&ras, &decs, &ephem, None);

            // Should return 1x3 array (1 target, 3 times)
            assert_eq!(result.nrows(), 1);
            assert_eq!(result.ncols(), 3);

            // All should be close to 1.0 (zenith target)
            for i in 0..3 {
                assert!(result[[0, i]] > 0.99 && result[[0, i]] < 1.01);
            }
        }

        #[test]
        fn test_multiple_targets_multiple_times() {
            let base_time = DateTime::parse_from_rfc3339("2024-01-01T12:00:00Z")
                .unwrap()
                .with_timezone(&Utc);
            let times = vec![base_time, base_time + chrono::Duration::hours(1)];
            let ephem = MockEphemeris::new(0.0, 0.0, 0.0, times);

            let ras = vec![0.0, 90.0];
            let decs = vec![0.0, 0.0];

            let result = calculate_airmass_batch_fast(&ras, &decs, &ephem, None);

            // Should return 2x2 array (2 targets, 2 times)
            assert_eq!(result.nrows(), 2);
            assert_eq!(result.ncols(), 2);

            // Check all values are reasonable
            for i in 0..2 {
                for j in 0..2 {
                    let airmass = result[[i, j]];
                    assert!(airmass > 0.0);
                    // Allow infinite for targets that might be below horizon
                    assert!(airmass.is_finite() || airmass.is_infinite());
                }
            }
        }

        #[test]
        fn test_time_indices_filtering() {
            let base_time = DateTime::parse_from_rfc3339("2024-01-01T12:00:00Z")
                .unwrap()
                .with_timezone(&Utc);
            let times = vec![
                base_time,
                base_time + chrono::Duration::hours(1),
                base_time + chrono::Duration::hours(2),
                base_time + chrono::Duration::hours(3),
            ];
            let ephem = MockEphemeris::new(0.0, 0.0, 0.0, times);

            let ras = vec![0.0];
            let decs = vec![0.0];

            // Filter to only times 1 and 3 (0-indexed)
            let time_indices = vec![1, 3];
            let result = calculate_airmass_batch_fast(&ras, &decs, &ephem, Some(&time_indices));

            // Should return 1x2 array (1 target, 2 filtered times)
            assert_eq!(result.nrows(), 1);
            assert_eq!(result.ncols(), 2);
        }

        #[test]
        #[should_panic(expected = "RA and Dec arrays must have same length")]
        fn test_ra_dec_array_length_mismatch() {
            let times = vec![DateTime::parse_from_rfc3339("2024-01-01T12:00:00Z")
                .unwrap()
                .with_timezone(&Utc)];
            let ephem = MockEphemeris::new(0.0, 0.0, 0.0, times);

            let ras = vec![0.0, 90.0];
            let decs = vec![0.0]; // Different length

            // Should panic with assertion error
            calculate_airmass_batch_fast(&ras, &decs, &ephem, None);
        }
    }
}
