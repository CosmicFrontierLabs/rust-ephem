//! A private, requested-time view for constraint evaluation. Source ephemerides
//! and their caches remain immutable. Interpolation never extrapolates.

use super::ephemeris_common::{EphemerisBase, EphemerisData};
use crate::utils::interpolation::hermite_interpolate;
use chrono::{DateTime, Utc};
use ndarray::{s, Array2};
use pyo3::{exceptions::PyValueError, prelude::*};
use std::sync::OnceLock;

pub(crate) struct SampledEphemeris {
    data: EphemerisData,
    itrs: Option<Array2<f64>>,
    itrs_skycoord: OnceLock<Py<PyAny>>,
}

/// Exact rows or insertion points identifying the two bracketing samples.
/// Ephemeris constructors supply chronologically ordered timestamps.
pub(crate) fn locate_times(
    source: &[DateTime<Utc>],
    requested: &[DateTime<Utc>],
) -> PyResult<Vec<Result<usize, usize>>> {
    requested
        .iter()
        .map(|time| match source.binary_search(time) {
            Ok(index) => Ok(Ok(index)),
            Err(right) if right > 0 && right < source.len() => Ok(Err(right)),
            _ => Err(PyValueError::new_err(format!(
                "Time {} outside ephemeris range ({} to {})",
                time.to_rfc3339(),
                source.first().map(DateTime::to_rfc3339).unwrap_or_default(),
                source.last().map(DateTime::to_rfc3339).unwrap_or_default(),
            ))),
        })
        .collect()
}

/// Sample a (N, 6) state table at the requested times, given `locate_times`
/// output. Exact grid rows are copied unchanged; off-grid rows are cubic
/// Hermite interpolated over the bracketing pair only. This is the single
/// interpolation path shared by constraint evaluation (`SampledEphemeris`) and
/// the `*_pv_at` ephemeris methods.
pub(crate) fn sample_states(
    states: &Array2<f64>,
    source_times: &[DateTime<Utc>],
    times: &[DateTime<Utc>],
    locations: &[Result<usize, usize>],
) -> Array2<f64> {
    let mut output = Array2::zeros((times.len(), 6));
    for (i, location) in locations.iter().enumerate() {
        match *location {
            Ok(index) => output.row_mut(i).assign(&states.row(index)),
            Err(right) => {
                // Only copy the bracket, not a full orbit, for scalar queries.
                let bracket = states.slice(s![right - 1..=right, ..]).to_owned();
                let interpolated = hermite_interpolate(
                    &times[i..i + 1],
                    &source_times[right - 1..=right],
                    &bracket,
                );
                output.row_mut(i).assign(&interpolated.row(0));
            }
        }
    }
    output
}

/// Locate `times` on `source_times` and sample `states` there (no extrapolation).
pub(crate) fn sample_states_at(
    states: &Array2<f64>,
    source_times: &[DateTime<Utc>],
    times: &[DateTime<Utc>],
) -> PyResult<Array2<f64>> {
    let locations = locate_times(source_times, times)?;
    Ok(sample_states(states, source_times, times, &locations))
}

impl SampledEphemeris {
    pub(crate) fn new(
        source: &dyn EphemerisBase,
        times: &[DateTime<Utc>],
        locations: &[Result<usize, usize>],
    ) -> PyResult<Self> {
        let source_times = source
            .data()
            .times
            .as_deref()
            .ok_or_else(|| PyValueError::new_err("No ephemeris timestamps available"))?;
        let sample = |states: &Array2<f64>| sample_states(states, source_times, times, locations);
        let mut data = EphemerisData::new();
        data.times = Some(times.to_vec());
        data.gcrs = source.data().gcrs.as_ref().map(sample);
        data.sun_gcrs = source.data().sun_gcrs.as_ref().map(sample);
        data.moon_gcrs = source.data().moon_gcrs.as_ref().map(sample);
        // Preserve the source's Earth-fixed frame / polar-motion convention.
        // Derived angles, radii and geodetic caches are recomputed from these states.
        Ok(Self {
            data,
            itrs: source.get_itrs_data().map(sample),
            itrs_skycoord: OnceLock::new(),
        })
    }
}

impl EphemerisBase for SampledEphemeris {
    fn data(&self) -> &EphemerisData {
        &self.data
    }
    fn data_mut(&mut self) -> &mut EphemerisData {
        &mut self.data
    }
    fn get_itrs_data(&self) -> Option<&Array2<f64>> {
        self.itrs.as_ref()
    }
    fn get_itrs_skycoord_ref(&self) -> Option<&Py<PyAny>> {
        self.itrs_skycoord.get()
    }
    fn set_itrs_skycoord_cache(&self, value: Py<PyAny>) -> Result<(), Py<PyAny>> {
        self.itrs_skycoord.set(value)
    }
    fn radec_to_altaz(
        &self,
        ra: f64,
        dec: f64,
        indices: Option<&[usize]>,
    ) -> PyResult<Array2<f64>> {
        crate::utils::celestial::radec_to_altaz(ra, dec, self, indices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, TimeZone};
    use ndarray::array;

    fn grid() -> (Vec<DateTime<Utc>>, Array2<f64>) {
        let t0 = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let times = (0..3).map(|i| t0 + Duration::seconds(60 * i)).collect();
        // x = t^2 / 60 (km), vx = 2t / 60; y, z linear; Hermite is exact for cubics.
        let states = array![
            [0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            [60.0, 60.0, 1.0, 2.0, 1.0, 0.0],
            [240.0, 120.0, 1.0, 4.0, 1.0, 0.0],
        ];
        (times, states)
    }

    #[test]
    fn sample_states_copies_grid_rows_and_interpolates_between() {
        let (src, states) = grid();
        let mid = src[0] + Duration::seconds(90);
        // Unsorted with a duplicate: output follows input order.
        let req = vec![src[2], mid, src[0], mid];
        let out = sample_states_at(&states, &src, &req).unwrap();
        assert_eq!(out.nrows(), 4);
        assert_eq!(out.row(0), states.row(2));
        assert_eq!(out.row(2), states.row(0));
        assert_eq!(out.row(1), out.row(3));
        assert!((out[[1, 0]] - 135.0).abs() < 1e-9);
        assert!((out[[1, 1]] - 90.0).abs() < 1e-9);
        assert!((out[[1, 3]] - 3.0).abs() < 1e-9);
        assert!((out[[1, 4]] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn sample_states_rejects_out_of_range() {
        let (src, states) = grid();
        assert!(sample_states_at(&states, &src, &[src[0] - Duration::seconds(1)]).is_err());
        assert!(sample_states_at(&states, &src, &[src[2] + Duration::seconds(1)]).is_err());
        assert!(sample_states_at(&states, &src, &[src[0], src[2]]).is_ok());
    }
}
