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
        let sample = |states: &Array2<f64>| {
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
        };
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
