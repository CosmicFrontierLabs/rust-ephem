//! Paired target/time evaluation for moving bodies. Each output contains N
//! samples, never the Cartesian product of N directions and N timestamps.
use super::core::{value_key_prefix, ConstraintEvaluator};
use crate::ephemeris::ephemeris_common::{EphemerisBase, EphemerisData};
use ndarray::{s, Array1, Array2};
use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::OnceLock;

pub struct DiagonalEvaluation {
    pub violated: Vec<bool>,
    pub named: HashMap<String, Vec<bool>>,
    pub values: HashMap<String, Vec<f64>>,
    pub cause_value_keys: HashMap<String, Vec<String>>,
}

impl DiagonalEvaluation {
    pub fn leaf(name: &str, violated: Vec<bool>, values: HashMap<String, Vec<f64>>) -> Self {
        let tag = value_key_prefix(name);
        let mut keys: Vec<_> = values.keys().cloned().collect();
        keys.sort();
        Self {
            named: HashMap::from([(tag.clone(), violated.clone())]),
            cause_value_keys: HashMap::from([(tag, keys)]),
            violated,
            values,
        }
    }
}

pub fn time_indices(
    ephemeris: &dyn EphemerisBase,
    ras: &[f64],
    decs: &[f64],
    indices: Option<&[usize]>,
) -> PyResult<Vec<usize>> {
    let times = ephemeris
        .data()
        .times
        .as_ref()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("No times available"))?;
    if ras.len() != decs.len() || indices.is_some_and(|idx| idx.len() != ras.len()) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "target_ras, target_decs and selected timestamps must have the same length",
        ));
    }
    let indices = indices.map_or_else(|| (0..ras.len()).collect(), <[usize]>::to_vec);
    if indices.iter().any(|&i| i >= times.len()) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Not enough ephemeris time steps for diagonal evaluation",
        ));
    }
    Ok(indices)
}

/// Safe linear fallback for leaves without a dedicated paired kernel. A
/// one-sample ephemeris also bounds allocations inside existing batch helpers:
/// selecting a timestamp from the full ephemeris for each target would still
/// copy O(N) ephemeris data N times.
pub fn evaluate<T: ConstraintEvaluator + ?Sized>(
    evaluator: &T,
    ephemeris: &dyn EphemerisBase,
    ras: &[f64],
    decs: &[f64],
    indices: Option<&[usize]>,
) -> PyResult<DiagonalEvaluation> {
    let indices = time_indices(ephemeris, ras, decs, indices)?;
    let mut result = DiagonalEvaluation {
        violated: Vec::with_capacity(ras.len()),
        named: HashMap::new(),
        values: HashMap::new(),
        cause_value_keys: HashMap::new(),
    };
    for (i, &time_index) in indices.iter().enumerate() {
        let sample = SampleEphemeris::new(ephemeris, time_index);
        let mask = evaluator.in_constraint_batch_with_causes(
            &sample,
            &ras[i..i + 1],
            &decs[i..i + 1],
            None,
            None,
        )?;
        let values =
            evaluator.compute_named_values(&sample, &ras[i..i + 1], &decs[i..i + 1], None)?;
        if i == 0 {
            result.cause_value_keys = evaluator.compute_cause_value_keys(
                &sample,
                &ras[i..i + 1],
                &decs[i..i + 1],
                None,
            )?;
        }
        result.violated.push(mask.violated[[0, 0]]);
        for (key, values) in mask.named {
            result
                .named
                .entry(key)
                .or_insert_with(|| Vec::with_capacity(ras.len()))
                .push(values[[0, 0]]);
        }
        for (key, values) in values {
            result
                .values
                .entry(key)
                .or_insert_with(|| Vec::with_capacity(ras.len()))
                .push(values[[0, 0]]);
        }
    }
    // Preserve the value/cause schema for an empty explicit track.
    if ras.is_empty() {
        if ephemeris
            .data()
            .times
            .as_ref()
            .is_some_and(|times| times.is_empty())
        {
            return Ok(DiagonalEvaluation::leaf(
                &evaluator.name(),
                Vec::new(),
                HashMap::new(),
            ));
        }
        let sample = SampleEphemeris::new(ephemeris, 0);
        let mask =
            evaluator.in_constraint_batch_with_causes(&sample, &[0.0], &[0.0], None, None)?;
        result.named = mask
            .named
            .into_keys()
            .map(|key| (key, Vec::new()))
            .collect();
        result.values = evaluator
            .compute_named_values(&sample, &[0.0], &[0.0], None)?
            .into_keys()
            .map(|key| (key, Vec::new()))
            .collect();
        result.cause_value_keys =
            evaluator.compute_cause_value_keys(&sample, &[0.0], &[0.0], None)?;
    }
    Ok(result)
}

pub(crate) struct SampleEphemeris {
    data: EphemerisData,
    itrs: Option<Array2<f64>>,
    itrs_skycoord: OnceLock<Py<PyAny>>,
}

impl SampleEphemeris {
    pub(crate) fn new(ephemeris: &dyn EphemerisBase, index: usize) -> Self {
        let original = ephemeris.data();
        let row = |array: &Array2<f64>| array.slice(s![index..index + 1, ..]).to_owned();
        let mut data = EphemerisData::new();
        data.times = original.times.as_ref().map(|times| vec![times[index]]);
        data.gcrs = original.gcrs.as_ref().map(row);
        data.sun_gcrs = original.sun_gcrs.as_ref().map(row);
        data.moon_gcrs = original.moon_gcrs.as_ref().map(row);
        for (source, destination) in [
            (&original.latitude_deg_cache, &data.latitude_deg_cache),
            (&original.longitude_deg_cache, &data.longitude_deg_cache),
            (&original.latitude_rad_cache, &data.latitude_rad_cache),
            (&original.longitude_rad_cache, &data.longitude_rad_cache),
            (&original.height_cache, &data.height_cache),
            (&original.height_km_cache, &data.height_km_cache),
            (&original.sun_altitudes_cache, &data.sun_altitudes_cache),
        ] {
            if let Some(values) = source.get() {
                let _ = destination.set(Array1::from_vec(vec![values[index]]));
            }
        }
        Self {
            data,
            itrs: ephemeris.get_itrs_data().map(row),
            itrs_skycoord: OnceLock::new(),
        }
    }
}

impl EphemerisBase for SampleEphemeris {
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

#[allow(clippy::too_many_arguments)]
pub fn proximity(
    name: &str,
    value_key: &str,
    ras: &[f64],
    decs: &[f64],
    indices: &[usize],
    body: &Array2<f64>,
    observer: &Array2<f64>,
    min_angle: f64,
    max_angle: Option<f64>,
) -> DiagonalEvaluation {
    let min_cos = min_angle.to_radians().cos();
    let max_cos = max_angle.map(|max| max.to_radians().cos());
    let mut violated = Vec::with_capacity(ras.len());
    let mut angles = Vec::with_capacity(ras.len());
    for (i, &t) in indices.iter().enumerate() {
        let target = crate::utils::vector_math::radec_to_unit_vector(ras[i], decs[i]);
        let rel = [
            body[[t, 0]] - observer[[t, 0]],
            body[[t, 1]] - observer[[t, 1]],
            body[[t, 2]] - observer[[t, 2]],
        ];
        let norm = crate::utils::vector_math::vector_magnitude(&rel);
        let unit = [rel[0] / norm, rel[1] / norm, rel[2] / norm];
        let cosine = crate::utils::vector_math::dot_product(&target, &unit);
        violated.push(cosine > min_cos || max_cos.is_some_and(|max| cosine < max));
        // Match compute_angle_deg_batch's fallback for a zero body vector.
        let value_cosine = if norm > 0.0 { cosine } else { target[0] };
        angles.push(value_cosine.clamp(-1.0, 1.0).acos().to_degrees());
    }
    DiagonalEvaluation::leaf(
        name,
        violated,
        HashMap::from([(value_key.to_string(), angles)]),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::core::ConstraintResult;
    use chrono::{TimeZone, Utc};
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct RecordingLeaf {
        masks: AtomicUsize,
    }

    impl ConstraintEvaluator for RecordingLeaf {
        fn evaluate(
            &self,
            _: &dyn EphemerisBase,
            _: f64,
            _: f64,
            _: Option<&[usize]>,
        ) -> PyResult<ConstraintResult> {
            unreachable!("the paired path must use the boolean batch kernel")
        }
        fn in_constraint_batch(
            &self,
            ephemeris: &dyn EphemerisBase,
            ras: &[f64],
            _: &[f64],
            _: Option<&[usize]>,
        ) -> PyResult<Array2<bool>> {
            assert_eq!(ras.len(), 1);
            assert_eq!(ephemeris.data().times.as_ref().unwrap().len(), 1);
            self.masks.fetch_add(1, Ordering::Relaxed);
            Ok(Array2::from_elem(
                (1, 1),
                ras[0] > ephemeris.data().gcrs.as_ref().unwrap()[[0, 0]],
            ))
        }
        fn compute_named_values(
            &self,
            ephemeris: &dyn EphemerisBase,
            ras: &[f64],
            _: &[f64],
            _: Option<&[usize]>,
        ) -> PyResult<HashMap<String, Array2<f64>>> {
            assert_eq!(ras.len(), 1);
            assert_eq!(ephemeris.data().times.as_ref().unwrap().len(), 1);
            Ok(HashMap::from([(
                "sample".to_string(),
                Array2::from_elem((1, 1), ephemeris.data().gcrs.as_ref().unwrap()[[0, 0]]),
            )]))
        }
        fn name(&self) -> String {
            "RecordingLeaf".to_string()
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    #[test]
    fn fallback_visits_each_pair_once_without_a_cartesian_allocation() {
        let n = 64;
        let mut data = EphemerisData::new();
        let epoch = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        data.times = Some(
            (0..n)
                .map(|i| epoch + chrono::Duration::seconds(i as i64))
                .collect(),
        );
        data.gcrs = Some(Array2::from_shape_fn((n, 6), |(i, _)| i as f64));
        let ephemeris = SampleEphemeris {
            data,
            itrs: None,
            itrs_skycoord: OnceLock::new(),
        };
        let leaf = RecordingLeaf {
            masks: AtomicUsize::new(0),
        };
        let indices: Vec<_> = (0..n).rev().collect();
        let result = leaf
            .evaluate_diagonal(&ephemeris, &vec![30.0; n], &vec![0.0; n], Some(&indices))
            .unwrap();
        assert_eq!(leaf.masks.load(Ordering::Relaxed), n);
        assert_eq!(
            result.values["sample"],
            indices.iter().map(|&i| i as f64).collect::<Vec<_>>()
        );
        assert_eq!(
            result.violated,
            indices.iter().map(|&i| i < 30).collect::<Vec<_>>()
        );
        assert_eq!(result.named["recordingleaf"], result.violated);
    }
}
