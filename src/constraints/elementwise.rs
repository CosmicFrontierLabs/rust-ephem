//! Elementwise (paired) constraint evaluation: element `i` is the target
//! `(ra[i], dec[i])` at the time `time[i]`. Each output holds N samples, never
//! the N x N Cartesian grid that `in_constraint_batch` would produce.
//!
//! Roll-independent elements go through `ConstraintEvaluator::evaluate_diagonal`.
//! Elements that need roll semantics (an explicit spacecraft roll, or a free
//! roll on a roll-dependent tree) are grouped by (time, roll) and evaluated with
//! the same batch kernels `in_constraint_batch` / `evaluate_batch` use, restricted
//! to the one timestamp of each group, so the result is identical to the
//! diagonal of the full grid.
use super::core::{sweep_rolls_with_attribution, ConstraintEvaluator, EvaluationRoll};
use super::diagonal::DiagonalEvaluation;
use crate::ephemeris::ephemeris_common::EphemerisBase;
use crate::utils::time_utils::utc_to_python_datetime;
use chrono::{DateTime, Utc};
use numpy::IntoPyArray;
use pyo3::prelude::*;
use std::collections::{BTreeMap, HashMap};

/// Scatter per-subset results into full-length (N) outputs.
struct Accumulator {
    n: usize,
    violated: Vec<bool>,
    named: HashMap<String, Vec<bool>>,
    values: HashMap<String, Vec<f64>>,
}

impl Accumulator {
    fn new(n: usize) -> Self {
        Self {
            n,
            violated: vec![false; n],
            named: HashMap::new(),
            values: HashMap::new(),
        }
    }

    fn place_violated(&mut self, positions: &[usize], violated: impl Iterator<Item = bool>) {
        for (&i, v) in positions.iter().zip(violated) {
            self.violated[i] = v;
        }
    }

    fn place_named(
        &mut self,
        positions: &[usize],
        key: String,
        column: impl Iterator<Item = bool>,
    ) {
        let n = self.n;
        let target = self.named.entry(key).or_insert_with(|| vec![false; n]);
        for (&i, v) in positions.iter().zip(column) {
            target[i] = v;
        }
    }

    fn place_values(
        &mut self,
        positions: &[usize],
        key: String,
        column: impl Iterator<Item = f64>,
    ) {
        let n = self.n;
        let target = self.values.entry(key).or_insert_with(|| vec![f64::NAN; n]);
        for (&i, v) in positions.iter().zip(column) {
            target[i] = v;
        }
    }
}

/// Evaluate element `i` = (`ras[i]`, `decs[i]`) at ephemeris row `indices[i]`.
///
/// `rolls[i]` is `Some(roll)` for a fixed spacecraft roll (as
/// `in_constraint_batch(target_rolls=...)`), or `None` for the free-roll
/// semantics of `in_constraint_batch(target_rolls=None)`.
///
/// With `with_values = false` only `violated` is meaningful (named/values may
/// be partial or empty); this skips the continuous-value computation on the
/// grouped roll path.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_elements(
    evaluator: &dyn ConstraintEvaluator,
    ephemeris: &dyn EphemerisBase,
    ras: &[f64],
    decs: &[f64],
    indices: &[usize],
    rolls: Option<&[Option<f64>]>,
    n_roll_samples: usize,
    with_values: bool,
) -> PyResult<DiagonalEvaluation> {
    let n = ras.len();
    if decs.len() != n || indices.len() != n || rolls.is_some_and(|r| r.len() != n) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "target_ras, target_decs, times and target_rolls must have the same length",
        ));
    }
    let roll_dependent = evaluator.is_roll_dependent();
    let roll_of = |i: usize| rolls.and_then(|r| r[i]);

    // Fast path: the whole input is served by the paired kernel. Also used for
    // an empty input so the value/cause schema is still reported.
    if n == 0 || (!roll_dependent && (0..n).all(|i| roll_of(i).is_none())) {
        return evaluator.evaluate_diagonal(ephemeris, ras, decs, Some(indices));
    }

    let mut acc = Accumulator::new(n);
    let mut cause_value_keys = None;

    // Elements without a fixed roll on a roll-independent tree still use the
    // paired kernel; everything else is grouped by (time, roll).
    let mut diagonal_positions = Vec::new();
    let mut groups: BTreeMap<(usize, Option<u64>), Vec<usize>> = BTreeMap::new();
    for (i, &time_index) in indices.iter().enumerate() {
        match roll_of(i) {
            None if !roll_dependent => diagonal_positions.push(i),
            roll => groups
                .entry((time_index, roll.map(f64::to_bits)))
                .or_default()
                .push(i),
        }
    }

    if !diagonal_positions.is_empty() {
        let pick = |src: &[f64]| {
            diagonal_positions
                .iter()
                .map(|&i| src[i])
                .collect::<Vec<_>>()
        };
        let sub_indices: Vec<usize> = diagonal_positions.iter().map(|&i| indices[i]).collect();
        let result =
            evaluator.evaluate_diagonal(ephemeris, &pick(ras), &pick(decs), Some(&sub_indices))?;
        acc.place_violated(&diagonal_positions, result.violated.into_iter());
        for (key, column) in result.named {
            acc.place_named(&diagonal_positions, key, column.into_iter());
        }
        for (key, column) in result.values {
            acc.place_values(&diagonal_positions, key, column.into_iter());
        }
        cause_value_keys = Some(result.cause_value_keys);
    }

    for ((time_index, roll_bits), positions) in groups {
        let roll = roll_bits.map(f64::from_bits);
        let group_ras: Vec<f64> = positions.iter().map(|&i| ras[i]).collect();
        let group_decs: Vec<f64> = positions.iter().map(|&i| decs[i]).collect();
        let selected = [time_index];
        if with_values {
            // Mirrors `eval_batch_with_ephemeris` (the `evaluate_batch` kernel).
            let swept = match roll {
                Some(roll) => evaluator.in_constraint_batch_with_causes(
                    ephemeris,
                    &group_ras,
                    &group_decs,
                    Some(&selected),
                    Some(EvaluationRoll::Fixed(roll)),
                )?,
                None => sweep_rolls_with_attribution(
                    evaluator,
                    ephemeris,
                    &group_ras,
                    &group_decs,
                    Some(&selected),
                    n_roll_samples,
                )?,
            };
            acc.place_violated(&positions, swept.violated.column(0).iter().copied());
            for (key, mask) in swept.named {
                acc.place_named(&positions, key, mask.column(0).iter().copied());
            }
            let values = evaluator.compute_named_values_at_roll(
                ephemeris,
                &group_ras,
                &group_decs,
                Some(&selected),
                roll,
            )?;
            for (key, array) in values {
                acc.place_values(&positions, key, array.column(0).iter().copied());
            }
        } else {
            // Mirrors the `in_constraint_batch` kernels exactly.
            let mask = match roll {
                Some(roll) => evaluator.in_constraint_batch_at_roll(
                    ephemeris,
                    &group_ras,
                    &group_decs,
                    Some(&selected),
                    EvaluationRoll::Fixed(roll),
                )?,
                None => evaluator.in_constraint_batch_constrained_at_every_roll(
                    ephemeris,
                    &group_ras,
                    &group_decs,
                    Some(&selected),
                    n_roll_samples,
                )?,
            };
            acc.place_violated(&positions, mask.column(0).iter().copied());
        }
    }

    let cause_value_keys = match cause_value_keys {
        Some(keys) => keys,
        None if with_values => {
            evaluator.compute_cause_value_keys(ephemeris, &[0.0], &[0.0], Some(&[indices[0]]))?
        }
        None => HashMap::new(),
    };
    Ok(DiagonalEvaluation {
        violated: acc.violated,
        named: acc.named,
        values: acc.values,
        cause_value_keys,
    })
}

/// Result of `Constraint.evaluate_elementwise`: one entry per input element
/// `(target_ras[i], target_decs[i], times[i])`, in input order.
#[pyclass(name = "ElementwiseResult")]
pub struct ElementwiseResult {
    /// Constraint name/description
    #[pyo3(get)]
    pub constraint_name: String,
    /// Map from a `component_violated` cause tag to the `constraint_values` key(s)
    /// it corresponds to (see `ConstraintResult.cause_value_keys`).
    #[pyo3(get)]
    pub cause_value_keys: HashMap<String, Vec<String>>,
    times: Vec<DateTime<Utc>>,
    ras: Vec<f64>,
    decs: Vec<f64>,
    constraint_vec: Vec<bool>,
    constraint_values: HashMap<String, Vec<f64>>,
    component_violated: HashMap<String, Vec<bool>>,
}

impl ElementwiseResult {
    pub fn new(
        constraint_name: String,
        times: Vec<DateTime<Utc>>,
        ras: Vec<f64>,
        decs: Vec<f64>,
        evaluation: DiagonalEvaluation,
    ) -> Self {
        Self {
            constraint_name,
            cause_value_keys: evaluation.cause_value_keys,
            times,
            ras,
            decs,
            constraint_vec: evaluation.violated,
            constraint_values: evaluation.values,
            component_violated: evaluation.named,
        }
    }
}

#[pymethods]
impl ElementwiseResult {
    fn __repr__(&self) -> String {
        format!(
            "ElementwiseResult(constraint='{}', n={}, n_violated={})",
            self.constraint_name,
            self.constraint_vec.len(),
            self.constraint_vec.iter().filter(|&&v| v).count()
        )
    }

    fn __len__(&self) -> usize {
        self.constraint_vec.len()
    }

    /// numpy bool array (N,): True where the constraint is VIOLATED.
    #[getter]
    fn constraint_array(&self, py: Python) -> Py<PyAny> {
        self.constraint_vec
            .clone()
            .into_pyarray(py)
            .into_any()
            .unbind()
    }

    /// numpy bool array (N,): True where the target is visible (inverse of
    /// `constraint_array`).
    #[getter]
    fn visibility_flags(&self, py: Python) -> Py<PyAny> {
        self.constraint_vec
            .iter()
            .map(|&v| !v)
            .collect::<Vec<_>>()
            .into_pyarray(py)
            .into_any()
            .unbind()
    }

    /// dict[str, numpy float array (N,)] of the named continuous values, with the
    /// same keys `evaluate` / `evaluate_batch` report for this constraint.
    #[getter]
    fn constraint_values(&self, py: Python) -> PyResult<Py<PyAny>> {
        let dict = pyo3::types::PyDict::new(py);
        for (key, values) in &self.constraint_values {
            dict.set_item(key, values.clone().into_pyarray(py))?;
        }
        Ok(dict.into_any().unbind())
    }

    /// dict[str, numpy bool array (N,)]: per-leaf violation masks keyed by
    /// cause tag (e.g. `sun`, `moon`), True where that leaf is violated.
    #[getter]
    fn component_violated(&self, py: Python) -> PyResult<Py<PyAny>> {
        let dict = pyo3::types::PyDict::new(py);
        for (key, values) in &self.component_violated {
            dict.set_item(key, values.clone().into_pyarray(py))?;
        }
        Ok(dict.into_any().unbind())
    }

    /// numpy float array (N,) of target right ascensions in degrees.
    #[getter]
    fn ras(&self, py: Python) -> Py<PyAny> {
        self.ras.clone().into_pyarray(py).into_any().unbind()
    }

    /// numpy float array (N,) of target declinations in degrees.
    #[getter]
    fn decs(&self, py: Python) -> Py<PyAny> {
        self.decs.clone().into_pyarray(py).into_any().unbind()
    }

    /// numpy object array (N,) of the UTC datetime each element was evaluated at.
    #[getter]
    fn timestamp(&self, py: Python) -> PyResult<Py<PyAny>> {
        let np = pyo3::types::PyModule::import(py, "numpy")?;
        let list = pyo3::types::PyList::empty(py);
        for dt in &self.times {
            list.append(utc_to_python_datetime(py, dt)?)?;
        }
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("dtype", "object")?;
        Ok(np.getattr("array")?.call((list,), Some(&kwargs))?.unbind())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::bright_star::BrightStarConfig;
    use crate::constraints::core::ConstraintConfig;
    use crate::constraints::solar_roll::SolarRollConfig;
    use crate::constraints::sun_proximity::SunProximityConfig;
    use crate::constraints::test_support::TestEphemeris;

    const N_ROLL: usize = 12;

    fn elements(n_times: usize) -> (Vec<f64>, Vec<f64>, Vec<usize>) {
        let n = 23;
        let ras = (0..n).map(|i| (i as f64 * 47.0) % 360.0).collect();
        let decs = (0..n).map(|i| ((i as f64 * 31.0) % 160.0) - 80.0).collect();
        let indices = (0..n).map(|i| (i * 7 + 3) % n_times).collect();
        (ras, decs, indices)
    }

    fn evaluators() -> Vec<Box<dyn ConstraintEvaluator>> {
        let polygon = vec![[-10.0, -5.0], [10.0, -5.0], [10.0, 25.0], [-10.0, 25.0]];
        vec![
            SunProximityConfig {
                min_angle: 70.0,
                max_angle: Some(150.0),
            }
            .to_evaluator(),
            BrightStarConfig {
                stars: vec![[0.0, 0.0], [90.0, 10.0], [200.0, -30.0]],
                fov_radius: None,
                fov_polygon: Some(polygon),
                roll_deg: None,
            }
            .to_evaluator(),
            SolarRollConfig {
                tolerance_deg: 20.0,
                roll_deg: None,
                panel_normal: [0.0, 1.0, 0.0],
            }
            .to_evaluator(),
        ]
    }

    #[test]
    fn free_roll_elements_match_the_grid_diagonal() {
        let eph = TestEphemeris::new(9);
        let (ras, decs, indices) = elements(9);
        for evaluator in evaluators() {
            let grid = evaluator
                .in_constraint_batch_constrained_at_every_roll(&eph, &ras, &decs, None, N_ROLL)
                .unwrap();
            let values = evaluator
                .compute_named_values_at_roll(&eph, &ras, &decs, None, None)
                .unwrap();
            for with_values in [false, true] {
                let result = evaluate_elements(
                    &*evaluator,
                    &eph,
                    &ras,
                    &decs,
                    &indices,
                    None,
                    N_ROLL,
                    with_values,
                )
                .unwrap();
                let expected: Vec<bool> = indices
                    .iter()
                    .enumerate()
                    .map(|(i, &t)| grid[[i, t]])
                    .collect();
                assert_eq!(result.violated, expected, "{}", evaluator.name());
                if with_values {
                    assert_eq!(result.values.len(), values.len());
                    for (key, column) in &result.values {
                        for (i, &t) in indices.iter().enumerate() {
                            let want = values[key][[i, t]];
                            assert!(
                                column[i] == want || (column[i].is_nan() && want.is_nan()),
                                "{key}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn fixed_and_free_rolls_mix_per_element() {
        let eph = TestEphemeris::new(9);
        let (ras, decs, indices) = elements(9);
        let rolls: Vec<Option<f64>> = (0..ras.len())
            .map(|i| (i % 3 != 0).then_some(i as f64 * 29.0))
            .collect();
        for evaluator in evaluators() {
            let free = evaluator
                .in_constraint_batch_constrained_at_every_roll(&eph, &ras, &decs, None, N_ROLL)
                .unwrap();
            let result = evaluate_elements(
                &*evaluator,
                &eph,
                &ras,
                &decs,
                &indices,
                Some(&rolls),
                N_ROLL,
                true,
            )
            .unwrap();
            for (i, &t) in indices.iter().enumerate() {
                let expected = match rolls[i] {
                    None => free[[i, t]],
                    Some(roll) => evaluator
                        .in_constraint_batch_at_roll(
                            &eph,
                            &ras[i..=i],
                            &decs[i..=i],
                            Some(&[t]),
                            EvaluationRoll::Fixed(roll),
                        )
                        .unwrap()[[0, 0]],
                };
                assert_eq!(result.violated[i], expected, "{} #{i}", evaluator.name());
            }
        }
    }

    #[test]
    fn mismatched_lengths_are_rejected() {
        let eph = TestEphemeris::new(3);
        let evaluator = &evaluators()[0];
        assert!(
            evaluate_elements(&**evaluator, &eph, &[0.0], &[0.0], &[0, 1], None, 1, true).is_err()
        );
        assert!(evaluate_elements(
            &**evaluator,
            &eph,
            &[0.0],
            &[0.0],
            &[0],
            Some(&[None, None]),
            1,
            true
        )
        .is_err());
    }
}
