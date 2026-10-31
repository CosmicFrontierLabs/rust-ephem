//! Evaluation-local geometry for coordinated roll sweeps. Cell indices always
//! refer to the original target/time grid, even as resolved cells are removed.
use super::core::{ConstraintEvaluator, EvaluationRoll};
use crate::ephemeris::ephemeris_common::EphemerisBase;
use ndarray::Array2;
use pyo3::PyResult;
use std::ops::{Index, IndexMut};

/// Prepared attributed sweep, owned by one evaluation only. Cached results are
/// safe without fingerprints because the ephemeris and targets stay borrowed.
pub struct PreparedAttributedRollEvaluator<'a>(
    Box<dyn Fn(f64) -> PyResult<super::core::BatchWithCauses> + 'a>,
);

impl<'a> PreparedAttributedRollEvaluator<'a> {
    pub fn new(f: impl Fn(f64) -> PyResult<super::core::BatchWithCauses> + 'a) -> Self {
        Self(Box::new(f))
    }

    pub fn evaluate(&self, roll: f64) -> PyResult<super::core::BatchWithCauses> {
        (self.0)(roll)
    }
}

pub fn prepare_with_causes<'a, T: ConstraintEvaluator + ?Sized>(
    evaluator: &'a T,
    ephemeris: &'a dyn EphemerisBase,
    ras: &'a [f64],
    decs: &'a [f64],
    indices: Option<&'a [usize]>,
) -> PyResult<PreparedAttributedRollEvaluator<'a>> {
    if !evaluator.is_roll_dependent() {
        // Sweep candidates preserve configured fixed rolls, so these masks and
        // their cause tags are identical for every candidate angle.
        let result = evaluator.in_constraint_batch_with_causes(
            ephemeris,
            ras,
            decs,
            indices,
            Some(EvaluationRoll::Sweep(0.0)),
        )?;
        return Ok(PreparedAttributedRollEvaluator::new(move |_| {
            Ok(result.clone())
        }));
    }
    if let Some(prepared) =
        evaluator.prepare_attributed_roll_sweep(ephemeris, ras, decs, indices)?
    {
        return Ok(prepared);
    }
    Ok(PreparedAttributedRollEvaluator::new(move |roll| {
        evaluator.in_constraint_batch_with_causes(
            ephemeris,
            ras,
            decs,
            indices,
            Some(EvaluationRoll::Sweep(roll)),
        )
    }))
}

/// Adapt an existing prepared boolean kernel for a leaf's attributed sweep.
/// Geometry is retained per chunk; cell coordinates are regenerated in bounded
/// scratch space rather than stored for the entire target/time grid. Only leaves
/// use this adapter because their cause mask equals their combined mask.
pub fn prepare_attributed_leaf<'a, T: ConstraintEvaluator + ?Sized>(
    evaluator: &'a T,
    ephemeris: &'a dyn EphemerisBase,
    ras: &'a [f64],
    decs: &'a [f64],
    indices: Option<&'a [usize]>,
    time_invariant: bool,
) -> PyResult<Option<PreparedAttributedRollEvaluator<'a>>> {
    if ras.len() != decs.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "target_ras and target_decs must have the same length",
        ));
    }
    let n_times = indices.map_or_else(
        || ephemeris.get_times().map(|times| times.len()),
        |idx| Ok(idx.len()),
    )?;
    let n_targets = ras.len();
    // Catalog-star projections are time-invariant: retain one cell per target
    // and broadcast the resulting row states only after evaluating a roll.
    let n_columns = if time_invariant { 1 } else { n_times };
    let len = n_targets * n_columns;
    if n_times == 0 || len == 0 || !evaluator.is_roll_dependent() {
        return Ok(None);
    }
    let mut chunks = Vec::new();
    for start in (0..len).step_by(MAX_PREPARED_CELLS) {
        let end = (start + MAX_PREPARED_CELLS).min(len);
        let cells: Vec<_> = (start..end)
            .map(|index| (index / n_columns, index % n_columns))
            .collect();
        let Some(prepared) = evaluator.prepare_roll_sweep(ephemeris, ras, decs, indices, &cells)?
        else {
            return Ok(None);
        };
        chunks.push((start, end, prepared));
    }
    let tag = super::core::value_key_prefix(&evaluator.name());
    Ok(Some(PreparedAttributedRollEvaluator::new(move |roll| {
        let mut values = vec![false; len];
        for (start, end, prepared) in &chunks {
            let cells: Vec<_> = (*start..*end)
                .map(|index| (index / n_columns, index % n_columns))
                .collect();
            let chunk = prepared.evaluate(&cells, roll)?;
            values[*start..*end].copy_from_slice(&chunk);
        }
        let violated = if time_invariant {
            Array2::from_shape_fn((n_targets, n_times), |(i, _)| values[i])
        } else {
            Array2::from_shape_vec((n_targets, n_times), values).unwrap()
        };
        Ok(super::core::BatchWithCauses {
            named: std::collections::HashMap::from([(tag.clone(), violated.clone())]),
            violated,
        })
    })))
}

pub type RollCell = (usize, usize);

/// Evaluate a time-invariant leaf once per active target at this roll. The
/// cache spans only this chunk's rows and is rebuilt at each candidate roll.
pub fn evaluate_per_target(
    cells: &[RollCell],
    first_row: usize,
    n_rows: usize,
    mut evaluate: impl FnMut(usize) -> bool,
) -> Vec<bool> {
    let mut answers = vec![None; n_rows];
    cells
        .iter()
        .map(|&(i, _)| *answers[i - first_row].get_or_insert_with(|| evaluate(i)))
        .collect()
}
// Bound temporary geometry independently of the size of the output matrix.
const MAX_PREPARED_CELLS: usize = 65_536;
type RollEvaluation<'a> = dyn Fn(&[RollCell], f64) -> PyResult<Vec<bool>> + 'a;

pub struct PreparedRollEvaluator<'a>(Box<RollEvaluation<'a>>);

impl<'a> PreparedRollEvaluator<'a> {
    pub fn new(f: impl Fn(&[RollCell], f64) -> PyResult<Vec<bool>> + 'a) -> Self {
        Self(Box::new(f))
    }

    pub fn evaluate(&self, cells: &[RollCell], roll_deg: f64) -> PyResult<Vec<bool>> {
        (self.0)(cells, roll_deg)
    }
}

pub fn prepare<'a, T: ConstraintEvaluator + ?Sized>(
    evaluator: &'a T,
    ephemeris: &'a dyn EphemerisBase,
    ras: &'a [f64],
    decs: &'a [f64],
    time_indices: Option<&'a [usize]>,
    cells: &[RollCell],
) -> PyResult<PreparedRollEvaluator<'a>> {
    let independent = !evaluator.is_roll_dependent();
    if !independent {
        if let Some(prepared) =
            evaluator.prepare_roll_sweep(ephemeris, ras, decs, time_indices, cells)?
        {
            return Ok(prepared);
        }
    }
    // Other evaluators, including boresight transforms, keep their existing
    // fixed-roll implementation. Limit its grid to rows/columns used by this
    // chunk rather than evaluating the entire original grid for every chunk.
    let mut rows: Vec<_> = cells.iter().map(|&(i, _)| i).collect();
    rows.sort_unstable();
    rows.dedup();
    let mut columns: Vec<_> = cells.iter().map(|&(_, j)| j).collect();
    columns.sort_unstable();
    columns.dedup();
    let n_times = time_indices.map_or_else(
        || {
            ephemeris
                .data()
                .times
                .as_ref()
                .map(Vec::len)
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("No times available"))
        },
        |indices| Ok(indices.len()),
    )?;
    // Index only the original chunk span. Resolve local coordinates once, then
    // use direct array access as cells are pruned across candidate rolls.
    let mut local_cells = PreparedGrid::new(n_times, cells, (0, 0));
    for &(i, j) in cells {
        local_cells[(i, j)] = (
            rows.binary_search(&i).unwrap(),
            columns.binary_search(&j).unwrap(),
        );
    }
    let subset_ras: Vec<_> = rows.iter().map(|&i| ras[i]).collect();
    let subset_decs: Vec<_> = rows.iter().map(|&i| decs[i]).collect();
    let indices: Vec<_> = columns
        .iter()
        .map(|&j| time_indices.map_or(j, |idx| idx[j]))
        .collect();
    // Independent children need one compact mask per chunk, never a full
    // original-grid mask per nested child and chunk.
    let independent_mask = if independent {
        Some(evaluator.in_constraint_batch(ephemeris, &subset_ras, &subset_decs, Some(&indices))?)
    } else {
        None
    };
    Ok(PreparedRollEvaluator::new(move |cells, roll| {
        let mask;
        let mask = match &independent_mask {
            Some(mask) => mask,
            None => {
                mask = evaluator.in_constraint_batch_at_roll(
                    ephemeris,
                    &subset_ras,
                    &subset_decs,
                    Some(&indices),
                    EvaluationRoll::Sweep(roll),
                )?;
                &mask
            }
        };
        Ok(cells.iter().map(|&cell| mask[local_cells[cell]]).collect())
    }))
}

/// A contiguous span of the original grid, storing only this chunk's geometry.
pub struct PreparedGrid<T> {
    values: Vec<T>,
    start: usize,
    n_times: usize,
}

impl<T: Clone> PreparedGrid<T> {
    pub fn new(n_times: usize, cells: &[RollCell], value: T) -> Self {
        let first = cells
            .iter()
            .map(|&(i, j)| i * n_times + j)
            .min()
            .unwrap_or(0);
        let end = cells
            .iter()
            .map(|&(i, j)| i * n_times + j + 1)
            .max()
            .unwrap_or(first);
        Self {
            values: vec![value; end - first],
            start: first,
            n_times,
        }
    }
}

impl<T> Index<RollCell> for PreparedGrid<T> {
    type Output = T;
    fn index(&self, (i, j): RollCell) -> &T {
        &self.values[i * self.n_times + j - self.start]
    }
}

impl<T> IndexMut<RollCell> for PreparedGrid<T> {
    fn index_mut(&mut self, (i, j): RollCell) -> &mut T {
        &mut self.values[i * self.n_times + j - self.start]
    }
}

pub fn sweep_chunks<'a>(
    result: &mut Array2<bool>,
    active_value: bool,
    n_roll_samples: usize,
    mut prepare: impl FnMut(&[RollCell]) -> PyResult<PreparedRollEvaluator<'a>>,
) -> PyResult<()> {
    let n_times = result.ncols();
    for start in (0..result.len()).step_by(MAX_PREPARED_CELLS) {
        let end = (start + MAX_PREPARED_CELLS).min(result.len());
        let cells: Vec<_> = (start..end)
            .filter_map(|index| {
                let cell = (index / n_times, index % n_times);
                (result[cell] == active_value).then_some(cell)
            })
            .collect();
        if cells.is_empty() {
            continue;
        }
        let prepared = prepare(&cells)?;
        for &cell in &cells {
            result[cell] = true;
        }
        sweep(&prepared, cells, result, n_roll_samples)?;
    }
    Ok(())
}

/// Keep only cells still violated after each candidate roll. Cleared cells never
/// need another geometry calculation, but all children share each candidate roll.
pub fn sweep(
    prepared: &PreparedRollEvaluator<'_>,
    mut cells: Vec<RollCell>,
    result: &mut Array2<bool>,
    n_roll_samples: usize,
) -> PyResult<()> {
    for step in 0..n_roll_samples {
        if cells.is_empty() {
            break;
        }
        let values = prepared.evaluate(&cells, step as f64 * (360.0 / n_roll_samples as f64))?;
        let mut index = 0;
        cells.retain(|&(i, j)| {
            let violated = values[index];
            index += 1;
            if !violated {
                result[[i, j]] = false;
            }
            violated
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::core::ConstraintResult;
    use crate::constraints::test_support::TestEphemeris;
    use std::cell::RefCell;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct RecordingLeaf {
        cells: AtomicUsize,
        dependent: bool,
    }

    impl ConstraintEvaluator for RecordingLeaf {
        fn evaluate(
            &self,
            _: &dyn EphemerisBase,
            _: f64,
            _: f64,
            _: Option<&[usize]>,
        ) -> PyResult<ConstraintResult> {
            unreachable!()
        }
        fn in_constraint_batch(
            &self,
            eph: &dyn EphemerisBase,
            ras: &[f64],
            _: &[f64],
            indices: Option<&[usize]>,
        ) -> PyResult<Array2<bool>> {
            let indices = indices.map_or_else(
                || (0..eph.data().times.as_ref().unwrap().len()).collect(),
                <[usize]>::to_vec,
            );
            let count = ras.len() * indices.len();
            assert!(count <= 2 * MAX_PREPARED_CELLS);
            self.cells.fetch_add(count, Ordering::Relaxed);
            Ok(Array2::from_shape_fn(
                (ras.len(), indices.len()),
                |(i, j)| ras[i] >= indices[j] as f64,
            ))
        }
        fn is_roll_dependent(&self) -> bool {
            self.dependent
        }
        fn name(&self) -> String {
            "RecordingLeaf".to_owned()
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    #[test]
    fn attributed_leaf_kernels_match_ordinary_masks_and_causes() {
        use crate::constraints::core::ConstraintConfig;
        use crate::constraints::{
            body_proximity::BodyProximityConfig, bright_star::BrightStarConfig,
            solar_roll::SolarRollConfig,
        };
        let eph = TestEphemeris::new(5);
        let polygon = vec![[-10.0, -5.0], [10.0, -5.0], [10.0, 5.0], [-10.0, 5.0]];
        let mut leaves = vec![
            (
                BodyProximityConfig {
                    body: "sun".into(),
                    min_angle: None,
                    max_angle: None,
                    fov_polygon: Some(polygon.clone()),
                    roll_deg: None,
                }
                .to_evaluator(),
                false,
            ),
            (
                BrightStarConfig {
                    stars: vec![[0.0, 0.0], [90.0, 80.0], [359.0, -2.0]],
                    fov_radius: None,
                    fov_polygon: Some(polygon),
                    roll_deg: None,
                }
                .to_evaluator(),
                true,
            ),
        ];
        for panel in [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.5, 0.5]] {
            leaves.push((
                SolarRollConfig {
                    tolerance_deg: 20.0,
                    roll_deg: None,
                    panel_normal: panel,
                }
                .to_evaluator(),
                false,
            ));
        }
        let ras = [0.0, 5.0, 90.0, 180.0, 359.0];
        let decs = [0.0, 2.0, 89.999, -89.999, -3.0];
        for (leaf, time_invariant) in leaves {
            for indices in [None, Some(vec![4, 1, 1, 0])] {
                let prepared = prepare_attributed_leaf(
                    &*leaf,
                    &eph,
                    &ras,
                    &decs,
                    indices.as_deref(),
                    time_invariant,
                )
                .unwrap()
                .unwrap();
                for roll in [0.0, 15.0, 90.0, 180.0, 345.0] {
                    let expected = leaf
                        .in_constraint_batch_with_causes(
                            &eph,
                            &ras,
                            &decs,
                            indices.as_deref(),
                            Some(EvaluationRoll::Sweep(roll)),
                        )
                        .unwrap();
                    let actual = prepared.evaluate(roll).unwrap();
                    assert_eq!(
                        actual.violated,
                        expected.violated,
                        "{} at {roll}",
                        leaf.name()
                    );
                    assert_eq!(actual.named, expected.named, "{} at {roll}", leaf.name());
                }
            }
        }
    }

    #[test]
    fn attributed_leaf_adapter_handles_multiple_chunks_and_empty_grids() {
        use crate::constraints::{core::ConstraintConfig, solar_roll::SolarRollConfig};
        let eph = TestEphemeris::new(13);
        let ras = vec![90.0; 5042];
        let decs = vec![30.0; ras.len()];
        let leaf = SolarRollConfig {
            tolerance_deg: 20.0,
            roll_deg: None,
            panel_normal: [0.0, 1.0, 0.0],
        }
        .to_evaluator();
        let prepared = prepare_attributed_leaf(&*leaf, &eph, &ras, &decs, None, false)
            .unwrap()
            .unwrap();
        for roll in [0.0, 90.0] {
            let expected = leaf
                .in_constraint_batch_with_causes(
                    &eph,
                    &ras,
                    &decs,
                    None,
                    Some(EvaluationRoll::Sweep(roll)),
                )
                .unwrap();
            let actual = prepared.evaluate(roll).unwrap();
            assert_eq!(actual.violated, expected.violated);
            assert_eq!(actual.named, expected.named);
        }
        assert!(prepare_attributed_leaf(&*leaf, &eph, &[], &[], None, false)
            .unwrap()
            .is_none());
        assert!(
            prepare_attributed_leaf(&*leaf, &eph, &ras, &decs, Some(&[]), false)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn attributed_body_and_time_invariant_leaves_cross_chunk_boundaries() {
        use crate::constraints::{
            body_proximity::BodyProximityConfig, bright_star::BrightStarConfig,
            core::ConstraintConfig,
        };
        let eph = TestEphemeris::new(13);
        let polygon = vec![[-10.0, -5.0], [10.0, -5.0], [10.0, 5.0], [-10.0, 5.0]];
        for (leaf, n_targets) in [
            (
                BrightStarConfig {
                    stars: vec![[90.0, 30.0], [91.0, 31.0]],
                    fov_radius: None,
                    fov_polygon: Some(polygon.clone()),
                    roll_deg: None,
                }
                .to_evaluator(),
                // Time-invariant preparation chunks targets, rather than cells.
                MAX_PREPARED_CELLS + 1,
            ),
            (
                BodyProximityConfig {
                    body: "sun".into(),
                    min_angle: None,
                    max_angle: None,
                    fov_polygon: Some(polygon),
                    roll_deg: None,
                }
                .to_evaluator(),
                MAX_PREPARED_CELLS / 13 + 1,
            ),
        ] {
            let ras: Vec<_> = (0..n_targets).map(|i| (i % 360) as f64).collect();
            let decs = vec![30.0; n_targets];
            let prepared = prepare_with_causes(&*leaf, &eph, &ras, &decs, None).unwrap();
            for roll in [0.0, 90.0] {
                let actual = prepared.evaluate(roll).unwrap();
                let expected = leaf
                    .in_constraint_batch_with_causes(
                        &eph,
                        &ras,
                        &decs,
                        None,
                        Some(EvaluationRoll::Sweep(roll)),
                    )
                    .unwrap();
                assert_eq!(actual.violated, expected.violated, "{}", leaf.name());
                assert_eq!(actual.named, expected.named, "{}", leaf.name());
            }
        }
    }

    #[test]
    fn nested_independent_children_prepare_only_the_chunk_grid() {
        let eph = TestEphemeris::new(13);
        let ras = vec![20.0; 10_084];
        let decs = vec![0.0; ras.len()];
        let leaf = RecordingLeaf {
            cells: AtomicUsize::new(0),
            dependent: false,
        };
        let mut result = Array2::from_elem((ras.len(), 13), true);
        sweep_chunks(&mut result, true, 12, |cells| {
            prepare(&leaf, &eph, &ras, &decs, None, cells)
        })
        .unwrap();
        let work = leaf.cells.load(Ordering::Relaxed);
        assert!(work >= result.len());
        assert!(
            work <= result.len() + 4 * 13,
            "independent work repeated across chunks: {work}"
        );
        assert!(result.iter().all(|&v| v));
    }

    #[test]
    fn compact_cell_mapping_survives_reordering_and_pruning() {
        let eph = TestEphemeris::new(5);
        let ras = [0.0, 1.0, 2.0, 3.0];
        let decs = [0.0; 4];
        let indices = [4, 1, 4];
        let cells = [(1, 0), (1, 2), (2, 1)];
        for dependent in [false, true] {
            let leaf = RecordingLeaf {
                cells: AtomicUsize::new(0),
                dependent,
            };
            let prepared = prepare(&leaf, &eph, &ras, &decs, Some(&indices), &cells).unwrap();
            assert_eq!(
                prepared.evaluate(&cells, 0.0).unwrap(),
                [false, false, true]
            );
            assert_eq!(
                prepared.evaluate(&[(2, 1), (1, 0)], 45.0).unwrap(),
                [true, false]
            );
        }
    }

    #[test]
    fn time_invariant_leaf_runs_once_per_target_per_roll() {
        let cells = [(2, 0), (2, 1), (3, 4), (2, 5), (3, 6)];
        let mut calls = Vec::new();
        let result = evaluate_per_target(&cells, 2, 2, |i| {
            calls.push(i);
            i == 2
        });
        assert_eq!(calls, [2, 3]);
        assert_eq!(result, [true, true, false, true, false]);
        calls.clear();
        let next = evaluate_per_target(&[(3, 4), (3, 6)], 2, 2, |i| {
            calls.push(i);
            true
        });
        assert_eq!(calls, [3]);
        assert_eq!(next, [true, true]);
    }

    #[test]
    fn resolved_cells_are_removed_before_the_next_roll() {
        let calls = RefCell::new(Vec::new());
        let evaluator = PreparedRollEvaluator::new(|cells, roll| {
            calls.borrow_mut().push(cells.to_vec());
            Ok(cells.iter().map(|&(i, _)| roll < i as f64 * 90.0).collect())
        });
        let mut result = Array2::from_elem((4, 2), true);
        // (3, 0) was independently blocked; it must never enter the sweep.
        sweep(&evaluator, vec![(0, 1), (1, 0), (2, 1)], &mut result, 4).unwrap();
        assert_eq!(
            *calls.borrow(),
            vec![
                vec![(0, 1), (1, 0), (2, 1)],
                vec![(1, 0), (2, 1)],
                vec![(2, 1)]
            ]
        );
        assert!(!result[[0, 1]] && !result[[1, 0]] && !result[[2, 1]]);
        assert!(result[[3, 0]]);
    }

    #[test]
    fn geometry_is_prepared_in_bounded_chunks() {
        let sizes = RefCell::new(Vec::new());
        let mut result = Array2::from_elem((5042, 13), false);
        sweep_chunks(&mut result, false, 12, |cells| {
            sizes.borrow_mut().push(cells.len());
            Ok(PreparedRollEvaluator::new(|cells, _| {
                Ok(vec![false; cells.len()])
            }))
        })
        .unwrap();
        assert_eq!(*sizes.borrow(), vec![MAX_PREPARED_CELLS, 10]);
        assert!(result.iter().all(|&v| !v));
    }
}
