//! Evaluation-local geometry for coordinated roll sweeps. Cell indices always
//! refer to the original target/time grid, even as resolved cells are removed.
use super::core::{ConstraintEvaluator, EvaluationRoll};
use crate::ephemeris::ephemeris_common::EphemerisBase;
use ndarray::Array2;
use pyo3::PyResult;
use std::collections::HashMap;
use std::ops::{Index, IndexMut};

pub type RollCell = (usize, usize);
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
    if !evaluator.is_roll_dependent() {
        let mask = evaluator.in_constraint_batch(ephemeris, ras, decs, time_indices)?;
        return Ok(PreparedRollEvaluator::new(move |cells, _| {
            Ok(cells.iter().map(|&(i, j)| mask[[i, j]]).collect())
        }));
    }
    if let Some(prepared) =
        evaluator.prepare_roll_sweep(ephemeris, ras, decs, time_indices, cells)?
    {
        return Ok(prepared);
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
    let row_map: HashMap<_, _> = rows
        .iter()
        .enumerate()
        .map(|(local, &i)| (i, local))
        .collect();
    let column_map: HashMap<_, _> = columns
        .iter()
        .enumerate()
        .map(|(local, &j)| (j, local))
        .collect();
    let subset_ras: Vec<_> = rows.iter().map(|&i| ras[i]).collect();
    let subset_decs: Vec<_> = rows.iter().map(|&i| decs[i]).collect();
    let indices: Vec<_> = columns
        .iter()
        .map(|&j| time_indices.map_or(j, |idx| idx[j]))
        .collect();
    Ok(PreparedRollEvaluator::new(move |cells, roll| {
        let mask = evaluator.in_constraint_batch_at_roll(
            ephemeris,
            &subset_ras,
            &subset_decs,
            Some(&indices),
            EvaluationRoll::Sweep(roll),
        )?;
        Ok(cells
            .iter()
            .map(|&(i, j)| mask[[row_map[&i], column_map[&j]]])
            .collect())
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
    use std::cell::RefCell;

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
