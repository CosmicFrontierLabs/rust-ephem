//! Dispatch CPU work to the shared Rayon pool (sized by RAYON_NUM_THREADS).
//! Kernels choose their own crossover because work per item varies widely.

pub(crate) fn should_parallelize(items: usize, minimum: usize) -> bool {
    items >= minimum && rayon::current_num_threads() > 1
}

/// Explicit CPU budget for the optional Parquet readers. With no positive
/// limit each backend keeps its own default, just as Rayon does.
pub(crate) fn configured_thread_limit() -> Option<usize> {
    std::env::var("RAYON_NUM_THREADS")
        .ok()?
        .parse::<usize>()
        .ok()
        .filter(|&threads| threads > 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_requires_enough_work_and_multiple_workers() {
        for workers in [1, 2] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .unwrap();
            pool.install(|| {
                assert!(!should_parallelize(7, 8));
                assert_eq!(should_parallelize(8, 8), workers > 1);
            });
        }
    }
}
