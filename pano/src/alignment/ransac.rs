/* RANSAC that runs in parallel */

use rayon::prelude::*;
use crate::utils::XORShiftPRNG;

/* Returns unique (if possible) set of random indices */
#[inline]
fn random_indices(seed: usize, up_to: usize, num: usize) -> Vec<usize> {
    let mut rng = XORShiftPRNG::new(seed as u64 ^ 1234567890);
    let mut indices: Vec<_> = (0..num).map(|_| rng.get_usize(up_to)).collect();
    for i in 1..num {
        while indices[0..i].iter().any(|&x| x == indices[i]) {
            indices[i] = rng.get_usize(up_to);
        }
    }
    return indices;
}

/* Runs RANSAC in parallel, uses callbacks */
fn _ransac<F1, F2, M, D>(
    iter_idx_from: usize, iter_idx_to: usize,
    n_data: usize, data_set: &[D],
    fit_model: F1, count_inliers: F2,
) -> Option<(M, usize)>
where
    F1: Fn(&[D]) -> Option<M> + Sync,
    F2: Fn(M) -> usize + Sync,
    D: Sync + Clone, M: Send + Clone
{
    /* Rayon parrallel iterator instead of for loop */
    (iter_idx_from..iter_idx_to).into_par_iter()
        .filter_map(|i| {
            let data_items: Vec<_> = random_indices(i, data_set.len(), n_data).into_iter().map(|i| data_set[i].clone()).collect();
            fit_model(data_items.as_slice()).map(|model| (model.clone(), count_inliers(model)))
        })
        .reduce_with(|best,new| {
            let (best_inl, new_inl) = (best.1, new.1);
            if new_inl > best_inl { new } else { best }
        })
}

#[inline]
pub fn ransac<F1, F2, F3, M, D>(
    max_iterations: usize, min_iterations: usize,
    prob_threshold: Option<f64>,
    n_data: usize, data_set: &[D],
    fit_model: F1,
    count_inliers: F2,
    exit_early: F3
) -> (Option<M>, usize)
where
    F1: Fn(&[D]) -> Option<M> + Sync,
    F2: Fn(M) -> usize + Sync, /* Model, best result (for early exit) */
    F3: Fn(usize,usize) -> bool + Sync,
    D: Sync + Clone, M: Send + Clone
{
    // let start = std::time::SystemTime::now();

    let (mut best_model, mut best_inliers) = (None, 0);
    let (mut iterations, mut batch_size) = (0, 10.max(min_iterations));

    while iterations < max_iterations && !exit_early(iterations,best_inliers) {
        let result = _ransac(
            iterations, (iterations+batch_size).min(max_iterations),
            n_data, data_set,
            &fit_model, &count_inliers
        );
        if let Some((model, inliers)) = result {
            if inliers > best_inliers { (best_model, best_inliers) = (Some(model), inliers) }
        }

        /* Count iterations completed */
        iterations += batch_size;

        /* Increase batch size each time */
        batch_size = ((batch_size * 15) / 10) + 10;

        /* Early exit if the probability of not having found the correct result is below the threshold */
        let inlier_ratio = best_inliers as f64 / data_set.len() as f64;
        let prob = (1.0 - inlier_ratio.powi(n_data as i32)).powi(iterations as i32);
        if prob < prob_threshold.unwrap_or(1.0e-10) { break; }
    }

    // println!("RANSAC time: {:.1?}ms, {} iterations, {}/{} inliers", std::time::SystemTime::now().duration_since(start).unwrap().as_micros() as f64 / 1000.0, iterations.min(max_iterations), best_inliers, data_set.len());

    (best_model, best_inliers)
}


/* Uses ransac to refine an already existing model, returns previous result if 
 * nothing better (more inliers) is found */
pub fn ransac_refine<F1, F2, M, D>(
    previous_best_model: M, previous_best_inliers: usize,
    max_iterations: usize, min_iterations: usize,
    n_data: usize, data_set: &[D],
    fit_model: F1, count_inliers: F2,
) -> (M, usize)
where
    F1: Fn(&[D]) -> Option<M> + Sync,
    F2: Fn(M) -> usize + Sync,
    D: Sync + Clone,
    M: Send + Clone
{
    let (best_model, best_inliers) = ransac(
        max_iterations, min_iterations, Some(0.0), n_data, data_set, fit_model, count_inliers,
        |_,inliers| inliers >= previous_best_inliers
    );
    match (best_model, best_inliers >= previous_best_inliers) {
        (Some(model), true) => (model, best_inliers),
        (_,_) => (previous_best_model, previous_best_inliers)
    }
}
