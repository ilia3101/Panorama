pub mod optimise;
pub mod functor;
pub mod traits;
pub mod parameters;

use maths::traits::Float;
use maths::dual_numbers::MultiDual;
use maths::linear_algebra::{Vector, Matrix, MatrixNxN};


pub fn least_squares<T, F, const N_PARAMS: usize, const N_RESIDUALS: usize>(
    n_iterations: usize,
    learn_rate: T,
    starting_params: [T; N_PARAMS],
    mut calculate_residuals: F
) -> [T; N_PARAMS]
where
    T: Float,
    F: FnMut([MultiDual<T,N_PARAMS>; N_PARAMS]) -> [MultiDual<T,N_PARAMS>; N_RESIDUALS]
{
    let mut params = Vector(starting_params);

    for _i in 0..n_iterations
    {
        let as_dual = Vector::from_fn(|i| MultiDual::new(params[i], Some(i)));
        let result = calculate_residuals(as_dual.as_array());

        let residuals = Vector::<_,N_RESIDUALS>::from_fn(|i| result[i].x);
        let jacobian = Matrix::<_,N_RESIDUALS, N_PARAMS>::from_fn(|i,j| result[i].dx[j]);
        let JT_J = jacobian.transpose() * jacobian;
        let gradient = jacobian.transpose() * residuals;

        /* TODO: look at sum of squared errors and do some levenberg marquat thing */
        if let Some(JT_J_inv) = JT_J.invert() {
            params -= JT_J_inv * gradient;
        } else {
            println!("JT_J could not be inverted");
            break;
        }
    }

    return params.as_array();
}



/* TODO: improve this */
// pub fn least_squares_function_fit<T, F, const N_PARAMS: usize, const N_IN: usize, const N_OUT: usize>(
//     n_iterations: usize,
//     learn_rate: T,
//     starting_params: [T; N_PARAMS],
//     inputs: &[[T; N_IN]],
//     outputs: &[[T; N_OUT]],
//     mut function: F
// ) -> [T; N_PARAMS]
// where
//     T: Float,
//     F: FnMut([MultiDual<T,N_PARAMS>; N_PARAMS], [MultiDual<T,N_PARAMS>; N_IN]) -> [MultiDual<T,N_PARAMS>; N_OUT]
// {
//     assert!(inputs.len() == outputs.len(), "Inputs and outputs must be the same length");
//     let mut params = Vector(starting_params);
//     let mut residuals: Vec<MultiDual<T,N_PARAMS>> = Vec::with_capacity(inputs.len() * outputs.len());

//     for _i in 0..n_iterations
//     {
//         let as_dual = core::array::from_fn(|i| MultiDual::new(params[i], Some(i)));
//         residuals.extend(inputs.iter().flat_map(|input| function(as_dual, input.map(|x| x.into()))));

//         let JT_J = MatrixNxN::<T,N_PARAMS>::from_fn(|r,c| residuals.iter().map(|residual| residual.dx[r] * residual.dx[c]).sum());
//         let JT_r = Vector::from_fn(|r| residuals.iter().map(|residual| residual.dx[r] * residual.x).sum());

//         if let Some(JT_J_inv) = JT_J.invert() {
//             params -= JT_J_inv * JT_r;
//         } else {
//             println!("JT_J could not be inverted");
//             break;
//         }

//         residuals.clear()
//     }

//     return params.as_array();

//     todo!()
// }