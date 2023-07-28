use maths::{
    traits::Float,
    dual_numbers::MultiDual,
    linear_algebra::{invert_mat_vec, MatrixNxN, Vector},
};
use super::traits::CalculateResiduals;
use super::parameters::{Parameter, GeneralParameter, Block, Parametric};
use super::functor::Functor;
use rayon::prelude::*;

/* Implementation of automatic differentiation of residuals */
pub trait OptimiseAutodiff<T, INPUT, const N_RESIDUALS: usize>
where
    Self: Functor<Parameter<T>> + Clone, INPUT: Functor<T> + Copy, T: Float,
{
    #[inline]
    fn calc_block<const N_GRADIENTS: usize>(self, inputs: &[INPUT]) -> Option<Block<T>>
    where
        Self::Wrapped<MultiDual<T,N_GRADIENTS>>:
            CalculateResiduals<MultiDual<T,N_GRADIENTS>, N_RESIDUALS, Input=INPUT::Wrapped<MultiDual<T,N_GRADIENTS>>>,
    {
        let mut block = Block{param_ids: vec![], residuals: vec![], gradients: vec![]};
        block.param_ids = self.find_unique_unlocked_parameters();
        self.clone().update_block::<N_GRADIENTS>(inputs, &mut block).ok()?;
        Some(block)
    }

    #[inline]
    fn update_block<const N_GRADIENTS: usize>(self, inputs: &[INPUT], block: &mut Block<T>) -> Result<(),()>
    where
        Self::Wrapped<MultiDual<T,N_GRADIENTS>>:
            CalculateResiduals<MultiDual<T,N_GRADIENTS>, N_RESIDUALS, Input=INPUT::Wrapped<MultiDual<T,N_GRADIENTS>>>
    {
        block.set_num_rows(inputs.len() * N_RESIDUALS);

        /* Debugging */
        if block.param_ids.len() > N_GRADIENTS { panic!("Using extra chunks arggh!!!!") }

        let blocks: Vec<_> = block.param_ids.chunks(N_GRADIENTS).map(|block| {
            (self.clone().fmap(|par| {
                if par.locked { MultiDual::<T,N_GRADIENTS>::new(par.value, None) }
                else { MultiDual::new(par.value, block.binary_search(&par.id).ok()) }
            }), block.len())
        }).collect();

        /* Calculate all residuals and their gradients in blocks of N_GRADIENTS at a time */
        let mut grad_index = 0;
        for (as_dual, block_size) in blocks {
            let ctx = as_dual.prepare();
            let out_rows = block.iter_rows_mut().map(|(r,gradients)| (r, gradients[grad_index..].split_at_mut(block_size).0));
            let results = inputs.iter().flat_map(|input| as_dual.run(&ctx, input.fmap(|x|x.into())).into_iter()).map(|r| (r.x, r.dx.0));
            for ((out_x, out_dx), (x, dx)) in out_rows.zip(results) {
                *out_x = x;
                out_dx.copy_from_slice(&dx[..block_size]);
            }
            grad_index += block_size;
        }
        Ok(())
    }

    /* Refines the model using autodiff and gauss-newton non linear least squares */
    #[inline]
    fn _refine<const N_GRADIENTS: usize>(
        mut self, param_ids: &[u64],
        inputs: &[INPUT], learning_rate: T,
        n_iterations: usize
    ) -> Option<Self>
    where
        Self::Wrapped<MultiDual<T,N_GRADIENTS>>:
            CalculateResiduals<MultiDual<T,N_GRADIENTS>, N_RESIDUALS, Input=INPUT::Wrapped<MultiDual<T,N_GRADIENTS>>>,
        INPUT: Sync, T: Send, Self: Sync,
    {
        for i in 0..n_iterations {
            /* Non parallel version */
            // let block = self.clone().calc_block(inputs)?;
            // let step = sparse_step(param_ids, std::slice::from_ref(&block))?;
            /* Parallel version */
            let blocks: Vec<_> = inputs.par_chunks(100).filter_map(|chunk| self.clone().calc_block::<N_GRADIENTS>(chunk)).collect();
            let step = sparse_step(param_ids, &blocks)?;
            /* Return early if a NaN appeared, this makes RANSAC exit faster */
            if step.iter().any(|x| x.is_nan()) { return None; }
            // else {println!("{:?}", step)}
            /* Apply step */
            self = self.apply_step(&step, param_ids, learning_rate);
        }

        // /* Version with less allocations */
        // let mut residuals = Vec::with_capacity(inputs.len() * N_RESIDUALS);
        // for i in 0..n_iterations
        // {
        //     let as_dual = self.clone().fmap(|par| {
        //         if par.locked { MultiDual::<T,N_GRADIENTS>::new(par.value, None) }
        //         else { MultiDual::new(par.value, param_ids.binary_search(&par.id).ok()) }
        //     });

        //     let ctx = as_dual.prepare();
        //     // residuals.par_extend(inputs.par_iter().flat_map(|input| as_dual.run(&ctx, input.fmap(|x|x.into()))));
        //     residuals.extend(inputs.iter().flat_map(|input| as_dual.run(&ctx, input.fmap(|x|x.into()))));

        //     /* Do step */
        //     let mut JT_J = MatrixNxN::<T,N_GRADIENTS>::from_fn(|r,c| residuals.iter().map(|residual| residual.dx[r] * residual.dx[c]).sum());
        //     let mut JT_r = Vector::from_fn(|r| residuals.iter().map(|residual| residual.dx[r] * residual.x).sum::<T>());

        //     /* Put a one in zero locations in the matrix diagonally so it can invert */
        //     for i in 0..N_GRADIENTS {
        //         if JT_r[i].is_zero() {
        //             JT_J[i][i] = T::one();
        //         }
        //     }

        //     /* TODO: look at sum of squared errors and do some levenberg marquat thing */
        //     if let Some(JT_J_inv) = JT_J.invert() {
        //         let step = JT_J_inv * JT_r;
        //         if step.0.iter().any(|x| x.is_nan()) { return None; }
        //         self = self.apply_step(&step.0, param_ids, learning_rate);
        //     } else {
        //         return None;
        //     }

        //     residuals.clear();
        // }

        Some(self)
    }
}

impl<M, INPUT, T, const N: usize> OptimiseAutodiff<T,INPUT,N> for M
where
    M: Functor<Parameter<T>> + Clone, INPUT: Functor<T> + Copy, T: Float,
{}

pub trait ApplyStep<T>: Functor<Parameter<T>> {
    #[inline]
    fn apply_step(self, step: &[T], param_ids: &[u64], learning_rate: T) -> Self
      where T: std::ops::Mul<T,Output=T> + std::ops::Sub<T,Output=T> + Copy {
        self.tmap(|mut par| {
            if !par.locked {
                if let Ok(index) = param_ids.binary_search(&par.id) {
                    par.value = par.value - step[index] * learning_rate;
                }
            } par
        })
    }
}

impl<M,T> ApplyStep<T> for M where M: Functor<Parameter<T>> {}




pub trait Optimise<T, PAR_T, WRAPPED_PAR, INPUT, const NRESIDUALS: usize>
where
    Self: Functor<PAR_T, Wrapped<Parameter<T>>=WRAPPED_PAR> + Clone,
    WRAPPED_PAR: Functor<Parameter<T>> + Clone + Send + Sync,
    PAR_T: GeneralParameter<T>,
    INPUT: Functor<T> + Copy + Sync,
    T: Float + Send + Sync,
{
    /* Refines the model using autodiff and gauss-newton non linear least squares */
    #[inline]
    fn refine(self, inputs: &[INPUT], learning_rate: T, n_iterations: usize) -> Option<Self>
    where
        WRAPPED_PAR::Wrapped<MultiDual<T,2>>: CalculateResiduals<MultiDual<T,2>, NRESIDUALS, Input=INPUT::Wrapped<MultiDual<T,2>>>,
        WRAPPED_PAR::Wrapped<MultiDual<T,4>>: CalculateResiduals<MultiDual<T,4>, NRESIDUALS, Input=INPUT::Wrapped<MultiDual<T,4>>>,
        WRAPPED_PAR::Wrapped<MultiDual<T,8>>: CalculateResiduals<MultiDual<T,8>, NRESIDUALS, Input=INPUT::Wrapped<MultiDual<T,8>>>,
        WRAPPED_PAR::Wrapped<MultiDual<T,12>>: CalculateResiduals<MultiDual<T,12>, NRESIDUALS, Input=INPUT::Wrapped<MultiDual<T,12>>>,
        WRAPPED_PAR::Wrapped<MultiDual<T,16>>: CalculateResiduals<MultiDual<T,16>, NRESIDUALS, Input=INPUT::Wrapped<MultiDual<T,16>>>,
        /* Debugging */
        // <WRAPPED_PAR as Functor<Parameter<T>>>::Wrapped<Option<usize>>: std::fmt::Debug, WRAPPED_PAR: std::fmt::Debug,
    {
        let as_pars = self.clone().fmap(|x| x.dereference());
        let par_ids = as_pars.find_unique_unlocked_parameters();

        // /* Collect locked and unlocked parameters */
        // let (mut locked, mut unlocked) = (vec![], vec![]);
        // let as_par_indices = as_pars.clone().fmap(|par| par_ids.binary_search(&par.id).ok());
        // as_pars.clone().fmap(|par| match par.locked {
        //     true => locked.push(par),
        //     false => unlocked.push(par),
        // });

        // /* Debugging */
        // println!("locked = {:.3?}", locked.iter().map(|x| x.id).collect::<Vec<_>>());
        // println!("unlocked = {:.3?}", unlocked.iter().map(|x| x.id).collect::<Vec<_>>());
        /* Debugging */
        // if par_ids.len() > 7 {
        //     println!("as_pars = {:#.3?}", as_pars);
        //     let as_par_indices = as_pars.fmap(|par| par_ids.binary_search(&par.id).ok());
        //     println!("as_par_indices = {:#.3?}", as_par_indices);
        //     panic!("poo");
        // }

        /* Do refine */
        let result = if par_ids.len() <= 2 { as_pars._refine::<2>(&par_ids, inputs, learning_rate, n_iterations)? }
                else if par_ids.len() <= 4 { as_pars._refine::<4>(&par_ids, inputs, learning_rate, n_iterations)? }
                else if par_ids.len() <= 8 { as_pars._refine::<8>(&par_ids, inputs, learning_rate, n_iterations)? }
                else if par_ids.len() <= 12 { as_pars._refine::<12>(&par_ids, inputs, learning_rate, n_iterations)? }
                else { as_pars._refine::<16>(&par_ids, inputs, learning_rate, n_iterations)? };

        /* Return with new parameter values... */
        let mut new_par_values = vec![];
        result.fmap(|par| if !par.locked { new_par_values.push(par.value); } );
        let mut param_iter = new_par_values.into_iter();
        Some(self.tmap(|mut par| {
            if !par.is_locked() {
                par.set_value(param_iter.next().unwrap());
            } par
        }))
    }

    // /* Nparams should be larger than or equal to the number of parameters in the structure.
    //  * This function won't allocate, but everything must be fixed at compile time. */
    // #[inline]
    // fn refine_static<const NDATA: usize, const NPARAMS: usize>(
    //     self, inputs: &[INPUT; NDATA], learning_rate: T, n_iterations: usize
    // ) -> Option<Self> {
    //     // let par_ids = [0; NPARAMS];
    //     // let
    //     let residuals: [_; NDATA] = core::array::from_fn(|i| {
    //         s
    //     });
    //     todo!()
    // }
}

impl<M, PAR_T, WRAPPED_PAR, INPUT, T, const N: usize> Optimise<T,PAR_T,WRAPPED_PAR,INPUT,N> for M
where
    Self: Functor<PAR_T, Wrapped<Parameter<T>>=WRAPPED_PAR> + Clone,
    WRAPPED_PAR: Functor<Parameter<T>> + Clone + Send + Sync,
    PAR_T: GeneralParameter<T>,
    INPUT: Functor<T> + Copy + Sync,
    T: Float + Send + Sync,
{}


#[inline]
pub fn sparse_step<T: Float>(param_ids: &[u64], blocks: &[Block<T>]) -> Option<Vec<T>>
{
    let n_params = param_ids.len();

    /* JT * residuals */
    let JT_r: Vec<T> = param_ids.iter().map(|&id|
        blocks.iter().filter_map(|block| block.residuals_gradients_mulsum(id)).sum()
    ).collect();

    /* JT * J */
    let mut JT_J: Vec<T> = param_ids.iter().flat_map(|id_row| param_ids.iter().map(|id_col| 
        blocks.iter().filter_map(|block| block.gradient_columns_mulsum(*id_row, *id_col)).sum()
    )).collect();

    // use opencv::{prelude::*, self as cv};
    // use crate::utils::*;
    // if n_params > 80 {
    //     println!("\n\n\n\n\n\n\nJT_J_inv:");
    //     for row in JT_J.chunks(n_params) {
    //         for x in row { print!("{:?}, ", x); }
    //         println!("");
    //     }
    //     let mut sparseness: Vec<_> = param_ids.iter().flat_map(|id_row| param_ids.iter().map(|id_col| {
    //         let sum: T = blocks.iter().filter_map(|block| block.gradient_columns_mulsum(*id_row, *id_col)).sum();
    //         // let sum = blocks.iter().filter_map(|block| block.gradient_columns_mulsum(*id_row, *id_col)).count();
    //         if sum.is_zero() { 0 } else { 255u8 }
    //     })).collect();
    //     let cvi = make_cv_image(n_params, n_params, &mut sparseness);
    //     cv::imgcodecs::imwrite("sparseness.png", &cvi, &cv::core::Vector::default());
    // }

    /* If any parameter has no 'gradient', put a one in its diagonal position of JT_J (the 'hessian'),
     * to make inversion not fail (it will not have any effect on optimisation and only relevant parameters will be optimised) */
    for p in 0..n_params {
        if JT_r[p].is_zero() {
            JT_J[p*n_params+p] = T::one();
            if n_params > 20 { println!("warning: no gradient for parameter #{}", p); }
        }
    }

    /* Calculate step */
    let JT_J_inv = invert_mat_vec(JT_J, n_params)?;

    // radius
    /* Multiply  */
    Some(JT_J_inv.chunks_exact(n_params).map(|row| row.iter().zip(JT_r.iter()).map(|(&a,&b)|a*b).sum()).collect())
}