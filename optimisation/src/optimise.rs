use maths::{
    traits::Float,
    dual_numbers::MultiDual,
    linear_algebra::{invert_mat_vec},
};
use super::traits::CalculateResiduals;
use super::parameters::{Parameter, GeneralParameter, Block, Parametric};
use super::functor::Functor;
use rayon::prelude::*;

/* Implementation of automatic differentiation of residuals */
pub trait OptimiseAutodiff<T, Input, const N_RESIDUALS: usize>
where
    Self: Functor<Parameter<T>> + Clone, Input: Functor<T> + Copy, T: Float,
{
    #[inline]
    fn calc_block<const N_GRADIENTS: usize>(self, inputs: &[Input]) -> Option<Block<T>>
    where
        Self::Wrapped<MultiDual<T,N_GRADIENTS>>:
            CalculateResiduals<MultiDual<T,N_GRADIENTS>, N_RESIDUALS, Input=Input::Wrapped<MultiDual<T,N_GRADIENTS>>>,
    {
        let mut block = Block{param_ids: vec![], residuals: vec![], gradients: vec![]};
        block.param_ids = self.find_unique_unlocked_parameters();
        self.clone().update_block::<N_GRADIENTS>(inputs, &mut block).ok()?;
        Some(block)
    }

    #[inline]
    fn update_block<const N_GRADIENTS: usize>(self, inputs: &[Input], block: &mut Block<T>) -> Result<(),()>
    where
        Self::Wrapped<MultiDual<T,N_GRADIENTS>>:
            CalculateResiduals<MultiDual<T,N_GRADIENTS>, N_RESIDUALS, Input=Input::Wrapped<MultiDual<T,N_GRADIENTS>>>
    {
        block.set_num_rows(inputs.len() * N_RESIDUALS);

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
        inputs: &[Input], learning_rate: T,
        n_iterations: usize
    ) -> Option<Self>
    where
        Self::Wrapped<MultiDual<T,N_GRADIENTS>>:
            CalculateResiduals<MultiDual<T,N_GRADIENTS>, N_RESIDUALS, Input=Input::Wrapped<MultiDual<T,N_GRADIENTS>>>,
        Input: Sync, T: Send, Self: Sync,
    {
        for _i in 0..n_iterations {
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

        Some(self)
    }
}

impl<M, Input, T, const N: usize> OptimiseAutodiff<T,Input,N> for M
where
    M: Functor<Parameter<T>> + Clone, Input: Functor<T> + Copy, T: Float,
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




pub trait Optimise<T, ParT, WrappedPar, Input, const N_RESIDUALS: usize>
where
    Self: Functor<ParT, Wrapped<Parameter<T>>=WrappedPar> + Clone,
    WrappedPar: Functor<Parameter<T>> + Clone + Send + Sync,
    ParT: GeneralParameter<T>,
    Input: Functor<T> + Copy + Sync,
    T: Float + Send + Sync,
{
    /* Refines the model using autodiff and gauss-newton non linear least squares */
    #[inline]
    fn refine(self, inputs: &[Input], learning_rate: T, n_iterations: usize) -> Option<Self>
    where
        WrappedPar::Wrapped<MultiDual<T,2>>: CalculateResiduals<MultiDual<T,2>, N_RESIDUALS, Input=Input::Wrapped<MultiDual<T,2>>>,
        WrappedPar::Wrapped<MultiDual<T,4>>: CalculateResiduals<MultiDual<T,4>, N_RESIDUALS, Input=Input::Wrapped<MultiDual<T,4>>>,
        WrappedPar::Wrapped<MultiDual<T,8>>: CalculateResiduals<MultiDual<T,8>, N_RESIDUALS, Input=Input::Wrapped<MultiDual<T,8>>>,
        WrappedPar::Wrapped<MultiDual<T,12>>: CalculateResiduals<MultiDual<T,12>, N_RESIDUALS, Input=Input::Wrapped<MultiDual<T,12>>>,
        WrappedPar::Wrapped<MultiDual<T,16>>: CalculateResiduals<MultiDual<T,16>, N_RESIDUALS, Input=Input::Wrapped<MultiDual<T,16>>>,
    {
        let as_pars = self.clone().fmap(|x| x.dereference());
        let par_ids = as_pars.find_unique_unlocked_parameters();

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

    /* Nparams should be larger than or equal to the number of parameters in the structure.
     * This function won't allocate, but everything must be fixed at compile time. */
    // #[inline]
    // fn refine_static<const N_DATA: usize, const N_PARAMS: usize>(
    //     self, inputs: &[Input; N_DATA], learning_rate: T, n_iterations: usize
    // ) -> Option<Self> {
    //     // let par_ids = [0; NPARAMS];
    //     // let
    //     let residuals: [_; N_DATA] = core::array::from_fn(|i| {
    //         s
    //     });
    //     todo!()
    // }
}

impl<M, ParT, WrappedPar, Input, T, const N: usize> Optimise<T,ParT,WrappedPar,Input,N> for M
where
    Self: Functor<ParT, Wrapped<Parameter<T>>=WrappedPar> + Clone,
    WrappedPar: Functor<Parameter<T>> + Clone + Send + Sync,
    ParT: GeneralParameter<T>,
    Input: Functor<T> + Copy + Sync,
    T: Float + Send + Sync,
{}


#[allow(non_snake_case)]
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

    /* Multiply  */
    Some(JT_J_inv.chunks_exact(n_params).map(|row| row.iter().zip(JT_r.iter()).map(|(&a,&b)|a*b).sum()).collect())
}
