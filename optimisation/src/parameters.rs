use super::functor::Functor;
use maths::traits::{Float, One, Zero, FloatCast};
use std::sync::atomic::{AtomicU64, Ordering::SeqCst};

/***************************** Residuals/jacobian block type ****************************/

/*
 * Residuals and partial derivatives are stored in a Block<T>. Multiple blocks
 * can be used to represent a sparse jacobian matrix for bundle adjustment.
 */
#[derive(Clone, Debug)]
pub struct Block<T> {
    /* Unique IDs of parameters in a sorted order */
    pub param_ids: Vec<u64>,
    /* Residuals of this blocks */
    pub residuals: Vec<T>,
    /* Jacobian matrix stored in flat array,
     * of length: residuals.len() * param_ids.len() */
    pub gradients: Vec<T>,
}

impl<T> Block<T> {
    #[inline] pub fn num_rows(&self) -> usize { self.residuals.len() }
    #[inline] pub fn num_cols(&self) -> usize { self.param_ids.len() }
    #[inline] pub fn param_col(&self, id: u64) -> Option<usize> { self.param_ids.binary_search_by(|x| x.cmp(&id)).ok() }
    #[inline]
    pub fn gradient_columns_mulsum(&self, param1_id: u64, param2_id: u64) -> Option<T> where T: Float {
        let (c1, c2) = (self.param_col(param1_id)?, self.param_col(param2_id)?);
        Some(self.gradients.chunks_exact(self.num_cols()).map(|row| row[c1] * row[c2]).sum())
    }
    #[inline]
    pub fn residuals_gradients_mulsum(&self, param_id: u64) -> Option<T> where T: Float {
        let (cols, rows, col) = (self.num_cols(), self.num_rows(), self.param_col(param_id)?);
        Some((0..rows).map(|row| self.residuals[row] * self.gradients[row*cols + col]).sum())
    }
    #[inline]
    pub fn get_gradients_row_mut(&mut self, row: usize) -> &mut [T] {
        let cols = self.num_cols();
        &mut self.gradients[row*cols..(row+1)*cols]
    }
    #[inline]
    pub fn get_residuals_mut(&mut self) -> &mut [T] {
        self.residuals.as_mut_slice()
    }
    #[inline]
    pub fn set_num_rows(&mut self, num_rows: usize)
      where T: Default + Clone {
        let num_cols = self.num_cols();
        self.residuals.resize(num_rows, T::default());
        self.gradients.resize(num_rows * num_cols, T::default());
    }
    /* A refernce to the residual value and the gradient row as a mutable slice */
    #[inline]
    pub fn iter_rows_mut(&mut self) -> std::iter::Zip<std::slice::IterMut<T>, std::slice::ChunksExactMut<T>> {
        let num_cols = self.num_cols();
        self.residuals.iter_mut().zip(self.gradients.chunks_exact_mut(num_cols))
    }
}

/*********************************** Parameter type *************************************/

/*
 * This wraps a parameter value with with a unique ID and a lock flag.
 * The unique ID can be used to indicate to the optimiser that two
 * parameter instances are actually the same parameter, and should be
 * optimised as one (in which case their values should be the same).
 */
#[derive(Clone,Copy,Debug)]
pub struct Parameter<T> {
    pub id: u64,
    pub locked: bool,
    pub value: T,
}

/* All parameters have a unique ID (this is a atomic global variable yeah) */
static _param_id: AtomicU64 = AtomicU64::new(0);

impl<T> Parameter<T> {
    /* Constructors */
    #[inline] pub fn new(value: T, locked: bool) -> Self { Self { id: _param_id.fetch_add(1, SeqCst), locked, value } }
    #[inline] pub fn locked(value: T) -> Self { Self::new(value, true) }
    #[inline] pub fn unlocked(value: T) -> Self { Self::new(value, false) }

    /* Lock or unlock a parameter, returning a modified copy */
    #[inline] pub fn to_locked(self) -> Self { Self { locked: true, ..self } }
    #[inline] pub fn to_unlocked(self) -> Self { Self { locked: false, ..self } }

    /* Lock/unlock parameter by mutable reference, because "model.parameter.lock()"
     * looks nicer than "model.parameter = model.parameter.locked()" */
    #[inline] pub fn lock(&mut self) { self.locked = true }
    #[inline] pub fn unlock(&mut self) { self.locked = false }
}

impl<T: One> One for Parameter<T> {
    #[inline] fn one() -> Self { Self::locked(T::one()) }
}
impl<T: Zero> Zero for Parameter<T> {
    #[inline] fn zero() -> Self { Self::locked(T::zero()) }
}
impl<T: FloatCast> FloatCast for Parameter<T> {
    #[inline] fn frac(top: i64, bottom: u64) -> Self { Self::locked(T::frac(top, bottom)) }
    #[inline] fn int(x: i32) -> Self { Self::locked(T::int(x)) }
    #[inline] fn as_i64(self) -> i64 { self.value.as_i64() }
    #[inline] fn as_u64(self) -> u64 { self.value.as_u64() }
    #[inline] fn as_i32(self) -> i32 { self.value.as_i32() }
    #[inline] fn as_u32(self) -> u32 { self.value.as_u32() }
    #[inline] fn as_i16(self) -> i16 { self.value.as_i16() }
    #[inline] fn as_u16(self) -> u16 { self.value.as_u16() }
    #[inline] fn as_i8(self) -> i8 { self.value.as_i8() }
    #[inline] fn as_u8(self) -> u8 { self.value.as_u8() }
}
impl<T: Default> Default for Parameter<T> {
    #[inline] fn default() -> Self { Self::locked(T::default()) }
}

/* This trait could be used to implement a smart parameter type which would allow for
 * linking of parameters without having to synchronise them manually */
pub trait GeneralParameter<T> {
    fn get_id(&self) -> Option<u64>;
    fn is_locked(&self) -> bool;
    fn get_value(&self) -> T;
    fn set_value(&mut self, value: T);
    fn set_locked(&mut self, locked: bool);
    #[inline]
    fn dereference(&self) -> Parameter<T> {
        match self.get_id() {
            Some(id) => Parameter{id, locked: self.is_locked(), value: self.get_value()},
            None => Parameter::new(self.get_value(), self.is_locked()),
        }
    }
}

impl<T: Float> GeneralParameter<T> for T {
    #[inline] fn get_id(&self) -> Option<u64> { None }
    #[inline] fn is_locked(&self) -> bool { false }
    #[inline] fn get_value(&self) -> T { *self }
    #[inline] fn set_value(&mut self, value: T) { std::mem::replace(self, value); }
    #[inline] fn set_locked(&mut self, _locked: bool) { unimplemented!("Can't lock a T") }
}

impl<T: Clone> GeneralParameter<T> for Parameter<T> {
    #[inline] fn get_id(&self) -> Option<u64> { Some(self.id) }
    #[inline] fn is_locked(&self) -> bool { self.locked }
    #[inline] fn get_value(&self) -> T { self.value.clone() }
    #[inline] fn set_value(&mut self, value: T) { self.value = value; }
    #[inline] fn set_locked(&mut self, locked: bool) { self.locked = locked; }
}

/* Convert a Model<T> into Model<Parameter<T>> */
pub trait ToParameters<T: Float>: Functor<T> {
    #[inline] fn to_pars(self) -> Self::Wrapped<Parameter<T>> { self.fmap(|v| Parameter::locked(v)) }
}
impl<M, T: Float> ToParameters<T> for M where M: Functor<T> {}

/* Conver back to T */
pub trait FromParameters<T: Float>: Functor<Parameter<T>> {
    #[inline] fn from_pars(self) -> Self::Wrapped<T> { self.fmap(|p| p.value) }
}
impl<M, T: Float> FromParameters<T> for M where M: Functor<Parameter<T>> {}

/************************************* Parametric trait *************************************/

/*
 * This trait automatically implements locking and unlocking methods
 * for whole structs of Parameter<T>s, as an example, if camera.distortion is a
 * struct with multiple fields, all of them can be locked at once:
 * 
 *      camera.distortion.lock()
 * 
 * Compared to locking each individual parameter:
 * 
 *      camera.distortion.k1.lock()
 *      camera.distortion.k2.lock()
 *         ...
 *      camera.distortion.kN.lock()
 * 
 * This also allows generic functions which know nothing of the specifics
 * of the structure to lock all of its fields.
 *
 */
pub trait Parametric<T,P>
where
    Self: Functor<P> + Clone, P: GeneralParameter<T>
{
    #[inline] fn lock(&mut self) { *self = self.clone().tmap(|mut p| {p.set_locked(true); p}) }
    #[inline] fn unlock(&mut self) { *self = self.clone().tmap(|mut p| {p.set_locked(false); p}) }
    #[inline]
    fn find_unique_unlocked_parameters(&self) -> Vec<u64> {
        // let mut result = vec![];
        // self.clone().fmap(|parameter| {
        //     let parameter = parameter.dereference();
        //     if !parameter.locked {
        //         if let Err(pos) = result.binary_search(&parameter.id) {
        //             result.insert(pos, parameter.id) /* Not already in the list */
        //         }
        //     }
        // }); return result;
        let mut ids = vec![];
        self.clone().fmap(|parameter| {
            let parameter = parameter.dereference();
            if !parameter.locked {
                ids.push(parameter.id);
            }
        });
        ids.sort(); ids.dedup();
        return ids;
    }
}

impl<M,T,P> Parametric<T,P> for M
where
    Self: Functor<P> + Clone, P: GeneralParameter<T>
{}