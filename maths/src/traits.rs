use core::ops::*;
use core::num::FpCategory;

/* This file is inspired by the num-traits crate */

pub trait Zero {
    fn zero() -> Self;
    fn is_zero(&self) -> bool { unimplemented!() }
}

pub trait One {
    fn one() -> Self;
    fn is_one(&self) -> bool { unimplemented!() }
}

pub trait NumOps<Rhs = Self>:
    Add<Rhs, Output=<Self as NumOps<Rhs>>::Output>
  + Sub<Rhs, Output=<Self as NumOps<Rhs>>::Output>
  + Mul<Rhs, Output=<Self as NumOps<Rhs>>::Output>
  + Div<Rhs, Output=<Self as NumOps<Rhs>>::Output>
{
    type Output;
}

impl<T, RHS, OUT> NumOps<RHS> for T where
    T: Add<RHS, Output=OUT>
     + Sub<RHS, Output=OUT>
     + Mul<RHS, Output=OUT>
     + Div<RHS, Output=OUT>
{
    type Output = OUT;
}

pub trait NumAssignOps<Rhs = Self>:
    AddAssign<Rhs> + SubAssign<Rhs> + MulAssign<Rhs> + DivAssign<Rhs>
{}

impl<T, Rhs> NumAssignOps<Rhs> for T where
    T: AddAssign<Rhs> + SubAssign<Rhs> + MulAssign<Rhs> + DivAssign<Rhs>
{}

pub trait FloatCast {
    fn frac(top: i64, bottom: u64) -> Self;
    fn int(x: i32) -> Self;
    fn as_i64(self) -> i64;
    fn as_u64(self) -> u64;
    fn as_i32(self) -> i32;
    fn as_u32(self) -> u32;
    fn as_i16(self) -> i16;
    fn as_u16(self) -> u16;
    fn as_i8(self) -> i8;
    fn as_u8(self) -> u8;
}

pub trait Float:
    NumOps<Output=Self> +
    NumAssignOps<Self> +
    Neg<Output=Self> +
    Rem<Output=Self> +
    PartialOrd +
    PartialEq +
    FloatCast +
    core::fmt::Debug +
    std::iter::Sum<Self> +
    Default +
    Copy +
    One +
    Zero
{
    /* TODO: replicate all funcitionality in https://doc.rust-lang.org/std/primitive.f64.html or not */
    fn floor(self) -> Self;
    fn ceil(self) -> Self;
    fn round(self) -> Self;
    fn trunc(self) -> Self;
    fn fract(self) -> Self;
    fn abs(self) -> Self;
    fn signum(self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn tan(self) -> Self;
    fn asin(self) -> Self;
    fn acos(self) -> Self;
    fn atan(self) -> Self;
    fn atan2(self, other: Self) -> Self;
    fn ln(self) -> Self;
    fn log2(self) -> Self;
    fn log10(self) -> Self;
    fn exp(self) -> Self;
    fn exp2(self) -> Self;
    fn sqrt(self) -> Self;
    fn cbrt(self) -> Self;
    fn powf(self, n: Self) -> Self;
    fn powi(self, n: i32) -> Self;

    /* Classifications. TODO: turn this trait into 'real' and put the float specific functions in a 'float' trait (?????) */
    #[inline] fn is_nan(self) -> bool { self.classify() == FpCategory::Nan }
    #[inline] fn is_infinite(self) -> bool { self.classify() == FpCategory::Infinite }
    #[inline] fn is_finite(self) -> bool { self.classify() != FpCategory::Infinite && self.classify() != FpCategory::Nan }
    #[inline] fn is_subnormal(self) -> bool { self.classify() == FpCategory::Subnormal }
    #[inline] fn is_normal(self) -> bool { self.classify() == FpCategory::Normal }
    fn classify(self) -> FpCategory;

    /* Functions with default implementations */
    #[inline] fn max(self, other: Self) -> Self { if self > other { self } else { other } }
    #[inline] fn min(self, other: Self) -> Self { if self < other { self } else { other } }
    #[inline] fn div_euclid(self, rhs: Self) -> Self { (self / rhs).floor() }
    #[inline] fn rem_euclid(self, rhs: Self) -> Self { self - (self / rhs).floor() * rhs }
}


/************************ Implementations for f32 and f64 **********************/

macro_rules! impl_from_fraction_float {
    ($t: ty) => {
        impl FloatCast for $t {
            #[inline(always)] fn frac(top: i64, bottom: u64) -> Self { top as $t / bottom as $t }
            #[inline(always)] fn int(top: i32) -> Self { top as $t }
            #[inline(always)] fn as_i64(self) -> i64 { self as i64 }
            #[inline(always)] fn as_u64(self) -> u64 { self as u64 }
            #[inline(always)] fn as_i32(self) -> i32 { self as i32 }
            #[inline(always)] fn as_u32(self) -> u32 { self as u32 }
            #[inline(always)] fn as_i16(self) -> i16 { self as i16 }
            #[inline(always)] fn as_u16(self) -> u16 { self as u16 }
            #[inline(always)] fn as_i8(self) -> i8 { self as i8 }
            #[inline(always)] fn as_u8(self) -> u8 { self as u8 }
        }
    };
}

impl_from_fraction_float!(f64);
impl_from_fraction_float!(f32);

macro_rules! impl_one_zero {
    ($t: ty, $zero_literal: expr, $one_literal: expr) => {
        impl Zero for $t {
            #[inline(always)] fn zero() -> Self { $zero_literal }
            #[inline(always)] fn is_zero(&self) -> bool { *self == $zero_literal }
        }
        impl One for $t {
            #[inline(always)] fn one() -> Self { $one_literal }
            #[inline(always)] fn is_one(&self) -> bool { *self == $one_literal }
        }
    };
}

impl_one_zero!(f32, 0.0, 1.0);
impl_one_zero!(f64, 0.0, 1.0);
impl_one_zero!(u8, 0, 1);
impl_one_zero!(i8, 0, 1);
impl_one_zero!(u16, 0, 1);
impl_one_zero!(i16, 0, 1);
impl_one_zero!(u32, 0, 1);
impl_one_zero!(i32, 0, 1);
impl_one_zero!(u64, 0, 1);
impl_one_zero!(i64, 0, 1);

macro_rules! existing_impl {
    ($( fn $fname:ident ( self $( , $arg:ident : $t:ty )* ) -> $ret:ty; )*)
        => {$(
            #[inline(always)]
            fn $fname(self $(, $arg:$t)*) -> $ret {
                Self::$fname(self $(, $arg)*)
            }
    )*};
}

macro_rules! impl_float {
    ($t: ty) => {
        impl Float for $t {
            existing_impl! {
                fn floor(self) -> Self;
                fn ceil(self) -> Self;
                fn round(self) -> Self;
                fn trunc(self) -> Self;
                fn fract(self) -> Self;
                fn abs(self) -> Self;
                fn signum(self) -> Self;
                fn powf(self, n: Self) -> Self;
                // fn mul_add(self, a: Self, b: Self) -> Self;
                fn sqrt(self) -> Self;
                fn exp(self) -> Self;
                fn exp2(self) -> Self;
                fn ln(self) -> Self;
                fn log2(self) -> Self;
                fn log10(self) -> Self;
                fn cbrt(self) -> Self;
                fn sin(self) -> Self;
                fn cos(self) -> Self;
                fn atan2(self, other: Self) -> Self;
                fn tan(self) -> Self;
                fn asin(self) -> Self;
                fn acos(self) -> Self;
                fn atan(self) -> Self;
                fn powi(self, n: i32) -> Self;
                fn is_nan(self) -> bool;
                fn classify(self) -> FpCategory;
                fn div_euclid(self, rhs: Self) -> Self;
                fn rem_euclid(self, rhs: Self) -> Self;
            }
        }
    };
}

impl_float!(f32);
impl_float!(f64);
