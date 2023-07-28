use super::linear_algebra::Vector;
use super::traits::*;
use core::ops::{Mul, Div, Add, Sub, Neg, Rem, AddAssign, SubAssign, MulAssign, DivAssign};
use std::fmt::Debug;
use core::cmp::Ordering;
use core::iter::{Sum};

/* Generic dual type */
#[derive(Clone,Copy,Debug,Default)]
pub struct Dual<T,U> { pub x: T, pub dx: U }
#[inline(always)] pub const fn Dual<T,U>(x: T, dx: U) -> Dual<T,U> { Dual{x, dx} }

/* Same thing as Ceres Jet (probably) */
pub type MultiDual<T, const N: usize> = Dual<T,Vector<T,N>>;
impl<T, const N: usize> MultiDual<T,N> {
    #[inline]
    pub fn new(x: T, dx: Option<usize>) -> Self
      where T: Float {
        let mut v = Vector::zero();
        if let Some(i) = dx {
            if let Some(vi) = v.0.get_mut(i) {
                *vi = T::one();
            }
        }
        Dual(x, v)
    }
}

/*************************** Trait implementations **************************/

impl <T: PartialEq, U> PartialEq for Dual<T, U> {
    #[inline] fn eq(&self, other: &Self) -> bool { self.x.eq(&other.x) }
}
impl <T: PartialOrd, U> PartialOrd for Dual<T,U> {
    #[inline] fn partial_cmp(&self, other: &Self) -> Option<Ordering> { self.x.partial_cmp(&other.x) }
}
impl <T: Zero, U: Zero> Zero for Dual<T,U> {
    #[inline] fn zero() -> Self { Self {x: T::zero(), dx: U::zero()} }
    #[inline] fn is_zero(&self) -> bool { self.x.is_zero() }
}
impl <T: One, U: Zero> One for Dual<T,U> {
    #[inline] fn one() -> Self { Self {x: T::one(), dx: U::zero()} }
    #[inline] fn is_one(&self) -> bool { self.x.is_one() }
}
impl<T, U: Zero> From<T> for Dual<T,U> {
    #[inline] fn from(value: T) -> Self { Dual{ x: value, dx: U::zero() } }
}

/************************* Basic numeric operations *************************/

impl<T: Copy + Mul<Output=T>, U: Add<Output=U> + Mul<T,Output=U>> Mul<Self> for Dual<T,U> {
    type Output = Self;
    #[inline] fn mul(self, rhs: Self) -> Self { Dual { x: self.x*rhs.x, dx: self.dx*rhs.x + rhs.dx*self.x } }
}
impl<T: Copy + Div<Output=T> + Mul<Output=T>, U: Sub<Output=U> + Mul<T,Output=U> + Div<T,Output=U>> Div<Self> for Dual<T,U> {
    type Output = Self;
    #[inline] fn div(self, rhs: Self) -> Self { Dual { x: self.x/rhs.x, dx: (self.dx*rhs.x - rhs.dx*self.x) / (rhs.x*rhs.x) } }
}
impl<T: Add<Output=T>, U: Add<Output=U>> Add<Self> for Dual<T,U> {
    type Output = Self;
    #[inline] fn add(self, with: Self) -> Self { Dual { x: self.x + with.x, dx: self.dx + with.dx } }
}
impl<T: Sub<Output=T>, U: Sub<Output=U>> Sub<Self> for Dual<T,U> {
    type Output = Self;
    #[inline] fn sub(self, with: Self) -> Self { Dual { x: self.x - with.x, dx: self.dx - with.dx } }
}
impl <T: Neg<Output=Tout>, U: Neg<Output=Uout>, Tout,Uout> Neg for Dual<T,U> {
    type Output = Dual<Tout, Uout>;
    #[inline] fn neg(self) -> Self::Output { Dual { x: -self.x, dx: -self.dx } }
}

/* Assign ops */
impl<T: Copy + Add<T,Output=T>, U: Copy + Add<U,Output=U>> AddAssign for Dual<T,U> {
    #[inline] fn add_assign(&mut self, with: Self) { *self = (*self) + with }
}
impl<T: Copy + Sub<T,Output=T>, U: Copy + Sub<U,Output=U>> SubAssign for Dual<T,U> {
    #[inline] fn sub_assign(&mut self, with: Self) { *self = (*self) - with }
}
impl<T: Copy + Mul<T,Output=T>, U: Copy + Mul<T,Output=U> + Add<U,Output=U>> MulAssign for Dual<T,U> {
    #[inline] fn mul_assign(&mut self, with: Self) { *self = (*self) * with; }
}
impl<T: Copy + Div<T,Output=T> + Mul<T,Output=T>, U: Copy + Mul<T,Output=U> + Div<T,Output=U> + Sub<U,Output=U>> DivAssign for Dual<T,U> {
    #[inline] fn div_assign(&mut self, with: Self) { *self = (*self) / with; }
}

impl <T: Float, U: Sub<Output=U> + Mul<T,Output=U>> Rem for Dual<T,U> {
    type Output = Self;
    #[inline]
    fn rem(self, rhs: Self) -> Self {
        /* Equivalent to: self - (self / rhs).trunc() * rhs
         * from: https://doc.rust-lang.org/stable/core/primitive.f64.html#impl-Rem%3Cf64%3E-for-f64 */
        /* Main calculation is done on T directly as operating on the whole dual njumber is slower
         * (becayse trunc removes all partial derivaives calculated in the division anyway) */
        let div_trunc = (self.x / rhs.x).trunc();
        Dual {
            x: self.x - rhs.x * div_trunc,
            dx: self.dx - rhs.dx * div_trunc
        }
    }
}

impl<T: Add<T,Output=T> + Zero, U: Add<U,Output=U> + Zero> Sum<Self> for Dual<T,U> {
    #[inline] fn sum<I>(iter: I) -> Self where I: Iterator<Item=Self> { iter.fold(Self::zero(), |a,b| a+b) }
}
impl<T: FloatCast, U: Zero> FloatCast for Dual<T,U> {
    #[inline] fn frac(t: i64, b: u64) -> Self { Dual(T::frac(t,b), U::zero()) }
    #[inline] fn int(x: i32) -> Self { Dual(T::int(x), U::zero()) }
    #[inline] fn as_i64(self) -> i64 { self.x.as_i64() }
    #[inline] fn as_u64(self) -> u64 { self.x.as_u64() }
    #[inline] fn as_i32(self) -> i32 { self.x.as_i32() }
    #[inline] fn as_u32(self) -> u32 { self.x.as_u32() }
    #[inline] fn as_i16(self) -> i16 { self.x.as_i16() }
    #[inline] fn as_u16(self) -> u16 { self.x.as_u16() }
    #[inline] fn as_i8(self) -> i8 { self.x.as_i8() }
    #[inline] fn as_u8(self) -> u8 { self.x.as_u8() }
}

impl<T,U> Float for Dual<T,U>
where
    T: Float,
    U: Mul<T,Output=U> + Div<T,Output=U> + Add<Output=U> + Sub<Output=U> + Neg<Output=U> + Zero + Copy + Default + Debug
{
    #[inline] fn floor(self) -> Self { Dual { x: self.x.floor(), dx: Zero::zero() } }
    #[inline] fn ceil(self) -> Self { Dual { x: self.x.ceil(), dx: Zero::zero() } }
    #[inline] fn round(self) -> Self { Dual { x: self.x.round(), dx: Zero::zero() } }
    #[inline] fn trunc(self) -> Self { Dual { x: self.x.trunc(), dx: Zero::zero() } }
    #[inline] fn fract(self) -> Self { Dual { x: self.x.fract(), dx: self.dx } }
    #[inline] fn abs(self) -> Self { if self.x < T::zero() { -self } else { self } }
    #[inline] fn signum(self) -> Self { Dual { x: self.x.signum(), dx: Zero::zero() } }
    #[inline]
    fn powf(self, n: Self) -> Self {
        let x = self.x.powf(n.x);
        let dx = self.dx * n.x * self.x.powf(n.x - T::one()) + n.dx * self.x.ln() * x;
        Dual{x, dx}
    }
    #[inline]
    fn sqrt(self) -> Self {
        match self.x.partial_cmp(&T::zero()) {
            Some(Ordering::Greater) => {
                let x = self.x.sqrt();
                Dual{x, dx: self.dx / (x+x)}
            },
            Some(Ordering::Equal) => Dual{x: T::zero(), dx: U::zero()}, /* TODO: return infinity gradient? ?? IDKDK */
            _ => Dual{x: T::zero(), dx: U::zero()}, /* TODO: return Nan here */
        }
    }
    #[inline]
    fn exp(self) -> Self {
        let x = self.x.exp();
        Dual{x, dx: self.dx * x}
    }
    #[inline]
    fn exp2(self) -> Self {
        let x = self.x.exp();
        Dual{x, dx: self.dx * T::int(2).ln() * x}
    }
    #[inline] fn ln(self) -> Self { Dual { x: self.x.ln(), dx: self.dx / self.x} }
    #[inline]
    fn log2(self) -> Self {
        let x = self.x.log2();
        Dual{x, dx: self.dx / (x * T::int(2).ln())}
    }
    #[inline]
    fn log10(self) -> Self {
        let x = self.x.log10();
        Dual{x, dx: self.dx / (x * T::int(10).ln())}
    }
    #[inline]
    fn cbrt(self) -> Self {
        let x = self.x.cbrt();
        if !x.is_zero() {
            Dual{x, dx: self.dx * (T::frac(1,3) / x.powi(2))}
        } else {
            Dual{x, dx: U::zero()}
        }
    }
    #[inline] fn sin(self) -> Self { Dual(self.x.sin(), self.dx * self.x.cos()) }
    #[inline] fn cos(self) -> Self { Dual(self.x.cos(), self.dx * (-self.x.sin())) }
    #[inline] fn atan2(self, _other: Self) -> Self { todo!() }
    #[inline] fn tan(self) -> Self { self.sin() / self.cos() }
    #[inline] fn asin(self) -> Self { Dual(self.x.asin(), self.dx/(T::one()-self.x.powi(2)).sqrt()) }
    #[inline] fn acos(self) -> Self { Dual(self.x.acos(), self.dx/(-(T::one()-self.x.powi(2)).sqrt())) }
    #[inline] fn atan(self) -> Self { Dual(self.x.atan(), self.dx / (self.x.powi(2) + T::one())) }
    #[inline] fn powi(self, n: i32) -> Self { Dual(self.x.powi(n), self.dx * (self.x.powi(n-1)) * T::int(n)) }
    #[inline] fn is_nan(self) -> bool { self.x.is_nan() }
    #[inline] fn classify(self) -> core::num::FpCategory { self.x.classify() }
    #[inline] fn div_euclid(self, rhs: Self) -> Self { Dual(self.x.div_euclid(rhs.x), U::zero()) }
    #[inline]
    fn rem_euclid(self, rhs: Self) -> Self {
        let div_euclid = self.x.div_euclid(rhs.x);
        Dual {
            x: self.x - rhs.x * div_euclid,
            dx: self.dx - rhs.dx * div_euclid
        }
    }
}