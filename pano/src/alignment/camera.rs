use maths::{
    linear_algebra::{Point2D, Point3D, Vector3D},
    traits::{Float,One,Zero},
};
use optimisation::{functor::*,};

/* The camera trait for implementing camera models */
pub trait Camera<T> {
    fn project_to_film(&self, point: Point3D<T>) -> Point2D<T>;
    fn project_from_film(&self, point: Point2D<T>) -> Point3D<T>;
}

/* Pinhole camera */
#[derive(Clone, Copy, Debug)]
pub struct PinholeCamera<T> { pub focal_length: T }

impl<T:Float> Camera<T> for PinholeCamera<T> {
    #[inline] fn project_to_film(&self, p: Point3D<T>) -> Point2D<T> { p.xy() / p.z() * self.focal_length }
    #[inline] fn project_from_film(&self, p: Point2D<T>) -> Point3D<T> { Point3D(p.x(), p.y(), self.focal_length) }
}

impl <T: One> Default for PinholeCamera<T> {
    #[inline] fn default() -> Self { Self { focal_length: T::one() } }
}

impl<A> Functor<A> for PinholeCamera<A> {
    type Wrapped<B> = PinholeCamera<B>;
    #[inline] fn fmap<F: FnMut(A)->B, B>(self, mut f: F) -> PinholeCamera<B> { PinholeCamera { focal_length: f(self.focal_length) } }
}


/* General camera, uses Gennery (2006) generalised lens model,
 * can use any radial distortion model for additional correction.
 * TODO: principal point, tangential distortion maybe */
#[derive(Clone, Copy, Debug)]
pub struct GeneralCamera<T, D> {
    pub focal_length: T,
    pub linearity: T,
    pub radial_distortion: D,
}

/* Gennery (2006) mapping functions */
impl <T:Float, D> GeneralCamera<T,D>
{
    #[inline]
    fn theta_to_radius(&self, t: T) -> T {
        if self.linearity > T::zero() {
            (t * self.linearity).tan() / self.linearity
        } else {
            (t * self.linearity).sin() / self.linearity
        }
    }

    #[inline]
    fn radius_to_theta(&self, r: T) -> T {
        if self.linearity > T::zero() {
            (r * self.linearity).atan() / self.linearity
        } else {
            (r * self.linearity).asin() / self.linearity
        }
    }
}

impl <T: One, D: Default> Default for GeneralCamera<T,D> {
    #[inline]
    fn default() -> Self {
        Self {
            focal_length: T::one(),
            linearity: T::one(), /* L = 1 = rectilinear projection */
            radial_distortion: D::default(),
        }
    }
}

impl<T, D> From<PinholeCamera<T>> for GeneralCamera<T,D>
  where Self: Default {
    #[inline] fn from(pinhole: PinholeCamera<T>) -> Self {
        Self { focal_length: pinhole.focal_length, ..Self::default() }
    }
}

impl<T:Float, D:RadialDistortion<T>> Camera<T> for GeneralCamera<T,D>
{
    #[inline]
    fn project_to_film(&self, point: Point3D<T>) -> Point2D<T> {
        /* Project */
        let point_xy = Point2D(point.x(), point.y());
        let len = point_xy.magnitude();
        if !len.is_zero() {
            let theta = point.z().acos();
            let radius = self.theta_to_radius(theta);
            (point_xy/len) * self.radial_distortion.distort(radius * self.focal_length)
        } else { 
            point_xy * self.focal_length
        }
    }

    #[inline]
    fn project_from_film(&self, point: Point2D<T>) -> Point3D<T>
    {
        /* Undistort */
        let mut point = point;
        let mut radius = point.magnitude();

        if !radius.is_zero() {
            let distorted_radius = self.radial_distortion.undistort(radius);
            // let distorted_radius = radius;
            point = (point/radius) * distorted_radius;
            radius = distorted_radius;
        }

        /* Project */
        let tan_theta = self.radius_to_theta(radius / self.focal_length).tan();
        Vector3D(
            point.x(),
            point.y(),
            radius / tan_theta
        ).normalised()
    }
}

impl<A, D:Functor<A>> Functor<A> for GeneralCamera<A,D> {
    type Wrapped<B> = GeneralCamera<B,D::Wrapped<B>>;
    #[inline]
    fn fmap<F, B>(self, mut f: F) -> Self::Wrapped<B>
      where F: FnMut(A) -> B {
        GeneralCamera {
            focal_length: f(self.focal_length),
            linearity: f(self.linearity),
            radial_distortion: self.radial_distortion.fmap(&mut f),
        }
    }
}

/*****************************************************************************************/
/*****************************************************************************************/
/*********************************** Radial distortion ***********************************/
/*****************************************************************************************/
/*****************************************************************************************/

pub trait RadialDistortion<T>
{
    fn distort_point(&self, p: Point2D<T>) -> Point2D<T> {
        todo!();
    }

    /* Apply distortion */
    fn distort(&self, x: T) -> T;

    /* Gradient of distort function dx */
    fn dx(&self, _x: T) -> T {
        todo!("Implement one of:\n1. dx()\n2. distort_dx()\n3. undistort()");
    }

    /* Calculate istortion and gradient */
    #[inline]
    fn distort_dx(&self, x: T) -> (T,T) where T: Copy {
        (self.distort(x), self.dx(x))
    }

    /* Automatic inverse as a default, using 4 iterations of newton root finding */
    #[inline]
    fn undistort(&self, y: T) -> T
      where T: Float {
        (0..4).fold(y, |estimate, _| {
            let (x, dx) = self.distort_dx(estimate);
            estimate - (x - y) / dx
        })
    }
}

/*****************************************************************************************/
/************************************* No distortion *************************************/
/*****************************************************************************************/

/* No distortion */
impl<T> RadialDistortion<T> for () {
    #[inline] fn distort(&self, x: T) -> T { x }
    #[inline] fn undistort(&self, x: T) -> T { x }
}

/*****************************************************************************************/
/***************************************** Poly3 *****************************************/
/*****************************************************************************************/

/* Poly3 distortion, same as in Lowe 2007 */
#[derive(Clone, Copy, Debug, Default)]
pub struct Poly3<T> (pub T);

impl<T:Float> RadialDistortion<T> for Poly3<T> {
    #[inline] fn distort(&self, x: T) -> T { x + self.0*x.powi(3) }
    #[inline] fn dx(&self, x: T) -> T { T::one() + self.0*T::frac(3,1)*x.powi(2) }
}

impl<A> Functor<A> for Poly3<A> {
    type Wrapped<B> = Poly3<B>;
    #[inline] fn fmap<F, B>(self, mut f: F) -> Poly3<B> where F: FnMut(A) -> B { Poly3(f(self.0)) }
}

/*****************************************************************************************/
/***************************************** Poly5 *****************************************/
/*****************************************************************************************/

/* Poly5 distortion, suggested by zeiss document */
#[derive(Clone, Copy, Debug, Default)]
pub struct Poly5<T> (pub T, pub T);

impl<T:Float> RadialDistortion<T> for Poly5<T> {
    #[inline] fn distort(&self, x: T) -> T { x + self.0*x.powi(3) + self.1*x.powi(5) }
    #[inline] fn dx(&self, x: T) -> T { T::one() + self.0*T::frac(3,1)*x.powi(2) + self.1*T::frac(5,1)*x.powi(4) }
}

impl<A> Functor<A> for Poly5<A> {
    type Wrapped<B> = Poly5<B>;
    #[inline] fn fmap<F, B>(self, mut f: F) -> Poly5<B> where F: FnMut(A) -> B { Poly5(f(self.0),f(self.1)) }
}

/*****************************************************************************************/
/************************************* PTLens model **************************************/
/*****************************************************************************************/

/* Three parameter polynomial model, equivalent to the 'PTLens' model used in panotools ptgui and lensfun (but without the terrible scaling) */
#[derive(Clone,Copy,Debug,Default)]
pub struct PTLens<T> (pub T, pub T, pub T);

impl<T:Float> RadialDistortion<T> for PTLens<T> {
    #[inline] fn distort(&self, x: T) -> T { x + self.0*x.powi(2) + self.1*x.powi(3) + self.2*x.powi(4) }
    #[inline] fn distort_dx(&self, x: T) -> (T,T) {
        let (a,b,c) = (self.0, self.1, self.2);
        let (x2,x3,x4) = (x.powi(2), x.powi(3), x.powi(4));
        (x+a*x2+b*x3+c*x4, T::one() + T::frac(2,1)*a*x + T::frac(3,1)*b*x2 + T::frac(4,1)*c*x3)
    }
}

impl<A> Functor<A> for PTLens<A> {
    type Wrapped<B> = PTLens<B>;
    #[inline] fn fmap<F, B>(self, mut f: F) -> PTLens<B> where F: FnMut(A) -> B { PTLens(f(self.0),f(self.1),f(self.2)) }
}

