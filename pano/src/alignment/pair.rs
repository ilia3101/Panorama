use maths::{
    linear_algebra::{Matrix3x3, Vector2D, Vector},
    traits::{Float},
};

use optimisation::{traits::*, functor::*,};

use super::camera::*;
use super::image::Image;
use super::super::features::Match;

/* Image pairs are used to calculate residuals between a pair of images,
 * both for pair alignment and bundle adjustment */

#[derive(Clone, Copy, Debug, Default)]
pub struct ImagePair<T,CAMERA> {
    pub image0: Image<T,CAMERA>,
    pub image1: Image<T,CAMERA>,
}

impl<T,C> ImagePair<T,C> {
    /* Returns the relative rotation between camera0 and camera1 */
    #[inline]
    pub fn get_rotation(&self) -> Matrix3x3<T>
      where T: Float, C: Camera<T> {
        match (self.image0.try_get_rotation_matrix(), self.image1.try_get_rotation_matrix()) {
            (Some(rotation0), Some(rotation1)) => rotation1 * rotation0.invert3x3(),
            (Some(rotation0), None) => rotation0.invert3x3(),
            (None, Some(rotation1)) => rotation1,
            (None, None) => Matrix3x3::id()
        }
    }
    #[inline]
    pub fn from<C2: Into<C>>(from: ImagePair<T,C2>) -> Self {
        let (i0, i1) = (from.image0, from.image1);
        Self {
            image0: Image { camera: i0.camera.into(), rotation_matrix: i0.rotation_matrix, rotation_xyz: i0.rotation_xyz },
            image1: Image { camera: i1.camera.into(), rotation_matrix: i1.rotation_matrix, rotation_xyz: i1.rotation_xyz },
        }
    }
}

impl<T,C> CalculateResiduals<T,2> for ImagePair<T,C>
where
    T: Float,
    C: Camera<T>
{
    type Input = Match<T>;

    /* Precalculated rotation matrices for cam0->cam1 and cam1->cam0 */
    type Context = (Matrix3x3<T>, Matrix3x3<T>);

    /* Precalculates forward and inverse rotation matrices */
    #[inline]
    fn prepare(&self) -> Self::Context {
        let rotation = self.get_rotation();
        (rotation, rotation.invert3x3())
    }

    /* Reprojects feature coordinates, returns xy residuals through both cameras */
    #[inline]
    fn run(&self, ctx: &Self::Context, input: Match<T>) -> [T; 2] {
        let (p_cam0, p_cam1) = (input.0, input.1);
        let cam0_to_cam1 = self.image1.camera.project_to_film(ctx.0 * self.image0.camera.project_from_film(p_cam0));
        let cam1_to_cam0 = self.image0.camera.project_to_film(ctx.1 * self.image1.camera.project_from_film(p_cam1));
        let (d0, d1) = (cam1_to_cam0.distance(p_cam0), cam0_to_cam1.distance(p_cam1));
        [d0,d1]
        /* TODO: When the distances are zero, there is a numerical gradient issue because of the square root operation.
         * Solve this somehow, maybe by returning the raw x and y residuals from just
         * one of the images instead. */
        // let (d0, d1) = (cam1_to_cam0 - p_cam0, cam0_to_cam1 - p_cam1);
        // [d0[0], d0[1], d1[0], d1[1]]
    }
}

impl<A, C: Functor<A>> Functor<A> for ImagePair<A,C> {
    type Wrapped<B> = ImagePair<B,C::Wrapped<B>>;
    #[inline]
    fn fmap<F, B>(self, mut f: F) -> Self::Wrapped<B>
      where F: FnMut(A) -> B {
        ImagePair {
            image0: self.image0.fmap(&mut f),
            image1: self.image1.fmap(&mut f),
        }
    }
}