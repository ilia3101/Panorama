use maths::{
    traits::Float,
    linear_algebra::{Matrix3x3, Matrix, Vector3D}
};
use optimisation::functor::Functor;
use super::camera::Camera;

/* An "Image" contains a camera and its orientation */
#[derive(Clone,Copy,Debug,Default)]
pub struct Image<T,CAMERA> {
    /* A camera object */
    pub camera: CAMERA,
    /* Orientation is set using a matrix and a yaw-pitch-roll xyz triplet,
     * which is applied after the matrix */
    pub rotation_matrix: Option<Matrix3x3<T>>,
    pub rotation_xyz: Option<Vector3D<T>>
}

impl<T: Float, C: Camera<T>> Image<T,C>
{
    /* Commits all rotation adjutsments into the main rotation matrix,
     * resets the yaw pitch roll adjustment */
    pub fn apply_rotation(&mut self) {
        self.rotation_matrix = self.try_get_rotation_matrix();
        self.rotation_xyz = None;
    }

    /* Returns the overall rotation matrix if any of rotation parameters are present */
    pub fn try_get_rotation_matrix(&self) -> Option<Matrix3x3<T>> {
        match (self.rotation_matrix, self.rotation_xyz) {
            (Some(mat), Some(xyz)) => Some(Matrix::rotation_euler(xyz.x(),xyz.y(),xyz.z()) * mat),
            (None, Some(xyz)) => Some(Matrix::rotation_euler(xyz.x(),xyz.y(),xyz.z())),
            (Some(mat), None) => Some(mat),
            (None, None) => None
        }
    }

    pub fn get_rotation_matrix(&self) -> Matrix3x3<T> {
        self.try_get_rotation_matrix().unwrap_or(Matrix3x3::id())
    }
}

impl<A, C: Functor<A>> Functor<A> for Image<A,C> {
    type Wrapped<B> = Image<B,C::Wrapped<B>>;
    #[inline]
    fn fmap<F, B>(self, mut f: F) -> Self::Wrapped<B>
      where F: FnMut(A) -> B {
        Image {
            camera: self.camera.fmap(&mut f),
            rotation_matrix: self.rotation_matrix.fmap(&mut f),
            rotation_xyz: self.rotation_xyz.fmap(&mut f),
        }
    }
}