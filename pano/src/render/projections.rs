use maths::traits::Float;
use maths::linear_algebra::{Point2D, Point3D};
use super::super::alignment::camera::Camera;
use core::f32::consts::TAU;

/* An equirectangular camera for rendering 360 panoramas */
pub struct EquirectangularCamera;

impl Camera<f32> for EquirectangularCamera {
    fn project_to_film(&self, _point: Point3D<f32>) -> Point2D<f32> {
        unimplemented!()
    }
    #[inline]
    fn project_from_film(&self, point: Point2D<f32>) -> Point3D<f32> {
        let (x,y) = (point.x() * TAU, point.y() * TAU);
        let (sinx, cosx) = (x.sin(), x.cos());
        let (siny, cosy) = (y.sin(), y.cos());
        Point3D(cosy * sinx, siny, cosy * cosx)
    }
}

// /* Cylindtrical camera */
// pub struct CylindricalCamera {focal_length: f32}

// impl Camera<f32> for CylindricalCamera {
//     fn project_to_film(&self, _point: Point3D<f32>) -> Point2D<f32> {
//         unimplemented!()
//     }
//     #[inline]
//     fn project_from_film(&self, point: Point2D<f32>) -> Point3D<f32> {
//         let (x,y) = (point.x() * TAU, point.y() * TAU);
//         let (sinx, cosx) = (x.sin(), x.cos());
//         let (siny, cosy) = (y.sin(), y.cos());
//         Point3D(cosy * sinx, siny, cosy * cosx)
//     }
// }