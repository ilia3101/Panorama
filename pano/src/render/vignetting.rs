


// use super::super::image::ImageBuffer;

// pub struct Vignette<T> { pub k1: T, pub k2: T, pub k3: T }

// impl<T: Float> Vignette<T> {
//     pub fn vignette(&self, r: T) -> {
//         T::one()
//             + self.k1 * r.powi(2)
//             + self.k2 * r.powi(4)
//             + self.k3 * r.powi(6)
//     }
// }

// pub struct VignetteCorrespondence<T> {
//     pub points: (Point2D<T>,Point2D<T>),
//     pub values: (T,T),
// }

// pub fn calculate_vignetting_for_images(panorama_alignment: )

// pub struct VignettePair<T> {
//     pub image0: Vignette<T>,
//     pub image1: Vignette<T>,
// }

// // impl<T,C> CalculateResiduals<T,4> for ImagePair<T,C>
// // where
// //     T: Float,
// //     C: Camera<T>
// // {
// //     type Input = Match<T>;

// //     /* Precalculated rotation matrices for cam0->cam1 and cam1->cam0 */
// //     type Context = (Matrix3x3<T>, Matrix3x3<T>);

// //     /* Precalculates forward and inverse rotation matrices */
// //     #[inline]
// //     fn prepare(&self) -> Self::Context {
// //         let rotation = self.get_rotation();
// //         (rotation, rotation.invert3x3())
// //     }

// //     /* Reprojects feature coordinates, returns xy residuals through both cameras */
// //     #[inline]
// //     fn run(&self, ctx: &Self::Context, input: Match<T>) -> [T; 4] {
// //         let (p_cam0, p_cam1) = (input.0, input.1);
// //         let cam0_to_cam1 = self.image1.camera.project_to_film(ctx.0 * self.image0.camera.project_from_film(p_cam0));
// //         let cam1_to_cam0 = self.image0.camera.project_to_film(ctx.1 * self.image1.camera.project_from_film(p_cam1));
// //         let (d0, d1) = (cam1_to_cam0 - p_cam0, cam0_to_cam1 - p_cam1);
// //         [d0[0], d0[1], d1[0], d1[1]]
// //     }
// // }