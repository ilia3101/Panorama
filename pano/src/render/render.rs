use super::projections::EquirectangularCamera;
use super::super::alignment::image::*;
use super::super::alignment::camera::*;
use image::imagebuffer::ImageBuffer;
use maths::linear_algebra::{Vector3D, Point2D, Matrix3x3};
use rayon::prelude::*;


pub struct PanoRenderer<const N_CHANNELS: usize, CAMERA> {
    pub images: Vec<Image<f32, CAMERA>>,
    pub image_data: Vec<ImageBuffer<N_CHANNELS,f32>>,
    pub orientation: Matrix3x3<f32>,
}

impl<const N_CHANNELS: usize, CAMERA: Camera<f32> + Sync> PanoRenderer<N_CHANNELS, CAMERA>
{
    pub fn render_pano(
        &self,
        view: impl Camera<f32> + Sync,
        width: usize,
        height: usize
    ) -> ImageBuffer<N_CHANNELS,f32>
    {
        let image_orientations: Vec<_> = self.images.iter().map(|image| {
            image.get_rotation_matrix() * self.orientation.invert3x3()
        }).collect();

        /* Determine limits of each image based on dot product with corner angle,
         * this reduces wasted projections into cameras and potential glitches */
        let image_limits: Vec<_> = self.images.iter().zip(self.image_data.iter()).map(|(image, imdata)| {
            let corner_point = Point2D(0.5, 0.5 / imdata.width as f32 * imdata.height as f32);
            let point = image.camera.project_from_film(corner_point).normalised();
            point.dot(Vector3D(0.,0.,1.))
        }).collect();

        /* For linear blending */
        let mut weights = ImageBuffer::<1,f32>::new(width, height);
        let mut result = ImageBuffer::new(width, height);

        let row_iter = (0..height)
            .map(|y| (y as f32 - height as f32 / 2.0) / width as f32)
            .zip(result.data.chunks_exact_mut(width * N_CHANNELS))
            .zip(weights.data.chunks_exact_mut(width))
            .par_bridge(); // Parallel

        row_iter.for_each(|((y, row), row_weights)| {
            for i in 0..self.images.len() {
                let image = &self.images[i];
                let imagedata = &self.image_data[i];
                let rotate = image_orientations[i];
                let limit = image_limits[i];
                let iter_pixels = (0..width)
                    .map(|x| (x as f32 / width as f32) - 0.5)
                    .zip(row.chunks_exact_mut(N_CHANNELS))
                    .zip(row_weights.iter_mut());
                iter_pixels.for_each(|((x, pix), sum_weight)| {
                    let view_vec = view.project_from_film(Point2D(x,y));
                    let image_vec = rotate * view_vec;
                    if image_vec.dot(Vector3D(0.,0.,1.)) > limit {
                        let im_coord = image.camera.project_to_film(image_vec);
                        if let Some(value) = imagedata.sample_normalised(im_coord.x(),im_coord.y()) {
                            let weight = imagedata.weight_normalised(im_coord.x(),im_coord.y());
                            *sum_weight += weight;
                            for i in 0..N_CHANNELS { pix[i] += value[i] * weight; }
                            // for i in 0..N_CHANNELS { pix[i] = value[i]; }
                        }
                    }
                });
            }
        });

        /* Normalise the weights */
        result.data.chunks_exact_mut(width * N_CHANNELS)
            .zip(weights.data.chunks_exact_mut(width))
            .par_bridge()
            .for_each(|(row, row_weights)| {
                let iter_pixels = row.chunks_exact_mut(N_CHANNELS).zip(row_weights.iter_mut());
                iter_pixels.for_each(|(pix, weight)| {
                    if *weight != 0.0 {
                        pix.iter_mut().for_each(|x| *x /= *weight);
                    }
                });
            });

        return result;
    }

    /* Result width is double the height */
    pub fn render_360(&self, height: usize) -> ImageBuffer<N_CHANNELS,f32> {
        self.render_pano(EquirectangularCamera, height*2, height)
    }
}