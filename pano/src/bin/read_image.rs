use opencv::{
    prelude::*,
    self as cv,
};
use pano::utils::*;
use image::imagebuffer::ImageBuffer;
use rawloader;
use std::error::Error;

/* TODO */
pub fn read_image_full_quality(file_path: &str) -> Result<ImageBuffer<3,f32>, Box<dyn Error>>
{
    let mut raw = rawloader::decode_file(file_path)?;
    let (width, height, bl, wl) = (raw.width, raw.height, raw.blacklevels[0], raw.whitelevels[0]);
    let [crop_top, crop_right, crop_bottom, crop_left] = raw.crops; /* Top righ bottom left */
    let real_width = width - (crop_right + crop_left);
    let real_height = height - (crop_top + crop_bottom);

    match (&mut raw.data, raw.cpp) {
        (rawloader::RawImageData::Integer(ref mut raw_data), 1) => {
            /* Make cropped cv_image pointing to the raw data */
            let raw_image = make_cv_image_crop(width, height, crop_top, crop_right, crop_bottom, crop_left, raw_data.as_mut_slice());

            /* OpenCV demosaicking as a temporary solution */
            let mut dcv = cv::core::Mat::default();
            opencv::imgproc::demosaicing(&raw_image, &mut dcv, opencv::imgproc::COLOR_BayerBG2BGR, 3)?;

            let mut as_f32: Vec<_> = (0..dcv.rows())
                .flat_map(|i|
                    unsafe{std::slice::from_raw_parts(dcv.row(i).unwrap().data() as *const u16, (dcv.cols() * 3) as usize)}
                    .iter()
                    .map(|x| (((*x as f32 - bl as f32) / (wl - bl) as f32) * 5.1))
                ).collect();

            Ok(ImageBuffer::new_with_data(real_width, real_height, as_f32))
        }
        (rawloader::RawImageData::Integer(ref mut _raw_data), 3) => Err("3 channel raws not yet supported".into()),
        (rawloader::RawImageData::Integer(_), _cpp) => Err("Invalid components per pixel number".into()),
        (rawloader::RawImageData::Float(_), _) => Err("Floating point raw files are not supported".into())
    }
}
