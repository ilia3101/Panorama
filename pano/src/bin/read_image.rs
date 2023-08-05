use opencv::{
    prelude::*,
    core::BorderTypes::BORDER_REFLECT,
    imgproc::InterpolationFlags::INTER_AREA,
    self as cv,
};
use pano::utils::*;
use image::imagebuffer::ImageBuffer;
use rawloader;
use std::error::Error;
use rayon::prelude::*;

#[derive(Debug)]
pub enum SourceImage {
    /* Just a normal 8-bit file such as JPEG/PNG/TIFF */
    Image(cv::core::Mat),
    /* Raw image. */
    RawFile(rawloader::RawImage),
}

fn decode_sRGB(u: f32) -> f32 { if u < 0.04045 {u*(25.0/323.0)} else {((u+0.055)/1.055).powf(2.4)} }
fn decode_sRGB8(u: u8) -> f32 { decode_sRGB(u as f32 / 255.0) }

impl SourceImage {
    pub fn new(file_path: &str) -> Result<Self, Box<dyn Error>> {
        let extension = file_path.to_string().split('.').last().unwrap().to_lowercase();
        match extension.as_str() {
            "jpg" | "jpeg" | "png" | "tiff" => Ok(Self::Image(cv::imgcodecs::imread(file_path, cv::imgcodecs::IMREAD_COLOR)?)),
            "nef" | "cr2" | "crw" | "dng" | "arw" => Ok(Self::RawFile(rawloader::decode_file(&file_path)?)),
            _ => Err("Couldn't read file".into())
        }
    }

    /* Return image for feature detection/matching quickly and at lowered resolution */
    pub fn get_features_image(&self) -> Result<ImageBuffer<1,u8>, Box<dyn Error>> {
        match self {
            Self::RawFile(raw) => {
                /* Always downscale raw files 4x. TODO: maybe use 2x for lower resolutions. (must be a multiple of 2 to remove bayer pattern)
                 * Note: for some reason opencv makes a black image at higher than 4x downscaling factors */
                let feature_downscale_factor = 4;
                let feature_scale = 1.0 / feature_downscale_factor as f64;
    
                /* Get image info */
                let (width, height, _bl, _wl) = (raw.width, raw.height, raw.blacklevels[0], raw.whitelevels[0]);
                let [crop_top, crop_right, crop_bottom, crop_left] = raw.crops; /* Top righ bottom left */
                let real_width = width - (crop_right + crop_left);
                let real_height = height - (crop_top + crop_bottom);

                match (&raw.data, raw.cpp) {
                    (rawloader::RawImageData::Integer(ref raw_data), 1) => {
                        /* Make cropped cv_image pointing to the raw data */
                        let cv_image = make_cv_image_crop(width, height, crop_top, crop_right, crop_bottom, crop_left, raw_data);

                        /* Downscale using openCV. INTER_LINEAR INTER_LANCZOS4 INTER_AREA */
                        let (mut downscaled, mut downscaled_cv) = cv_image_alloc::<u16>(real_width/feature_downscale_factor, real_height/feature_downscale_factor, 1);
                        cv::imgproc::resize(&cv_image, &mut downscaled_cv, opencv::core::Size::new(0,0), feature_scale, feature_scale, INTER_AREA as i32)?;

                        /* Find and subtract the minimum pixel value in the image to increase contrast */
                        let mut min_value = u16::MAX;
                        for &x in &downscaled { if x < min_value { min_value = x; } }
                        if min_value > 0 { for x in downscaled.as_mut_slice().iter_mut() { *x -= min_value; } }

                        /* Blur the image with a box blur 3 times */
                        let radius = 100;
                        let (blurred, mut blurred_cv) = cv_image_alloc::<u16>(real_width/feature_downscale_factor, real_height/feature_downscale_factor, 1);
                        cv::imgproc::blur(&downscaled_cv, &mut blurred_cv, opencv::core::Size::new(radius,radius), opencv::core::Point::new(-1,-1), BORDER_REFLECT as i32)?;
                        let (blurred2, mut blurred2_cv) = cv_image_alloc::<u16>(real_width/feature_downscale_factor, real_height/feature_downscale_factor, 1);
                        cv::imgproc::blur(&blurred_cv, &mut blurred2_cv, opencv::core::Size::new(radius,radius), opencv::core::Point::new(-1,-1), BORDER_REFLECT as i32)?;
                        cv::imgproc::blur(&blurred2_cv, &mut blurred_cv, opencv::core::Size::new(radius,radius), opencv::core::Point::new(-1,-1), BORDER_REFLECT as i32)?;
                        drop(blurred2);

                        /* Divide the image by the blur image to normalise exposure for better feature detection */
                        let mut normalised: Vec<_> = downscaled.into_par_iter().zip(blurred.into_par_iter())
                            .map(|(x, x_blurred)| {
                                let normalised = (x as f32 / x_blurred as f32);
                                (normalised / (normalised + 1.0) * 255.0).clamp(0.0,255.0) as u8
                            }).collect();

                        Ok(ImageBuffer{
                            width: real_width/feature_downscale_factor,
                            height: real_height/feature_downscale_factor,
                            data: normalised
                        })
                    }
                    (rawloader::RawImageData::Integer(ref _raw_data), 3) => Err("3 channel raws not yet supported".into()),
                    (rawloader::RawImageData::Integer(_), _cpp) => Err("Invalid components per pixel number".into()),
                    (rawloader::RawImageData::Float(_), _) => Err("Floating point raw files are not supported".into())
                }
            }
            Self::Image(cv_image) => {
                /* Not raw */
                let (image_width, image_height) = (cv_image.cols() as usize, cv_image.rows() as usize);
                let resolution = ((image_width.pow(2) + image_height.pow(2)) as f64).sqrt();
                /* Select downscale factor */
                let feature_downscale_factor = if resolution < 2000.0 {1} else if resolution < 4000.0 {2} else {4};
                let feature_scale = 1.0 / feature_downscale_factor as f64;
                /* Convert to greyscale */
                let mut cv_image_greyscale = cv::core::Mat::default();
                cv::imgproc::cvt_color(&cv_image, &mut cv_image_greyscale, cv::imgproc::COLOR_BGR2GRAY, 0)?;
                /* Downscale */
                let mut downscaled_cv = cv::core::Mat::default();
                cv::imgproc::resize(&cv_image_greyscale, &mut downscaled_cv, opencv::core::Size::new(0,0), feature_scale, feature_scale, INTER_AREA as i32)?;
                // println!("Downscale = {feature_downscale_factor}");
                // cv::imgcodecs::imwrite("FEARTURES IMAGE.png", &downscaled_cv, &cv::core::Vector::default())?;

                let mut data_u8: Vec<_> = (0..downscaled_cv.rows())
                    .flat_map(|i|
                        unsafe{std::slice::from_raw_parts(downscaled_cv.row(i).unwrap().data() as *const u8, downscaled_cv.cols() as usize)}
                        .iter()
                        .map(|x| *x)
                    ).collect();
                Ok(ImageBuffer::new_with_data(downscaled_cv.cols() as usize, downscaled_cv.rows() as usize, data_u8))
            }
        }
    }

    /* Returns full quality image used for final blending */
    pub fn get_full_quality(&self) -> Result<ImageBuffer<3,f32>, Box<dyn Error>> {
        match self {
            Self::RawFile(raw) => {
                /* TODO: fix this code duplication in btoh functions */
                let (width, height, bl, wl) = (raw.width, raw.height, raw.blacklevels[0], raw.whitelevels[0]);
                let [crop_top, crop_right, crop_bottom, crop_left] = raw.crops; /* Top righ bottom left */
                let real_width = width - (crop_right + crop_left);
                let real_height = height - (crop_top + crop_bottom);

                match (&raw.data, raw.cpp) {
                    (rawloader::RawImageData::Integer(ref raw_data), 1) => {
                        /* Make cropped cv_image pointing to the raw data */
                        let raw_image = make_cv_image_crop(width, height, crop_top, crop_right, crop_bottom, crop_left, raw_data);

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
                    (rawloader::RawImageData::Integer(ref _raw_data), 3) => Err("3 channel raws not yet supported".into()),
                    (rawloader::RawImageData::Integer(_), _cpp) => Err("Invalid components per pixel number".into()),
                    (rawloader::RawImageData::Float(_), _) => Err("Floating point raw files are not supported".into())
                }
            }
            Self::Image(cv_image) => {
                /* Use a lookup table to speed up 8-bit sRGB to float conversion */
                let srgb_to_lin: [f32; 256] = core::array::from_fn(|x| decode_sRGB(x as f32 / 255.0));
                let mut as_f32: Vec<_> = (0..cv_image.rows())
                    .flat_map(|i|
                        unsafe{std::slice::from_raw_parts(cv_image.row(i).unwrap().data() as *const u8, (cv_image.cols() * 3) as usize)}
                        .iter()
                        .map(|x| srgb_to_lin[*x as usize])
                    ).collect();
                Ok(ImageBuffer::new_with_data(cv_image.cols() as usize, cv_image.rows() as usize, as_f32))
            }
        }
    }
}

