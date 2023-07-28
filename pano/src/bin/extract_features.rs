use opencv::{
    prelude::*,
    core::BorderTypes::BORDER_REFLECT,
    imgproc::InterpolationFlags::INTER_AREA,
    self as cv,
};
use pano::utils::*;
use maths::linear_algebra::*;
use pano::features::{SIFTDescriptor};
use image::imagebuffer::ImageBuffer;
use std::error::Error;
use rayon::prelude::*;

/* Loads camera raw images */
use rawloader;

/* Runs sift and maps coordinates in to +-0.5 range */
pub fn run_sift(image: &mut ImageBuffer<1,u8>) -> Result<(Vec<Point2D<f32>>, Vec<SIFTDescriptor>), Box<dyn Error>> {
    let (pt, desc) = SIFTDescriptor::detect_and_compute(image)?;
    Ok((
        /* Map feature coordinates in to a -0.5,+0.5 range */
        pt.into_iter().map(|p|{
            let p = p - Point2D(image.width as f32, image.height as f32) / 2.0;
            p / image.width as f32
        }).collect(),
        desc
    ))
}

/* TODO: clean this up. */
pub fn find_keypoints_in_rawimage(file_path: &str) -> Result<(Vec<Point2D<f32>>, Vec<SIFTDescriptor>), Box<dyn Error>>
{
    /* For some reason opencv makes a black image at above 4x downscaling */
    let feature_downscale_factor = 4;
    let feature_scale = 1.0 / feature_downscale_factor as f64;

    // let start = std::time::SystemTime::now();
    let mut raw = rawloader::decode_file(file_path)?;
    // println!("orientation = {:?} ", raw.orientation);

    /* Get image info */
    let (width, height, _bl, _wl) = (raw.width, raw.height, raw.blacklevels[0], raw.whitelevels[0]);
    let [crop_top, crop_right, crop_bottom, crop_left] = raw.crops; /* Top righ bottom left */
    let real_width = width - (crop_right + crop_left);
    let real_height = height - (crop_top + crop_bottom);

    // println!("real_width: {}, real_height: {}", real_width, real_height);
    // println!("Decode time: {:.1?}ms", std::time::SystemTime::now().duration_since(start).unwrap().as_micros() as f64 / 1000.0);

    match (&mut raw.data, raw.cpp) {
        (rawloader::RawImageData::Integer(ref mut raw_data), 1) => {
            // let start1 = std::time::SystemTime::now();
            /* Make cropped cv_image pointing to the raw data */
            let cv_image = make_cv_image_crop(width, height, crop_top, crop_right, crop_bottom, crop_left, raw_data.as_mut_slice());

            /* Downscale using openCV INTER_LINEAR INTER_LANCZOS4 INTER_AREA */
            let (mut downscaled, mut downscaled_cv) = cv_image_alloc::<u16>(real_width/feature_downscale_factor, real_height/feature_downscale_factor, 1);
            cv::imgproc::resize(&cv_image, &mut downscaled_cv, opencv::core::Size::new(0,0), feature_scale, feature_scale, INTER_AREA as i32)?;

            /* Find and subtract the minimum pixel value in the image to increase contrast */
            let mut min_value = 65535;
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

            // cv::imgcodecs::imwrite("output.png", &some_cv_image, &cv::core::Vector::default())?;

            let mut image = ImageBuffer{
                width: real_width/feature_downscale_factor,
                height: real_height/feature_downscale_factor,
                data: normalised
            };
            
            run_sift(&mut image)
        }
        (rawloader::RawImageData::Integer(ref mut _raw_data), 3) => Err("3 channel raws not yet supported".into()),
        (rawloader::RawImageData::Integer(_), _cpp) => Err("Invalid components per pixel number".into()),
        (rawloader::RawImageData::Float(_), _) => Err("Floating point raw files are not supported".into())
    }
}

/************************************ Match storage data structure ******************************/

#[derive(Clone,Debug)]
pub struct PanoImage<T,D> {
    pub file_path: String,
    pub file_name: String,
    pub width: usize,
    pub height: usize,
    pub keypoints: Vec<Point2D<T>>,
    pub descriptors: Vec<D>,
}

impl<T:From<f32>> PanoImage<T, SIFTDescriptor> {
    pub fn new(file_path: &str) -> Result<Self, Box<dyn Error>> {
        let file_name = std::path::Path::new(file_path).file_name().ok_or("File path error.")?.to_str().ok_or("OsStr to str error")?;
        let (keypoints, descriptors) = find_keypoints_in_rawimage(file_path)?;
        Ok(Self{
            file_path: file_path.to_string(),
            file_name: file_name.to_string(),
            width:0, height:0,
            keypoints: keypoints.into_iter().map(|pt| pt.map(Into::into)).collect(),
            descriptors: descriptors,
        })
    }
}