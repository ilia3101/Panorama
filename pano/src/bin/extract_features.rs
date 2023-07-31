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

use crate::read_image::SourceImage;

/* TODO: clean this up. */
pub fn find_keypoints_in_rawimage(file_path: &str) -> Result<(Vec<Point2D<f32>>, Vec<SIFTDescriptor>), Box<dyn Error>> {
    let mut features_image = SourceImage::new(file_path)?.get_features_image()?;
    run_sift(&mut features_image)
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