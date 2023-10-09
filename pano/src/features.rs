use maths::linear_algebra::{Point2D};
use opencv::{self as cv, prelude::*};

use optimisation::functor::Functor;
use crate::utils::make_cv_image;
use image::imagebuffer::ImageBuffer;

/* Keypoint match type */
#[derive(Copy, Clone, Debug)]
pub struct Match<T> (pub Point2D<T>, pub Point2D<T>);

impl<A> Functor<A> for Match<A> {
    type Wrapped<B> = Match<B>;
    #[inline]
    fn fmap<F, B>(self, mut f: F) -> Match<B>
      where F: FnMut(A) -> B {
        Match (self.0.fmap(&mut f), self.1.fmap(&mut f))
    }
}

pub trait Descriptor {
    /* Compare descriptors, return a similarity/distance metric */
    fn compare(&self, with: &Self) -> f32;
}

#[derive(Copy, Clone, Debug)]
pub struct SIFTDescriptor ([u8; 128]);

impl SIFTDescriptor
{
    pub fn detect_and_compute(image: &ImageBuffer<1,u8>) -> Result<(Vec<Point2D<f32>>, Vec<SIFTDescriptor>), Box<dyn std::error::Error>>
    {
        let img = make_cv_image(image.width, image.height, &image.data);

        /* SIFT parameters */
        const CONTRAST_THRESHOLD: f64 = 0.03; /* 0.04 is opencv default, 0.09 is equivalent to the original paper */
        const EDGE_THRESHOLD: f64 = 10.;
        const MAX_KEYPOINTS: i32 = 1500; /* Set this to 0 for no limit */

        /* Does nothing */
        let mask = cv::core::Mat::default();

        /* SIFT output goes here */
        let mut cv_keypoints = cv::core::Vector::default();
        let mut cv_descriptors = cv::core::Mat::default();

        /* Run SIFT */
        let mut sift = cv::features2d::SIFT::create_1(MAX_KEYPOINTS, 3, CONTRAST_THRESHOLD, EDGE_THRESHOLD, 1.6, cv::core::CV_8U, false)?;
        sift.detect_and_compute(&img, &mask, &mut cv_keypoints, &mut cv_descriptors, false)?;

        /* Result vectors */
        let num_features = cv_descriptors.rows();
        let mut keypoints = Vec::with_capacity(num_features as usize);
        let mut descriptors = Vec::with_capacity(num_features as usize);

        if (num_features >= 2) && (cv_descriptors.cols() == 128) {
            for i in 0..num_features {
                /* Convert from opencv pointers in to rust data structures */
                let descriptor: Self = unsafe { (std::slice::from_raw_parts(cv_descriptors.row(i)?.data() as *mut _, 1))[0] };
                let cv_point = cv_keypoints.get(i as usize)?;
                descriptors.push(descriptor);
                let (pt, _size, _angle) = (cv_point.pt(), cv_point.size(), cv_point.angle());
                keypoints.push(Point2D(pt.x, pt.y));
            }
            Ok((keypoints, descriptors))
        } else {
            Err("Nothing detected!".into())
        }
    }
}


impl Descriptor for SIFTDescriptor {
    #[inline]
    fn compare(&self, with: &Self) -> f32 {
        /* 3. Runs even faster... but reduced precision and risk of overflow because of u16 (depending on chosen bit shift) */
        let mut sum = 0u16;
        for i in 0..128 {
            let diff = (self.0[i] as i16 - with.0[i] as i16).pow(2);
            /* A shift of 6 is needed to make overflow theoretically impossible, but that much shift reduces precision a lot...
             * but overflow is so unlikely that a shift of as little as 3 had no actual impact on a big panorama,
             * however, only shifting by 2 always led to failures (was always overflowing).
             * So 3 is was the best option. */
            sum += unsafe{std::mem::transmute::<i16,u16>(diff)} >> 3;
        }
        (sum as f32).sqrt()
    }
}

use rayon::prelude::*;

pub fn brute_force_match<T: Descriptor + Sync>(desc1: &Vec<T>, desc2: &Vec<T>, _Threshold: f32) -> Vec<(usize, usize)>
{
    let matches: Vec<_> = (0..desc1.len()).into_par_iter().filter_map(|a| {
        /* Find best and second best match */
        let (mut ind_2nd_best, mut ind_best) = (0, 0);
        let (mut err_2nd_best, mut err_best) = (f32::MAX, f32::MAX);

        for i in 0..desc2.len() {
            let error = desc1[a].compare(&desc2[i]);
            if error < err_best {
                (ind_2nd_best, ind_best) = (ind_best, i);
                (err_2nd_best, err_best) = (err_best, error);
            }
        }

        /* Only points whos first best match is significantly
         * better than the second best match are included */
        (err_best < (err_2nd_best * 0.8)).then_some((a, ind_best))
    }).collect();

    return matches;
}












/*****************************************************************/
/*****************************************************************/
/**** Work in progress, might be a better features abstraction ***/
/*****************************************************************/
/*****************************************************************/

/* A feature detector/matching abstraction */
pub trait Features: Sized {
    /* Detect and compute features using a single channel greyscale image */
    fn detect_and_compute(image: &ImageBuffer<1,u8>) -> Option<Self>;
    /* Match features two images */
    fn compute_matches(&self, with: &Self) -> Vec<Match<f32>>;
}

/* TODO: gate this behind an opencv dependency flag I guess ? */
#[derive(Clone, Debug)]
pub struct OpenCVSIFTFeatures {
    keypoints: Vec<Point2D<f32>>,
    descriptors: Vec<SIFTDescriptor>,
}

impl Features for OpenCVSIFTFeatures {
    fn detect_and_compute(image: &ImageBuffer<1,u8>) -> Option<Self> {
        let img = make_cv_image(image.width, image.height, &image.data);

        /* SIFT parameters */
        const CONTRAST_THRESHOLD: f64 = 0.04; /* 0.04 is default */
        const EDGE_THRESHOLD: f64 = 10.;
        const MAX_KEYPOINTS: i32 = 2750; /* Set this to 0 for no limit */

        /* OpenCV data structures which will store output and mask which does nothing */
        let (mut cv_keypoints, mut cv_descriptors, mask) = (cv::core::Vector::default(), cv::core::Mat::default(), cv::core::Mat::default());

        /* Run SIFT */
        let mut sift = cv::features2d::SIFT::create_1(MAX_KEYPOINTS, 3, CONTRAST_THRESHOLD, EDGE_THRESHOLD, 1.6, cv::core::CV_8U, false).ok()?;
        sift.detect_and_compute(&img, &mask, &mut cv_keypoints, &mut cv_descriptors, false).ok()?;
        let num_features = cv_descriptors.rows();

        /* Make sure more the descriptors output array has 128 columns */
        if (cv_descriptors.cols() != 128) { return None; }

        /* Convert the data to rust data structures now */
        let mut keypoints = Vec::with_capacity(num_features as usize);
        let mut descriptors = Vec::with_capacity(num_features as usize);
        for i in 0..num_features {
            let descriptor: SIFTDescriptor = unsafe { (std::slice::from_raw_parts(cv_descriptors.row(i).ok()?.data() as *mut _, 1))[0] };
            let cv_point = cv_keypoints.get(i as usize).ok()?;
            descriptors.push(descriptor);
            let (pt, _size, _angle) = (cv_point.pt(), cv_point.size(), cv_point.angle());
            keypoints.push(Point2D(pt.x, pt.y));
        }
        Some(Self{keypoints, descriptors})
    }

    fn compute_matches(&self, with: &Self) -> Vec<Match<f32>> {
        brute_force_match(&self.descriptors, &with.descriptors, 0.0)
            .into_iter()
            .map(|(a,b)| Match(self.keypoints[a], with.keypoints[b]))
            .collect()
    }
}