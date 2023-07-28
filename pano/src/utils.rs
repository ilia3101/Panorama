use opencv::{prelude::*, self as cv,};

/* Creates an opencv image that uses provided memory slice */
pub fn make_cv_image_crop<T>(
    width: usize, height: usize,
    crop_top: usize, crop_right: usize,
    crop_bottom: usize, crop_left: usize,
    data: &[T]
) -> cv::core::Mat {
    let elements_per_pixel = data.len() / (width * height);
    let element_size = std::mem::size_of::<T>();
    let image_type = match (element_size, elements_per_pixel) {
        (1, 1) => cv::core::CV_8UC1, (1, 3) => cv::core::CV_8UC3,
        (2, 1) => cv::core::CV_16UC1, (2, 3) => cv::core::CV_16UC3,
        (4, 1) => cv::core::CV_32FC1, (4, 3) => cv::core::CV_32FC3,
        _ => panic!("Unkonwn image format")
    };
    let start = (crop_top*width + crop_left) * elements_per_pixel;
    let (_, data) = data.split_at(start);
    let row_bytes = width * element_size * elements_per_pixel;
    unsafe {
        cv::core::Mat::new_rows_cols_with_data(
            (height - crop_bottom - crop_top) as i32,
            (width - crop_left - crop_right) as i32, image_type,
            data.as_ptr() as *mut std::os::raw::c_void,
            row_bytes
        ).unwrap()
    }
}

pub fn make_cv_image<T>(width: usize, height: usize, data: &[T]) -> cv::core::Mat {
    make_cv_image_crop(width, height, 0, 0, 0, 0, data)
}

pub fn cv_image_alloc<T>(width: usize, height: usize, channels: usize) -> (Vec<T>, cv::core::Mat) {
    let mut data = Vec::with_capacity(width*height*channels);
    unsafe { data.set_len(width*height*channels); };
    let cv_image = make_cv_image(width, height, data.as_mut_slice());
    (data, cv_image)
}


use std::num::Wrapping;

/* Uses xorshift* algorithm */
#[derive(Clone,Copy,Debug)]
pub struct XORShiftPRNG (pub Wrapping<u64>);

impl XORShiftPRNG
{
    #[inline] pub fn new(seed: u64) -> Self {Self(Wrapping(seed))}

    #[inline]
    fn get(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0 * Wrapping(0x2545F4914F6CDD1D)).0
    }

    #[inline]
    pub fn get_u64(&mut self, up_to: u64) -> u64 {
        let lz = up_to.leading_zeros();
        let mut value = self.get() >> lz;
        while value >= up_to { value = self.get() >> lz; }
        value
    }

    #[inline]
    pub fn get_usize(&mut self, up_to: usize) -> usize {
        self.get_u64(up_to as u64) as usize
    }
}