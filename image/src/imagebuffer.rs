#[derive(Clone,Default)]
pub struct ImageBuffer<const NCHANNELS: usize, T> {
    pub width: usize,
    pub height: usize,
    pub data: Vec<T>,
}

impl<const NCHANNELS: usize, T> ImageBuffer<NCHANNELS, T>
{
    #[inline]
    pub fn new(width: usize, height: usize) -> Self where T: Default + Copy {
        Self { width, height, data: vec![T::default(); width*height*NCHANNELS] }
    }

    #[inline]
    pub fn new_with_data(width: usize, height: usize, data: Vec<T>) -> Self {
        Self { width, height, data }
    }

    #[inline]
    fn row_index(&self, y: usize) -> usize {
        y * self.width * NCHANNELS
    }

    #[inline]
    fn row(&self, y: usize) -> &[T] {
        let index = self.row_index(y);
        &self.data[index..(index+self.width*NCHANNELS)]
    }

    // #[inline]
    // pub fn row_iter_mut(&self) -> u32 {
    //     self.data.chunks_exact_mut(self.width * NCHANNELS)
    // }
}

/************************* Image Saving **************************/

// use image;
fn encode_sRGB(u: f32) -> f32 { if u < 0.0031308 {u*(323.0/25.0)} else {1.055 * u.powf(1.0/2.4) - 0.055} }
fn encode_sRGB8(u: f32) -> u8 { (encode_sRGB(u) * 255.0) as u8 }
impl ImageBuffer<3,f32> {
    pub fn save(&self, out_file: &str) {
        // let mut output = image::ImageBuffer::new(self.width as u32, self.height as u32);
        // for y in 0..self.height {
        //     let rowi = self.row_index(y);
        //     for x in 0..self.width {
        //         let idx = rowi + x*3;
        //         *output.get_pixel_mut(x as u32, y as u32) = image::Rgb(
        //             [self.data[idx+2], self.data[idx+1], self.data[idx+0]].map(encode_sRGB8)
        //         );
        //     }
        // }
        // output.save(out_file);
    }
}


impl<const NCHANNELS: usize> ImageBuffer<NCHANNELS, f32>
{
    /* x and y coordinates are -0.5 to +0.5 */
    #[inline]
    pub fn sample(&self, x: f32, y: f32) -> Option<[f32; NCHANNELS]> {
        /* 0.5,0.5 = pixel center */
        let x_pix = x - 0.5;
        let y_pix = y - 0.5;
        
        let xi = x_pix.max(0.0) as usize;
        let yi = y_pix.max(0.0) as usize;

        if x_pix >= 0.0 && y_pix >= 0.0 && xi < (self.width-1) && yi < (self.height-1) {
            let x_off = x_pix - xi as f32;
            let y_off = y_pix - yi as f32;
            let (x_offi, y_offi) = (1.0 - x_off, 1.0 - y_off);

            let x0y0 = y_offi * x_offi;
            let x1y0 = y_offi * x_off;
            let x0y1 = y_off * x_offi;
            let x1y1 = y_off * x_off;

            let row_length = self.width * NCHANNELS;
            let row0 = row_length * yi;
            let row1 = row0 + row_length;

            let x_offset = NCHANNELS * xi;
            Some(core::array::from_fn(|c| {
                self.data[row0+x_offset+c] * x0y0 + self.data[row0+x_offset+NCHANNELS+c] * x1y0 +
                self.data[row1+x_offset+c] * x0y1 + self.data[row1+x_offset+NCHANNELS+c] * x1y1
            }))
        }
        else { None }
    }

    /* Function for aampling in a -0.5,+0.5 coordinate systsem */
    #[inline]
    pub fn sample_normalised(&self, x: f32, y: f32) -> Option<[f32; NCHANNELS]> {
        if self.width > self.height {
            let x = (x + 0.5) * self.width as f32;
            let y = (y * self.width as f32) + self.height as f32 * 0.5;
            self.sample(x, y)
        } else {
            let x = (x * self.height as f32) + self.width as f32 * 0.5;
            let y = (y + 0.5) * self.height as f32;
            self.sample(x, y)
        }
    }

    /* Calcualtes a weight for simple panorama blending. TODO: move this somewhere else */
    #[inline]
    pub fn weight_normalised(&self, x: f32, y: f32) -> f32 {
        if self.width > self.height {
            let x = (x + 0.5) * self.width as f32;
            let y = (y * self.width as f32) + self.height as f32 * 0.5;
            self.weight(x, y)
        } else {
            let x = (x * self.height as f32) + self.width as f32 * 0.5;
            let y = (y + 0.5) * self.height as f32;
            self.weight(x, y)
        }
    }

    #[inline]
    fn weight(&self, x: f32, y: f32) -> f32 {
        let x = x / self.width as f32;
        let y = y / self.height as f32;
        let weight_x = 1.0 - (x - 0.5).abs() * 2.0;
        let weight_y = 1.0 - (y - 0.5).abs() * 2.0;
        let weight = weight_x * weight_y;
        weight * weight
    }
}
