use rawloader;
use image;

fn main() {
    let rl = rawloader::decode_file(std::env::args().nth(1).unwrap()).unwrap();
    let (width, height, bl, wl) = (rl.width, rl.height, rl.blacklevels[0] as f32, rl.whitelevels[0] as f32);

    let mut output = image::ImageBuffer::new(width as u32, height as u32);

    if let rawloader::RawImageData::Integer(raw_data) = rl.data {
        for y in 0..height {
            for x in 0..width {
                let raw_value = (raw_data[y*width+x] as f32 - bl) / (wl - bl);
                let out: u8 = (raw_value.clamp(0.0,1.0).powf(1.0/2.2) * 255.0) as u8;
                *output.get_pixel_mut(x as u32, y as u32) = image::Luma([out]);
            }
        }
        output.save("output.png");
    }
}
