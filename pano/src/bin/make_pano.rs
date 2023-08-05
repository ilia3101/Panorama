//cargo build --release --bin make_pano --timings
use opencv::{prelude::*, self as cv};
use std::time::{SystemTime as time};
use std::error::Error;

use optimisation::functor::Functor;
use maths::{linear_algebra::{Matrix3x3}, traits::{Float}};

use pano::{render::render::*, panorama, alignment::camera::Camera, alignment::camera::PTLens, panorama::*, utils::*};
use image::imagebuffer::ImageBuffer;

mod read_image;
use read_image::SourceImage;

extern crate rayon;
use rayon::prelude::*;


/* This can be set to f32 or f64 (theres no performance difference and f32 works worse) */
type FloatType = f64;





/************************************ pano image struct *********************************/

use pano::features::SIFTDescriptor;
use maths::linear_algebra::Point2D;

#[derive(Debug)]
pub struct PanoImage<T,D> {
    pub file_path: String,
    pub file_name: String,
    pub image: SourceImage,
    pub width: usize,
    pub height: usize,
    pub keypoints: Vec<Point2D<T>>,
    pub descriptors: Vec<D>,
}

impl<T:From<f32>> PanoImage<T, SIFTDescriptor>
{
    /* Runs sift and maps coordinates in to +-0.5 range */
    fn run_sift(image: &mut ImageBuffer<1,u8>) -> Result<(Vec<Point2D<f32>>, Vec<SIFTDescriptor>), Box<dyn Error>> {
        let (pt, desc) = SIFTDescriptor::detect_and_compute(image)?;
        Ok((/* Map feature coordinates in to a -0.5,+0.5 range */
            pt.into_iter().map(|p|{
                let p = p - Point2D(image.width as f32, image.height as f32) / 2.0;
                p / image.width.max(image.height) as f32
            }).collect(),
            desc
        ))
    }

    pub fn new(file_path: &str) -> Result<Self, Box<dyn Error>> {
        let image = SourceImage::new(file_path)?;
        let file_name = std::path::Path::new(file_path).file_name().ok_or("File path error.")?.to_str().ok_or("OsStr to str error")?;

        let mut features_image = image.get_features_image()?;
        let (keypoints, descriptors) = Self::run_sift(&mut features_image)?;

        Ok(Self{
            file_path: file_path.to_string(),
            file_name: file_name.to_string(),
            image, width:0, height:0,
            keypoints: keypoints.into_iter().map(|pt| pt.map(Into::into)).collect(),
            descriptors: descriptors,
        })
    }
}




/************************************ Main function ************************************/

fn main() -> Result<(), Box<dyn Error>>
{
    /* Load images and detect features (this happens in PanoImage::new) */
    let start = time::now();
    let file_paths: Vec<_> = std::env::args().skip(1).collect();
    println!("Loading {} images...", file_paths.len());
    let images: Vec<_> = file_paths.par_iter().filter_map(|path| PanoImage::<FloatType,_>::new(path).ok()).collect();
    println!("{} images were loaded in {:.1?} seconds", images.len(), (time::now().duration_since(start)?.as_micros()) as f64 / 1000000.0);




    println!("\nAligning initial pairs...");
    let start_align = time::now();
    let mut image_set = PanoramaImageSet::new();
    let mut image_names = vec![];
    images.iter().for_each(|image| {
        image_set.add_image(image.keypoints.clone(), image.descriptors.clone());
        image_names.push((image.file_name.clone(), image.file_path.clone()))
    });
    image_set.connect_initial_pairs(0);
    println!("{} images were matched and aligned in {:.1?} seconds", image_set.get_num_images(), (time::now().duration_since(start_align)?.as_micros()) as f64 / 1000000.0);



    println!("\nBuilding panorama...");
    let start_pano = time::now();
    let panorama = image_set.build_panorama();
    println!("Panorama was built and refined in {:.1?} seconds", (time::now().duration_since(start_pano)?.as_micros()) as f64 / 1000000.0);






    /* IF alignment was succesful, read all images in full quality and stitch the result */
    if let Some(panorama) = panorama {
        /* Print alignment info */
        println!("\nCamera calibrations:");
        for i in 0..panorama.images.len() {
            if let Some(image) = &panorama.images[i] {
                println!("Image {} ({})", i+1, image_names[i].0);
                let focal_length = image.camera.focal_length * 36.0; /* x36 to make it 35mm film equivalent */
                let linearity = image.camera.linearity;
                let lens_type = if linearity > 0.75 || focal_length > 20.0 { "Rectilinear" }
                    else if linearity > 0.25 {"Stereographic fisheye"}
                    else if linearity > -0.25 {"Equidistant fisheye"}
                    else if linearity > -0.75 {"Equisolid angle fisheye"}
                    else {"Orthographic fisheye"};
                println!(" focal length: {:.2?}mm", focal_length);
                println!(" linearity: {:.2?}", image.camera.linearity);
                println!(" distortion: {:.3?}", image.camera.radial_distortion);
                println!(" lens type: {}", lens_type);
            }
        }

        println!("\nReading images in full quality...");
        let start_read = time::now();
        let mut images_full_quality: Vec<_> = images.into_par_iter()
            .map(|im| im.image.get_full_quality().ok())
            .collect();
        println!("Images read in {:.1?} seconds", (time::now().duration_since(start_read)?.as_micros()) as f64 / 1000000.0);


        let mut renderer = PanoRenderer{
            images: vec![],
            image_data: vec![],
            orientation: Matrix3x3::id()
        };

        for i in 0..panorama.images.len() {
            if let Some(image) = &panorama.images[i] {
                if let Some(image_data) = &mut images_full_quality[i] {
                    renderer.images.push(image.clone().fmap(|x| x as f32));
                    renderer.image_data.push(std::mem::take(image_data));
                }
            }
        }

        println!("\nTotal time {:.1?} seconds", (time::now().duration_since(start)?.as_micros()) as f64 / 1000000.0);

        /* User loop */
        let mut stdin = std::io::stdin();
        let mut input = String::new();
        /* Options */
        let mut is_360 = false;
        let mut camera = CameraModel::<f32>::default();
        let mut exposure = 3.8;
        let mut aspect_ratio = 1.5;
        let mut resolution = 6000usize;

        let pi = std::f32::consts::PI;

        let win = "pano_window";
        cv::highgui::start_window_thread()?;
        cv::highgui::named_window(win, cv::highgui::WINDOW_NORMAL | cv::highgui::WINDOW_GUI_NORMAL)?;
        cv::highgui::set_window_title(win, "Panorama");

        loop {
            println!("Enter command: ");
            input.clear();
            stdin.read_line(&mut input);

            if input.starts_with("exit") {
                break;
            } else if input.starts_with("360") {
                /* Set equirectangular projectxion */
                is_360 = true;
            } else if let Some(input) = input.strip_prefix("fl ") {
                if let Ok(focal_length) = input.trim().parse::<f32>() {
                    camera.focal_length = focal_length / 36.0;
                }
            } else if let Some(input) = input.strip_prefix("rectilinear") {
                camera.linearity = 1.0;
                is_360 = false;
            } else if let Some(input) = input.strip_prefix("stereographic") {
                camera.linearity = 0.5;
                is_360 = false;
            } else if let Some(input) = input.strip_prefix("fisheye") {
                camera.linearity = 0.0001;
                is_360 = false;
            } else if let Some(input) = input.strip_prefix("exposure ") {
                if let Ok(ev) = input.trim().parse() {
                    exposure *= 2.0.powf(ev);
                }
            } else if let Some(input) = input.strip_prefix("resolution ") {
                if let Ok(res) = input.trim().parse() {
                    resolution = res;
                }
            } else if let Some(input) = input.strip_prefix("aspect ") {
                if let Ok(aspect) = input.trim().parse() {
                    aspect_ratio = aspect;
                }
            } else if let Some(input) = input.strip_prefix("linearity ") {
                if let Ok(linearity) = input.trim().parse() {
                    camera.linearity = linearity;
                }
            } else if let Some(input) = input.strip_prefix("yaw ") {
                if let Ok(yaw) = input.trim().parse::<f32>() {
                    renderer.orientation = Matrix3x3::rotation_euler(0.0, -yaw/180.0*pi, 0.0) * renderer.orientation;
                }
            } else if let Some(input) = input.strip_prefix("pitch ") {
                if let Ok(pitch) = input.trim().parse::<f32>() {
                    renderer.orientation = Matrix3x3::rotation_euler(-pitch/180.0*pi, 0.0, 0.0) * renderer.orientation;
                }
            } else if let Some(input) = input.strip_prefix("roll ") {
                if let Ok(roll) = input.trim().parse::<f32>() {
                    renderer.orientation = Matrix3x3::rotation_euler(0.0, 0.0, roll/180.0*pi) * renderer.orientation;
                }
            } else if let Some(file_name) = input.strip_prefix("write ") {
                /* Save panorama */
                println!("Writing to file: {}", file_name.trim());
                let mut pano = render(&renderer, is_360, camera, aspect_ratio, resolution, exposure);
                let cv_image = make_cv_image(pano.width, pano.height, &mut pano.data);
                cv::imgcodecs::imwrite(file_name.trim(), &cv_image, &cv::types::VectorOfi32::new())?;
            } else if input.starts_with("view") {
                /* Show panorama */
                let mut pano = if is_360 {render(&renderer, is_360, camera, aspect_ratio, 3200, exposure)}
                                else {render(&renderer, is_360, camera, aspect_ratio, 1800, exposure)};
                let cv_image = make_cv_image(pano.width, pano.height, &mut pano.data);
                cv::highgui::imshow(win, &cv_image)?;
                cv::highgui::wait_key(1)?;
                cv::highgui::destroy_window(win)?;
                // cv::highgui::set_window_property(win, cv::highgui::WND_PROP_TOPMOST, 1.)?;
            } else {
                help();
            }
        }
    }
    else {
        println!("Failed to align images.");
    }

    Ok(())
}

fn render(renderer: &PanoRenderer<3,CameraModel<f32>>, is_360: bool, camera: CameraModel<f32>, aspect: f32, resolution: usize, exposure: f32) -> ImageBuffer<3,u8>
{
    let mut pano = if is_360 {
        renderer.render_360(resolution/2)
    } else {
        let height = resolution;
        let width = (resolution as f32 * aspect) as usize;
        renderer.render_pano(camera, width, height)
    };
    pano.save("PANO_TEST.png");
    let (width, height) = (pano.width, pano.height);
    // pano.data.par_iter_mut().skip(1).step_by(3).for_each(|x| *x = *x * 0.56);
    let mut as_u8: Vec<_> = pano.data.into_par_iter().map(|x| {
        // let exposure_adjusted = (x.max(0.0) * exposure).powf(1.65);
        // let compressed = (exposure_adjusted / (exposure_adjusted + 1.0));
        // (encode_sRGB(compressed) * 255.0) as u8
        (encode_sRGB(x) * 255.0) as u8
    }).collect();
    return ImageBuffer::new_with_data(width, height, as_u8);
}

fn encode_sRGB(u: f32) -> f32 {
    if u < 0.0031308 {(323.0*u)/25.0} else {1.055 * u.powf(1.0/2.4) - 0.055}
}

fn help()
{
    println!("Commands:");
    println!("   write file_name    - write to file");
    println!("   fl                 - set focal length");
    println!("   exposure           - adjust exposure");
    println!("   linearity          - set projection linearity parameter");
    println!("   rectilinear        - set rectilinear projection (1.0)");
    println!("   stereographic      - set stereographic projection (0.5)");
    println!("   fisheye            - set fisheye projection (0.0)");
    println!("   360                - set 360 degree equirectangular projection");
    println!("   yaw                - rotate yaw");
    println!("   pitch              - rotate pitch");
    println!("   roll               - rotate roll");
    println!("   aspect             - set aspect ratio");
    println!("   resolution         - set output resolution (horizontal)");
    println!("   view               - show preview");
    println!("");
}