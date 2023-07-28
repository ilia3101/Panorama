use super::features::{Match, Descriptor, brute_force_match};
use super::alignment::{ransac::*, pair::*, image::*, camera::*};
use optimisation::{parameters::*, functor::*, optimise::*, traits::*};
use maths::{
    linear_algebra::{Point2D, Vector},
    traits::{Float, Zero}
};
use rayon::prelude::*;

/************************************************************************************************/
/************************************************************************************************/
/**************************************** Configuration *****************************************/
/************************************************************************************************/
/************************************************************************************************/

/* Global configuration */
pub type CameraModel<T> = GeneralCamera<T, PTLens<T>>;

/************************************************************************************************/
/************************************************************************************************/
/**************************************** Configuration *****************************************/
/************************************************************************************************/
/************************************************************************************************/


#[derive(Debug, Clone)]
pub struct PairAlignment<T> {
    pub alignment: ImagePair<T, CameraModel<T>>,
    pub inlier_set: Vec<Match<T>>
}

impl<T: Float + Send + Sync> PairAlignment<T>
{
    /* Creates a pair alignment from matches.
     *
     * Multiple passes of RANSAC are done, first with a very large threshold
     * for many iterations, allowing lower quality inliers.
     * 
     * The inlier set is further refined by performing RANSAC on the inlier
     * set while comparing against the full set, this improves the inlier
     * quality and count.
     * 
     * Finally, a low threshold RANSAC is used to refine the set
     * and make it as accurate as possible.
     */
    #[inline]
    pub fn new(all_matches: &[Match<T>]) -> Option<Self>
    {
        if all_matches.len() < 40 { return None; }
    
        /* Thresholds for initial RANSAC, loose and accurate */
        let (threshold1, threshold2) = (T::frac(1,40), T::frac(1,800));
    
        /******************************* RASNSAC *******************************/
        /* First pass of RANSAC with a large threshold.
         * Allows for a large number of RANSAC iterations, with a minimum of 50 */
        let (alignment, mut best_num_inliers) = ransac(
            /* TODO: decide how many ransac iterations to allow, 25000 or 2500 (5 or 4 figures) */
            2500, 50, Some(1.0e-8), 4, all_matches,
            |matches| {
                /* Start with pinhole mode */
                let mut model = ImagePair::<T,PinholeCamera<T>>::default().to_pars();
                model.image0.rotation_xyz = Some(Vector::zero()).to_pars();
                /* Unlock rotation parameters and one camera's focal length, take one step */
                model.image0.rotation_xyz.unlock();
                model.image0.camera.focal_length.unlock();
                model = model.refine(matches, T::frac(5,10), 1)?;
                let (f0, f1) = (model.image0.camera.focal_length.value, model.image1.camera.focal_length.value);
                if (f0 < T::zero()) != (f1 < T::zero()) { return None; }
                /* Optimise with both cameras focal lengths unlocked */
                model.image1.camera.focal_length.unlock();
                model = model.refine(matches, T::frac(5,10), 10)?;
                
                let (f0, f1) = (model.image0.camera.focal_length.value, model.image1.camera.focal_length.value);
                if (f0 < T::zero()) != (f1 < T::zero()) {
                    return None;
                } if (f0 < T::frac(4,36)) || (f1 < T::frac(4,36)) {
                    return None;
                } if (f0 > T::frac(1500,36)) || (f1 > T::frac(1500,36)) {
                    return None;
                }

                /* Add linearity parameter, try further optimisation */
                let mut model = ImagePair::<_,CameraModel<_>>::from(model);
                model.image0.camera.linearity.unlock();
                model.image1.camera.linearity.unlock();
                if let Some(refined) = model.refine(matches, T::frac(5,10), 5) {
                    if let Some(refined) = refined.refine(matches, T::frac(7,10), 5) {
                        model = refined;
                    }
                }

                /* Make sure both focal lengths are of the same sign (flip them if they are negative) */
                let (f0, f1) = (&mut model.image0.camera.focal_length, &mut model.image1.camera.focal_length);
                if (f0.value < T::zero()) == (f1.value < T::zero()) {
                    (f0.value, f1.value) = (f0.value.abs(), f1.value.abs());
                    Some(model)
                } else { None }
            },
            |result| result.from_pars().count_inliers(all_matches, threshold1),
            |_,_| false
        );
        
        println!("num inliers = {}", best_num_inliers);
        /* Not enough inliers */
        if best_num_inliers < 50 { return None; }
        let mut alignment = alignment?;

        /* Refine the inlier set a few times */
        // for _i in 0..4 {
        //     let inlier_set = alignment.from_pars().filter_outliers(all_matches, threshold1);
        //     (alignment, best_num_inliers) = ransac_refine(
        //         alignment, best_num_inliers, 1000, 10, 4, &inlier_set,
        //         |matches| alignment.refine(matches, T::frac(6,10), 10),
        //         |result| result.from_pars().count_inliers(all_matches, threshold1),
        //     );
        // }

        // THIS IS BAD !! ! ALLOWING DISTORTION CAUSES OVERFITTING !!!!!!!!!! ONLY USE FISHEYE FACTOR !!!!!
        /* Lock fisheye parameter and unlock the distortion parameters (optimising both at once causes issues) */
        // alignment.image0.camera.linearity.lock();
        // alignment.image1.camera.linearity.lock();
        // alignment.image0.camera.radial_distortion.unlock();
        // alignment.image1.camera.radial_distortion.unlock();
    
        /* Final round of RANSAC, with a very small threshold and using more points per iteration (5 or 6, previously 8) */
        let inlier_set = alignment.from_pars().filter_outliers(all_matches, threshold1);
        let (mut alignment_extra_refined, _best_num_inliers) = ransac(
            500, 25, Some(0.0), 5, &inlier_set,
            |matches| alignment.refine(matches, T::frac(6,10), 10),
            |result| result.from_pars().count_inliers(all_matches, threshold2),
            |_,_| false
        );
        /* If the final optimisation worked, use it... */
        let mut alignment = alignment_extra_refined.unwrap_or(alignment);

        // println!("alignment = {:#.3?}", alignment.from_pars());
        // println!("alignment = {:#.3?}", alignment.from_pars().image0.camera.focal_length);

        /* Refine on the entire inlier set as the final step */
        let inlier_set = alignment.from_pars().filter_outliers(all_matches, threshold2);
        if inlier_set.len() < 15 { return None; }
        alignment = alignment.refine(&inlier_set, T::frac(9,10), 10)?;
        // println!("alignment = {:#?}", alignment.from_pars());

        let mut result = alignment.from_pars();
        result.image0.camera.focal_length = result.image0.camera.focal_length.abs();
        result.image1.camera.focal_length = result.image1.camera.focal_length.abs();
        Some(PairAlignment{alignment: result, inlier_set})
    }
}

#[derive(Debug,Clone)]
pub struct PanoramaAlignment<T> {
    pub images: Vec<Option<Image<T,CameraModel<T>>>>,
    pub edges: Vec<(usize,usize,Vec<Match<T>>)>,
}

impl<T: Float + Send + Sync> PanoramaAlignment<T>
{
    pub fn bundle_adjust(&mut self, n_iterations: usize, learning_rate: T) -> Result<(),()>
    {
        // self.images.iter().for_each(|im| if let Some(im)=im {println!("{:#?}\n{:?}", im.camera, im.rotation_xyz)});

        let mut param_id_vecs = vec![];
        let mut is_first = true;
        let mut as_params: Vec<Option<Image<Parameter<T>,CameraModel<Parameter<T>>>>> = self.images.iter().map(|im| {
            im.map(|mut im| {
                im.apply_rotation();
                if is_first { is_first = false }
                else { im.rotation_xyz = Some(Vector::zero()); }
                let mut as_pars = im.to_pars();
                /* Thge following parameters will be adjusted. TODO: allow user control. */
                as_pars.camera.radial_distortion.unlock();
                // as_pars.camera.linearity.unlock();
                as_pars.camera.focal_length.unlock();
                if !is_first { as_pars.rotation_xyz.unlock(); }
                param_id_vecs.push(as_pars.find_unique_unlocked_parameters());
                as_pars
            })
        }).collect();

        /* Merge parameter lists */
        let all_unique_param_ids = {
            let mut all_ids: Vec<_> = param_id_vecs.into_iter().flatten().collect();
            all_ids.sort(); all_ids.dedup();
            all_ids
        };

        println!("Optimising {} parameters", all_unique_param_ids.len());

        for _i in 0..n_iterations {
            let blocks: Vec<_> = self.edges.par_iter().filter_map(|(a,b,matches)| {
                let imagepair = ImagePair {
                    image0: as_params[*a].unwrap(),
                    image1: as_params[*b].unwrap(),
                };
                let unique_params = imagepair.find_unique_unlocked_parameters().len();
                // println!("{} unique params", unique_params);
                if unique_params <= 8 { imagepair.calc_block::<8>(matches) }
                else if unique_params <= 12 { imagepair.calc_block::<12>(matches) }
                else { imagepair.calc_block::<16>(matches) }
            }).collect();

            if let Some(step) = sparse_step(&all_unique_param_ids, &blocks) {
                if step.iter().any(|x| x.is_nan()) { return Err(()); }
                for im in as_params.iter_mut() {
                    *im = im.map(|im| im.apply_step(&step, &all_unique_param_ids, learning_rate));
                }
            }
        }

        for i in 0..self.images.len() {
            self.images[i] = as_params[i].map(|im| im.fmap(|par| par.value));
        }

        Ok(())
    }
}

#[derive(Debug,Clone,Default)]
pub struct PanoramaImageSet<T,D> {
    /* Images (feature coordinates, corresponding descriptors) */
    pub images: Vec<(Vec<Point2D<T>>, Vec<D>)>,
    /* Pair matches (index_a, index_b, vector_of_matches) */
    pub pair_matches: Vec<(usize,usize,Vec<Match<T>>)>,
    /* Pair alignments, unsuccessful ones are stored as well (index_a, index_b, alignment) */
    pub edges: Vec<(usize,usize,Option<PairAlignment<T>>)>,
    /* Connected images (main panorama) */
    pub connected: Vec<usize>
}

impl<T: Float + Send + Sync, D: Descriptor + Sync> PanoramaImageSet<T,D>
{
    pub fn new() -> Self { Self {images: vec![], pair_matches: vec![], edges: vec![], connected: vec![]} }

    pub fn add_image(&mut self, keypoints: Vec<Point2D<T>>, descriptors: Vec<D>) {
        self.images.push((keypoints, descriptors));
    }

    pub fn get_num_images(&self) -> usize { self.images.len() }

    /* Tries to find alignments pairs to connect all images,
     * assumes they are somewhat ordered */
    pub fn connect_initial_pairs(&mut self, key_image: usize)
    {
        if self.images.len() < 2 {return;}

        let num_images = self.images.len();
    
        /* Assume first image is connected */
        let mut connected = vec![key_image];
        let mut unconnected: Vec<_> = (0..num_images).filter(|&i| i != key_image).collect();
        let mut connections = vec![];

        /* Try to connect images sequentially */
        while unconnected.len() > 0 {
            let still_unconnected: Vec<_> = unconnected.iter().filter(|&i| {
                match connected.iter().rev().find_map(|&c| Some(self.pair_alignment(c,*i)?.clone())) {
                    Some(alignment) => {
                        connected.push(*i);
                        connections.push(alignment);
                        false
                    }, None => true,
                }
            }).map(|i| *i).collect();
            /* No progress was made. This means the rest are not connected. Break. */
            if unconnected.len() == still_unconnected.len() { break; }
            unconnected = still_unconnected;
        }

        /* Try to connect the first couple of images with the last image */
        let prev = (key_image as isize - 1 + num_images as isize) % num_images as isize;
        let next = (key_image + 1) % num_images;
        // println!("Alignineg first and last");
        self.pair_alignment(0,num_images-1);
        // self.pair_alignment(2,num_images-1);
        // self.pair_alignment(3,num_images-1);
        self.pair_alignment(key_image, prev as usize);
        self.pair_alignment(next, prev as usize);

        /* If theres only 2 images connect them backwards to inprove quality of alignment */
        if self.images.len() == 2 {
            self.pair_alignment(1,0);
        }

        self.connected = connected;
    }

    /* Builds panorama based on current pairs */
    pub fn build_panorama(&mut self) -> Option<PanoramaAlignment<T>>
    {
        if self.images.len() < 2 {return None;}

        /* Take all current connections */
        let mut connections = self.edges.iter().filter_map(|(a,b,al)| Some((*a,*b,al.clone()?))).collect::<Vec<_>>();

        /* Empty panorama */
        let mut panorama = PanoramaAlignment::<T> {
            images: vec![None; self.images.len()],
            edges: vec![]
        };

        /* Add first image pair */
        let first_pair = connections.pop()?;
        panorama.images[first_pair.0] = Some(first_pair.2.alignment.image0);
        panorama.images[first_pair.1] = Some(first_pair.2.alignment.image1);
        panorama.edges.push((first_pair.0, first_pair.1, first_pair.2.inlier_set.clone()));

        /* Add remaining pairs */
        while connections.len() > 0 {
            for i in 0..connections.len() {
                let (a,b,connection) = &connections[i];
                let (contains_a, contains_b) = (panorama.images[*a].is_some(), panorama.images[*b].is_some());
                if contains_a && contains_b {
                    /* Both images are alredy added to the panorama, so this set of points is only being
                     * added to improve alignment, not for the purpose of adding a new image */
                    panorama.edges.push((*a,*b,connection.inlier_set.clone()));
                    connections.remove(i);
                    /* Do bundle adjustment */
                    panorama.bundle_adjust(5, T::frac(8,10));
                    break;
                }
                else if contains_a != contains_b {
                    /* Only one of the images is already contained, this means a new image is being added to the panorama */
                    if contains_a {
                        /* Add image B, roughly aligned with A */
                        let mut image = connection.alignment.image1;
                        image.rotation_matrix = Some(connection.alignment.get_rotation() * panorama.images[*a].unwrap().get_rotation_matrix());
                        image.rotation_xyz = None;
                        image.camera.focal_length = image.camera.focal_length.abs(); //TODO: IMPROVE THIS
                        panorama.images[*b] = Some(image);

                        // let mut pair = connection.alignment.to_pars();
                        // pair.image1 = panorama.images[*a].unwrap().to_pars();
                        // pair.image0 = pair.image1;
                        // pair.image1.rotation_xyz.unlock();
                        // pair.image1.camera.focal_length.unlock();
                        // pair.image1.camera.linearity.unlock();
                        // pair.image1.camera.radial_distortion = Default::default();
                        // if let Some(refined) = pair.refine(&connection.inlier_set, T::frac(5,10), 50) {
                        //     panorama.images[*b] = Some(refined.image1.from_pars());
                        //     panorama.edges.push((*a,*b,connection.inlier_set.clone()));
                        // }
                    } else {
                        /* Add image A, roughly aligned with B */
                        let mut image = connection.alignment.image0;
                        image.rotation_matrix = Some(connection.alignment.get_rotation().invert3x3() * panorama.images[*b].unwrap().get_rotation_matrix());
                        image.rotation_xyz = None;
                        image.camera.focal_length = image.camera.focal_length.abs();
                        panorama.images[*a] = Some(image);

                        // let mut pair = connection.alignment.to_pars();
                        // pair.image1 = panorama.images[*b].unwrap().to_pars();
                        // pair.image0 = pair.image1;
                        // pair.image0.rotation_xyz.unlock();
                        // pair.image0.camera.focal_length.unlock();
                        // pair.image0.camera.linearity.unlock();
                        // pair.image0.camera.radial_distortion = Default::default();
                        // if let Some(refined) = pair.refine(&connection.inlier_set, T::frac(5,10), 50) {
                        //     panorama.images[*a] = Some(refined.image0.from_pars());
                        //     panorama.edges.push((*a,*b,connection.inlier_set.clone()));
                        // }
                    }
                    panorama.edges.push((*a,*b,connection.inlier_set.clone()));
                    connections.remove(i);

                    /* Now do bundle adjustment */
                    panorama.bundle_adjust(5, T::frac(7,10));
                    break;
                }
                else {
                    /* Cant be added to the panorama, not connected */
                }
            }
        }

        // panorama.images.iter().for_each(|im| if let Some(im)=im {println!("{:#?}\n{:?}", im.camera, im.rotation_xyz)});

        /* Bundle adjustment optimise */
        panorama.bundle_adjust(50, T::frac(8,10));

        /* Print final results */
        // panorama.images.iter().for_each(|im| if let Some(im)=im {println!("{:#?}\n{:?}", im.camera, im.rotation_xyz)});
        // println!("num edges: {}", panorama.edges.len());
        // println!("edges: {:?}", panorama.edges.iter().map(|(a,b,_)| (*a,*b)).collect::<Vec<_>>());

        Some(panorama)
    }

    pub fn connect_overlapping_pairs(&mut self, _panorama: &mut PanoramaAlignment<T>) {
        todo!()
        /* Loop through all pairs that aren't connected and check how much they are overlapping */
    }

    /**************************** Private functions ****************************/

    fn do_matching(&self, a: usize, b: usize) -> Vec<Match<T>> {
        brute_force_match(&self.images[a].1, &self.images[b].1, 0.0)
            .into_iter()
            .map(|(c,d)| Match(self.images[a].0[c], self.images[b].0[d]))
            .collect()
    }

    /* Returns matches between a and b, if not already found */
    fn pair_matches(&mut self, a: usize, b: usize) -> &Vec<Match<T>> {
        if let Some(i) = self.pair_matches.iter().position(|(a1,b1,_)| *a1 == a && *b1 == b) {
            return &self.pair_matches[i].2;
        } else {
            let matches = self.do_matching(a,b);
            self.pair_matches.push((a,b,matches));
            return &self.pair_matches.last().unwrap().2;
        }
    }

    /* Returns alignment (calculates it if not previously done) */
    fn pair_alignment(&mut self, a: usize, b: usize) -> Option<&PairAlignment<T>> {
        if a == b { return None; }
        /* Check if it's already calculated previously */
        match self.edges.iter().position(|(a1,b1,_)| *a1 == a && *b1 == b) {
            None => {
                println!("Aligning images {} and {}", a, b);
                let alignment = PairAlignment::new(self.pair_matches(a,b));
                self.edges.push((a,b,alignment));
                self.pair_alignment(a,b)
            },
            Some(i) => match &self.edges[i].2 {
                Some(alignment) => Some(alignment),
                None => None
            }
        }
    }

    /* Checks if image i is connected */
    fn is_image_connected(&mut self, i: usize) -> bool {
        self.edges.iter().any(|(a,b,al)| (*a == i || *b == i) && al.is_some())
    }

    /* Get connections with an image */
    fn get_image_connections(&self, i: usize) -> Vec<usize> {
        self.edges.iter()
            .filter_map(|(a,b,al)| al.as_ref().map(|_|(a,b)))
            .filter_map(|(a,b)| match (*a == i, *b == i) {
                (true, false) => Some(*b),
                (false, true) => Some(*a),
                _ => None,
            }).collect()
    }
}