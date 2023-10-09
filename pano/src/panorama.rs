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
pub type CameraModel<T> = GeneralCamera<T,PTLens<T>>;

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
    pub fn new(all_matches: &[Match<T>], max_ransac_iter: usize) -> Option<Self>
    {
        if all_matches.len() < 40 { return None; }
    
        /* Thresholds for initial RANSAC, loose and accurate */
        let (threshold1, threshold2) = (T::frac(1,40), T::frac(1,400));
    
        /******************************* RASNSAC *******************************/
        /* First pass of RANSAC with a large threshold.
         * Allows for a large number of RANSAC iterations, with a minimum of 50 */
        let (alignment, best_num_inliers) = ransac(
            /* TODO: decide how many ransac iterations to allow !! */
            max_ransac_iter, 50, Some(1.0e-8), 4, all_matches,
            |matches| {
                /* Start with pinhole mode */
                let mut model = ImagePair::<T,PinholeCamera<T>>::default().to_pars();
                model.image0.rotation_xyz = Some(Vector::zero()).to_pars();
                /* Unlock rotation parameters and one camera's focal length for first step */
                model.image0.rotation_xyz.unlock();
                model.image0.camera.focal_length.unlock();
                model = model.refine(matches, T::frac(5,10), 1)?;
                if (model.image0.camera.focal_length.value < T::zero()) != (model.image1.camera.focal_length.value < T::zero()) {return None;}
                /* Optimise both focal lengths now */
                model.image1.camera.focal_length.unlock();
                model = model.refine(matches, T::frac(5,10), 10)?;
                
                let (f0, f1) = (model.image0.camera.focal_length.value, model.image1.camera.focal_length.value);
                let (zero, fl_max, fl_min) = (T::zero(), T::frac(1500,36), T::frac(4,36));
                if (f0 < zero) != (f1 < zero) || (f0 < fl_min) || (f1 < fl_min) || (f0 > fl_max) || (f1 > fl_max) {
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
        if best_num_inliers < 25 { return None; }
        let alignment = alignment?;
    
        /* Final round of RANSAC, with a very small threshold and using more points per iteration (5 or 6, previously 8) */
        let inlier_set = alignment.from_pars().filter_outliers(all_matches, threshold1);
        let (alignment_extra_refined, _best_num_inliers) = ransac(
            500, 25, Some(0.0), 5, &inlier_set,
            |matches| alignment.refine(matches, T::frac(6,10), 10),
            |result| result.from_pars().count_inliers(all_matches, threshold2),
            |_,_| false
        );
        /* If the final optimisation worked, use it... */
        let alignment = alignment_extra_refined.unwrap_or(alignment);
        // println!("\n\nalignment = {:#.3?}\n\n", alignment.from_pars());

        /* Refine on the entire inlier set as the final step */
        let inlier_set = alignment.from_pars().filter_outliers(all_matches, threshold2);
        if inlier_set.len() < 10 { return None; }
        // alignment = alignment.refine(&inlier_set, T::frac(8,10), 10)?;
        println!("\n\nalignment = {:#.3?}\n\n", alignment.from_pars());

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
    pub fn bundle_adjust(&mut self, n_iterations: usize, learning_rate: T, adjust_distortion: bool) -> Result<(),()>
    {
        let mut param_id_vecs = vec![];
        let mut is_first = true;
        let mut as_params: Vec<Option<Image<Parameter<T>,CameraModel<Parameter<T>>>>> = self.images.iter().map(|im| {
            im.map(|mut im| {
                im.apply_rotation();
                if is_first { is_first = false }
                else { im.rotation_xyz = Some(Vector::zero()); }
                let mut as_pars = im.to_pars();
                /* Thge following parameters will be adjusted. TODO: allow user control. */
                if adjust_distortion {
                    as_pars.camera.radial_distortion.unlock();
                } else {
                    as_pars.camera.linearity.unlock();
                }
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
            else { break; }
        }

        for i in 0..self.images.len() {
            self.images[i] = as_params[i].map(|im| im.fmap(|par| par.value));
        }

        Ok(())
    }
}


#[derive(Debug,Clone)]
pub struct PanoramaImageSet<T,D> {
    /* Images (feature coordinates, corresponding descriptors) */
    pub images: Vec<(Vec<Point2D<T>>, Vec<D>)>,

    /* Unrefined pair feature matches (index_a, index_b, vector_of_matches) */
    pub pair_matches: Vec<(usize,usize,Vec<Match<T>>)>,

    /* Pair alignments, unsuccessful ones are stored as None (index_a, index_b, alignment) */
    pub edges: Vec<(usize,usize,Option<PairAlignment<T>>)>,

    /* Connected images (main panorama) */
    pub connected: Vec<usize>
}

impl<T: Float + Send + Sync, D: Descriptor + Sync> PanoramaImageSet<T,D>
{
    pub fn new() -> Self { Self {images: vec![], pair_matches: vec![], edges: vec![], connected: vec![]} }
    pub fn add_image(&mut self, keypoints: Vec<Point2D<T>>, descriptors: Vec<D>) { self.images.push((keypoints, descriptors)); }
    pub fn get_num_images(&self) -> usize { self.images.len() }

    /* Find which images are connected */
    fn find_connections(&self, key_image: usize) -> Vec<usize> {
        if self.images.len() == 0 { return vec![]; }
        fn dfs(edges: &[(usize, usize)], node: usize, connected: &mut [bool]) {
            connected[node] = true;
            for (a,b) in edges {
                if *a == node && !connected[*b] { dfs(edges, *b, connected); }
                if *b == node && !connected[*a] { dfs(edges, *a, connected); }
            }
        }
        let mut connected = vec![false; self.images.len()];
        connected[key_image] = true;
        let all_pairs: Vec<_> = self.edges.iter().filter_map(|(a,b,al)| al.as_ref().map(|_|(*a,*b))).collect();
        dfs(&all_pairs, key_image, &mut connected);
        /* Return indices of all connected */
        (0..connected.len()).zip(connected).filter_map(|(i,connected)| connected.then_some(i)).collect()
    }

    const MAX_RANSAC_ITER_IMPORTANT: usize = 10000;
    const MAX_RANSAC_ITER_UNIMPORTANT: usize = 250;

    /* Tries to find alignments pairs to connect all images,
     * assumes they are somewhat ordered */
    pub fn connect_initial_pairs(&mut self, key_image: usize)
    {
        let num_images = self.images.len();

        /* Try to connect all images in order */
        for i in 0..num_images {
            /* Try aligning a->b, then b->a if a->b failed (because feature matching is directional) */
            if self.pair_alignment(i,(i+1)%num_images, Self::MAX_RANSAC_ITER_IMPORTANT).is_none() {
                self.pair_alignment((i+1)%num_images,i, Self::MAX_RANSAC_ITER_IMPORTANT);
            }
        }

        let connected = self.find_connections(key_image);
        println!("Connections = {:?}", connected);

        /* Connect first and last image if its a 360 panorama */
        self.pair_alignment(connected[0], connected[connected.len()-1], Self::MAX_RANSAC_ITER_IMPORTANT);
        self.pair_alignment(connected[connected.len()-1], connected[0], Self::MAX_RANSAC_ITER_IMPORTANT);

        /* Match all images (slow) */
        for i in 0..num_images {
            for j in i..num_images { self.pair_alignment(i,j, Self::MAX_RANSAC_ITER_UNIMPORTANT); }
        }

        let connected = self.find_connections(key_image);
        println!("Connected images = {:?}", connected);
        println!("Connections:");
        self.edges.iter().for_each(|(a,b,al)| if let Some(al)=al{println!("{} -> {}: {} features", a,b,al.inlier_set.len());});

        self.connected = connected;
    }

    pub fn connect_overlapping_pairs(&mut self, _panorama: &mut PanoramaAlignment<T>) {
        /* Loop through all pairs that aren't connected and check how much they are overlapping */
        todo!()
    }

    /* Builds panorama based on current pairs */
    pub fn build_panorama(&mut self) -> Option<PanoramaAlignment<T>>
    {
        if self.images.len() < 2 { return None; }

        /* Take all current connections */
        let mut connections: Vec<_> = self.edges.iter().filter_map(|(a,b,al)| {
            if self.connected.contains(a) && self.connected.contains(b) { Some((*a,*b,al.clone()?)) }
            else { None }
        }).collect();

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
                    /* Both images are alredy in the panorama, so this set of points is only being
                     * added to improve alignment, not for the purpose of adding a new image */
                    panorama.edges.push((*a,*b,connection.inlier_set.clone()));
                    connections.remove(i);
                    break; /* DON'T Do bundle adjustment here it wastes time */
                }
                else if contains_a != contains_b {
                    /* Add image with approximate initial rotation using the pair alignment, then bundle adjust.
                     * This might be problematic when the focal length is esitmated wildly differntly by each pair for the same image */
                    if contains_a {
                        let mut image = connection.alignment.image1;
                        image.rotation_matrix = Some(connection.alignment.get_rotation() * panorama.images[*a].unwrap().get_rotation_matrix());
                        image.rotation_xyz = None;
                        image.camera.focal_length = image.camera.focal_length.abs(); //TODO: IMPROVE THIS
                        panorama.images[*b] = Some(image);
                    } else {
                        let mut image = connection.alignment.image0;
                        image.rotation_matrix = Some(connection.alignment.get_rotation().invert3x3() * panorama.images[*b].unwrap().get_rotation_matrix());
                        image.rotation_xyz = None;
                        image.camera.focal_length = image.camera.focal_length.abs();
                        panorama.images[*a] = Some(image);
                    }
                    panorama.edges.push((*a,*b,connection.inlier_set.clone()));
                    connections.remove(i);
                    panorama.bundle_adjust(10, T::frac(7,10), false).unwrap();
                    break;
                }
                // else if contains_a != contains_b {
                //     /* Only one of the images is already contained, add the new
                //      * image using the other image as its initial position */
                //     if contains_a {
                //         panorama.images[*b] = panorama.images[*a];
                //     } else {
                //         panorama.images[*a] = panorama.images[*b];
                //     }
                //     panorama.edges.push((*a,*b,connection.inlier_set.clone()));
                //     connections.remove(i);
                //     panorama.bundle_adjust(30, T::frac(6,10), false).unwrap();
                //     panorama.bundle_adjust(30, T::frac(9,10), false).unwrap();
                //     break;
                // }
                else { /* Cant be added to the panorama, not connected */ }
            }
        }

        /* Bundle adjustment, first without polynomial distortion parameters, then with */
        panorama.bundle_adjust(25, T::frac(6,10), false).unwrap();
        // panorama.bundle_adjust(25, T::frac(8,10), false).unwrap();
        Some(panorama)
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

    // /* Check if pair is connected */
    // fn is_pair_connected(&self, a: usize, b: usize) -> bool {
    //     self.edges.iter().any(|(a1,b1,alignment)| (*a1 == a && *b1 == b) || (*a1 == b && *b1 == a) && alignment.is_some())
    // }

    /* Returns alignment (calculates it if not previously done) */
    fn pair_alignment(&mut self, a: usize, b: usize, max_ransac_iter: usize) -> Option<&PairAlignment<T>> {
        if a == b { return None; }
        /* Check if it's already calculated previously */
        match self.edges.iter().position(|(a1,b1,_)| *a1 == a && *b1 == b) {
            None => {
                println!("Aligning images {} and {}", a, b);
                let alignment = PairAlignment::new(self.pair_matches(a,b), max_ransac_iter);
                self.edges.push((a,b,alignment));
                self.pair_alignment(a,b, max_ransac_iter)
            },
            Some(i) => match &self.edges[i].2 {
                Some(alignment) => Some(alignment),
                None => None
            }
        }
    }

    // /* Checks if image i is connected */
    // fn is_image_connected(&mut self, i: usize) -> bool {
    //     self.edges.iter().any(|(a,b,al)| (*a == i || *b == i) && al.is_some())
    // }

    // /* Get connections with an image */
    // fn get_image_connections(&self, i: usize) -> Vec<usize> {
    //     self.edges.iter()
    //         .filter_map(|(a,b,al)| al.as_ref().map(|_|(a,b)))
    //         .filter_map(|(a,b)| match (*a == i, *b == i) {
    //             (true, false) => Some(*b),
    //             (false, true) => Some(*a),
    //             _ => None,
    //         }).collect()
    // }
}