use super::features::{Match, Descriptor, brute_force_match};
use super::alignment::{ransac::*, pair::*, image::*, camera::*};
use optimisation::{parameters::*, functor::*, optimise::*, traits::*};
use maths::{
    linear_algebra::{Point2D, Vector},
    traits::{Float, Zero}
};
use rayon::prelude::*;

/************************************************************************************************/
/**************************************** Configuration *****************************************/
/************************************************************************************************/

/* Global configuration */
pub type CameraModel<T> = GeneralCamera<T, PTLens<T>>;

/************************************************************************************************/
/**************************************** Configuration *****************************************/
/************************************************************************************************/


pub struct PanoramaImage<T,D> {
    /* Image resolution */
    pub resolution: (usize,usize),
    /* Detected features, coordinates in pixel coordinates */
    pub feature_coords: Vec<Point2D<T>>,
    pub descriptors: Vec<D>,
    // pub camera: Option<CameraModel<T>>,
    // pub orientation: Option<Matrix3x3<T>>,
}

pub struct PairConnection<T> {
    pub ransac_refined_points: Option<Vec<Match<T>>>,
    pub alignment: Option<PairAlignment<T>>,
}

#[derive(Debug, Clone)]
pub struct PairAlignment<T> {
    pub inlier_set: Vec<Match<T>>,
    pub alignment: ImagePair<T, CameraModel<T>>
}

impl<T: Float + Send + Sync> PairAlignment<T>
{
    /* Creates a pair alignment from an unrefined set of matches.
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
    pub fn new_ransac(all_matches: &[Match<T>]) -> Option<Self>
    {
        /* Thresholds for initial RANSAC, loose and accurate */
        let (threshold1, threshold2) = (T::frac(1,40), T::frac(1,800));
    
        /******************************* RASNSAC *******************************/
        /* First pass of RANSAC with a large threshold.
         * Allows for a large number of RANSAC iterations, with a minimum of 50 */
        let (alignment, mut best_num_inliers) = ransac(
            /* TODO: decide how many ransac iterations to allow, 25000 or 2500 (5 or 4 figures) */
            2500, 50, Some(1.0e-8), 4, all_matches,
            |matches| {
                /* Start in pinhole mode */
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
        /* Not enough inliers. TODO: handle this elsewhere ?? */
        if best_num_inliers < 20 { return None; }
        let mut alignment = alignment?;
    
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
        if inlier_set.len() < 50 { return None; }
        alignment = alignment.refine(&inlier_set, T::frac(9,10), 10)?;
        // println!("alignment = {:#?}", alignment.from_pars());

        let mut result = alignment.from_pars();
        result.image0.camera.focal_length = result.image0.camera.focal_length.abs();
        result.image1.camera.focal_length = result.image1.camera.focal_length.abs();
        Some(PairAlignment{alignment: result, inlier_set})
    }
}

pub struct PanoramaImageSet<T,D>
{
    /* Vector of images (order is preserved if entries are deleted) */
    pub images: Vec<Option<PanoramaImage<T,D>>>,

    /* All computed sets of feature matches (image_a, image_a, vector_of_matches) */
    pub feature_matches: Vec<(usize,usize,Vec<Match<T>>)>,

    /* Refined feature matches using RANSAC (including the resulting alignment, None if the pair was not successfully aligned) */
    pub pair_connections: Vec<(usize, usize, PairConnection<T>)>,

    /* Sub-panoramas (one index stored here per subpanorama) */
    pub subpanoramas: Vec<Vec<usize>>
}

impl<T,D> PanoramaImageSet<T,D>
{
    pub fn add_image(&mut self, image: PanoramaImage<T,D>) -> usize {
        let index = self.images.len();
        self.images.push(Some(image));
        self.subpanoramas.push(vec![index]);
        return index;
    }

    pub fn remove_image(&mut self, image: usize) {
        self.images.remove(image);
        /* Shift indices of all other records */
        for (mut image_a, mut image_b, _) in self.feature_matches.iter_mut() {
            if image_a > image { image_a -= 1; }
            if image_b > image { image_b -= 1; }
        }
        for (mut image_a, mut image_b, _) in self.pair_connections.iter_mut() {
            if image_a > image { image_a -= 1; }
            if image_b > image { image_b -= 1; }
        }
        for img in self.subpanoramas.iter_mut().flat_map(|x| x.iter_mut()) {
            if *img > image { *img -= 1; }
        }
    }

    pub fn try_connect(&mut self, image_a: usize, image_b: usize, max_ransac_iterations: usize) {
        // return;
    }

    fn get_matches(&mut self, image_a: usize, image_b: usize) -> Option<&Vec<Match<T>>> {
        if image_a != image_b {
            return None;
        } else {
            // for i in 0..
            return None;
        }
    }

    /* Tries to align some images */
    pub fn auto_align(&mut self) {
        return;
    }

    /* Refines panoramas (merges sub-panoramas, if pair connections allow) */
    pub fn refine_panoramas(&mut self) {
        return;
    }

    /* Returns a panorama. */
    pub fn get_panorama(&self, panorama: usize) {
        /* Finalise panorama (refine distortion parameters) and return all the alignments. */
    }
}