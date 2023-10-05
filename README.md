# Panorama

Automatic feature-based panorama stitching implemented in Rust, focused on accurate image alignment and camera calibration. Capable of handling fisheye lenses and correcting distortion without any user input or usage of metadata.

OpenCV is used for its implementation of the SIFT feature detector/descriptor.

## Examples

### #1

360 degrees, 9 images, Samyang 12mm fisheye

Equirectangular (360) projection:

![portsmouth_360 copy](https://github.com/ilia3101/Panorama/assets/23642861/0921f505-09a5-41ff-a721-598e97669f11)

Stereographic projection:

<img src="https://github.com/ilia3101/Panorama/assets/23642861/65695597-690a-445b-9b05-f80c40db8dba" width="50%"/><img src="https://github.com/ilia3101/Panorama/assets/23642861/1bafeb1a-06fb-4901-8f4b-64497ab14560" width="50%"/>

### #2

8 images, iPhone 11 night mode

![spinakernight2 copy](https://github.com/ilia3101/Panorama/assets/23642861/b4441886-2b4e-41df-a19f-5740ff3edf16)

### #3

26 images, Sony RX1 (35mm)

![prague](https://github.com/ilia3101/Panorama/assets/23642861/9660feeb-d399-4f37-b4c2-26e07684bc9a)



## Running

```
cargo run -p pano --release --bin make_pano <image1> <image2> <...>
```
This runs the alignment process, outputting camera calibration details before entering an interactive shell with commands for adjusting the orientation and projection of the panorama alongside an OpenCV preview window.

It requires Rust/Cargo and OpenCV to be installed on your computer, the Homebrew OpenCV package should work on macs. Check [opencv-rust](https://github.com/twistedfall/opencv-rust) if you have any issues.

## Alignment process

Firstly, images are aligned in pairs, using RANSAC to exclude false feature matches. Secondly, the panorama is built up by adding images one by one, performing bundle adjustment optimisation at each step to improve accuracy.

A [generalised fisheye model](https://link.springer.com/article/10.1007/s11263-006-5168-1) with one parameter is utilised to compensate for radial distortion and fisheye projections during the RANSAC stage. Although the generalised fisheye model was intended only to model ideal rectilinear and fisheye lenses, I found that increasing the linearity (L) parameter above 1.0 approximates pincushion distortion. An additional three-parameter polynomial distortion model is layered on top of that to further improve accuracy when optimising the entire panorama.

## Technical details

All alignment procedures are implemented as iterative least squares solves, minimising reprojection errors. As this requires calculating derivatives for each parameter being optimised, it would have been difficult to implement any complex camera or distortion model by hand.

To make it feasible, [dual numbers](https://en.wikipedia.org/wiki/Dual_number) were used for automatic differentiation. A generic `Dual` type was created, wrapping any `Float` type, and implementing identical `Float` behaviour from the outside while performing derivative calculations internally, eliminating the need for manual differentiation. By using a vector as the derivative component of `Dual`, it is possible to differentiate multiple variables at once. I believe this is the same technique used by the Ceres solver with its `Jet` type.

The least squares solver is based around a (sort of) Higher Kinded Types pattern, where a structure is used to represent a set of parameters, for example:

```rust
struct Parameters<T> {
  parameter1: T,
  parameter2: T
}
```

A parameter struct must then implement the `Functor` trait (which allows all fields of a struct's inner type T to be converted to anything else using a common interface), and the `CalculateResiduals` trait. The solver, implemented as a supertrait over `CalculateResiduals`, works by converting the inner fields of a parameter struct to dual numbers using the `Functor` trait, then using the `CalculateResiduals` trait to perform differentiated residual calculations.

Additionaly, each parameter `T` in a struct can be further wrapped in a `Parameter<T>` type, allowing additional information to be included for the optimiser, such as a `lock` option, to indicate a parameter shouldn't be adjusted, and a numerical `id`, allowing multiple fields to be linked together and treated as one parameter.

For bundle adjustment, a sparse Jacobian matrix is represented as a list of blocks containing residuals and gradients from different residual-calculating structs, where parameters are linked together using id numbers. These are passed to a `sparse_step` function which calculates the appropriate step for each parameter, correctly linking parameters from different blocks using their ids.

Some details were omitted in the explanation to keep it short.

Implementing all of this with Rust's trait system was really fun!!!

## TODOs

- Multiband blending. Because image alignment was the main focus of the project, the final stitching/rendering stage only uses a simple weighted blending at the moment, gradually fading between each image over a large distance. Unfortunately this causes ghosting in images with moving objects or parallax.
- Calculation of exposure/vignetting parameters.
- Compute matches between not-yet-connected but overlapping images by checking for overlaps in the panorama (easy, high priority).
- Each image's camera parameters are optimised individually and not linked together, perhaps it would be nice to recognise when two or more images share the same camera/lens parameters and link them together.
- Use the Levenberg-Marquardt algorithm for optimisation, only a small change is needed to the current Gauss-Newton implementation (which has worked surprisingly well).
- Include a lens center shift/offset parameter.
- Refactor some things.
