use maths::linear_algebra::Vector;
use maths::traits::Float;


/*
 * CalculateResiduals trait. Implementing types should contain only parameters,
 * no partial evaluations, or cached values etc.
 * 
 * If running the function involves intermediate calculations dependent
 * on the parameters but not the input data, those calculations can be
 * done in the 'prepare' function, the output of which will
 * be passed to 'run' every time the function is run
 */
pub trait CalculateResiduals<T, const N_RESIDUALS: usize>
{
    /* Input data type */
    type Input;

    /* Pre calculate anything */
    type Context;
    fn prepare(&self) -> Self::Context;

    /* Run */
    fn run(&self, ctx: &Self::Context, input: Self::Input) -> [T; N_RESIDUALS];


    /* TODO: Move these elsewhere */
    #[inline]
    fn count_inliers(&self, inputs: &[Self::Input], threshold: T) -> usize
      where T: Float, Self::Input: Copy {
        let ctx = self.prepare();
        inputs.iter().filter(|&input| Vector(self.run(&ctx, *input)).magnitude() < threshold).count()
    }
    #[inline]
    fn filter_outliers(&self, inputs: &[Self::Input], threshold: T) -> Vec<Self::Input>
      where T: Float, Self::Input: Copy {
        let ctx = self.prepare();
        inputs.iter().filter(|&input| Vector(self.run(&ctx, *input)).magnitude() < threshold).map(|i| *i).collect()
    }
}
