/* A not very good functor trait, inspired by this comment:
 * https://www.reddit.com/r/rust/comments/vog2lc/comment/j2yzwcr/?context=3
 * When going A->B->A, rust doesn't know it's the same type based
 * on the trait alone, so I added a method which does't change the type. */
/* Also inspired by https://www.reddit.com/r/rust/comments/10bqmfs/comment/j4col9l/
 * and https://github.com/mtomassoli/HKTs, but those trait bounds still didn't help the compiler
 * realise that (A)->(B)->(A) == (A) */
/* Update: Just saw this exists also: https://github.com/bodil/higher */
pub trait Functor<A>: Sized
{
    type Wrapped<B> : Functor<B>
                    + Functor<B, Wrapped<A> = Self>
                    + Functor<B, Wrapped<B> = Self::Wrapped<B>>;

    fn fmap<F, B>(self, f: F) -> Self::Wrapped<B>
        where F: FnMut(A) -> B;

    /* Same as fmap, but when you don't change the inner type
     * (as the rust compiler is too dumb to realise that the
     * type doesn't change, despite the trait bounds I have
     * added to Wrapped above) */
    #[inline]
    fn tmap<F>(self, f: F) -> Self
      where F: FnMut(A) -> A {
        unsafe { core::mem::transmute_copy(&self.fmap(f)) }
    }
}

// pub trait BiFunctor<A,B>: Sized
// {
//     type Wrapped<C,D> : Functor<C,D>
//                       + Functor<C,D, Wrapped<A,B> = Self>
//                       + Functor<C,D, Wrapped<C,D> = Self::Wrapped<B>>;
//
//     fn fmap<F1,F2,C,D>(self, f1: F1, f2: F2) -> Self::Wrapped<C,D>
//         where F: FnMut(A) -> C;
//         where F: FnMut(B) -> D;
//
//     #[inline]
//     fn tmap<F2,F2>(self, f1: F1, f2: F1) -> Self
//       where F1: FnMut(A) -> A
//       where F2: FnMut(B) -> B {
//         unsafe { std::mem::transmute_copy(&self.fmap(f1,f2)) }
//     }
// }

/************** Some implementations of functor for common types **************/

use maths::linear_algebra::{Vector, Matrix};

impl<A> Functor<A> for () {
    type Wrapped<B> = ();
    #[inline] fn fmap<F, B>(self, _f: F) -> () {()}
}

impl<A> Functor<A> for Vec<A> {
    type Wrapped<B> = Vec<B>;
    #[inline] fn fmap<F, B>(self, f: F) -> Vec<B> where F: FnMut(A) -> B { self.into_iter().map(f).collect() }
}

impl<A, const N: usize> Functor<A> for Vector<A,N> {
    type Wrapped<B> = Vector<B,N>;
    #[inline] fn fmap<F, B>(self, f: F) -> Vector<B,N> where F: FnMut(A) -> B { self.map(f) }
}

impl<A, const N: usize> Functor<A> for [A;N] {
    type Wrapped<B> = [B;N];
    #[inline] fn fmap<F, B>(self, f: F) -> [B;N] where F: FnMut(A) -> B { self.map(f) }
}

impl<A, const N: usize, const M: usize> Functor<A> for Matrix<A,N,M> {
    type Wrapped<B> = Matrix<B,N,M>;
    #[inline] fn fmap<F, B>(self, f: F) -> Matrix<B,N,M> where F: FnMut(A) -> B { self.map(f) }
}

impl<T: Functor<A>, A> Functor<A> for Option<T> {
    type Wrapped<B> = Option<T::Wrapped<B>>;
    #[inline]
    fn fmap<F, B>(self, f: F) -> Self::Wrapped<B>
      where F: FnMut(A) -> B {
        self.map(|t| t.fmap(f))
    }
}
