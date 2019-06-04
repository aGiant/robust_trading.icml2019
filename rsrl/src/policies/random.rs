use crate::{
    core::*,
    geometry::Space,
    policies::{FinitePolicy, Policy},
};
use rand::{
    distributions::{Distribution, Uniform},
    rngs::ThreadRng,
    thread_rng,
};

// TODO: Generalise the random policy to work on any `Space`. This won't be hard
// at all, just use T: Into<Space>. Just make sure that you add all the relevant
// From implementations for the different spaces in the `spaces` crate; i.e.
// From<usize> for Ordinal etc etc...

pub struct Random(usize, ThreadRng);

impl Random {
    pub fn new(n_actions: usize) -> Self { Random(n_actions, thread_rng()) }

    pub fn from_space<S: Space>(space: S) -> Self { Self::new(space.dim()) }
}

impl Algorithm for Random {}

impl<S> Policy<S> for Random {
    type Action = usize;

    fn sample(&mut self, _: &S) -> usize { Uniform::new(0, self.0).sample(&mut self.1) }

    fn probability(&mut self, _: &S, _: usize) -> f64 { 1.0 / self.0 as f64 }
}

impl<S> FinitePolicy<S> for Random {
    fn n_actions(&self) -> usize { self.0 }

    fn probabilities(&mut self, _: &S) -> Vector<f64> { vec![1.0 / self.0 as f64; self.0].into() }
}

#[cfg(test)]
mod tests {
    use super::{FinitePolicy, Policy, Random};
    use crate::geometry::Vector;

    #[test]
    fn test_sampling() {
        let mut p = Random::new(2);

        let qs = vec![1.0, 0.0];

        let mut n0: f64 = 0.0;
        let mut n1: f64 = 0.0;
        for _ in 0..10000 {
            match p.sample(&qs) {
                0 => n0 += 1.0,
                _ => n1 += 1.0,
            }
        }

        assert!((0.50 - n0 / 10000.0).abs() < 0.05);
        assert!((0.50 - n1 / 10000.0).abs() < 0.05);
    }

    #[test]
    fn test_probabilites() {
        let mut p = Random::new(4);

        assert!(p
            .probabilities(&[1.0, 0.0, 0.0, 1.0])
            .all_close(&Vector::from_vec(vec![0.25; 4]), 1e-6));

        let mut p = Random::new(5);

        assert!(p
            .probabilities(&[1.0, 0.0, 0.0, 0.0, 0.0])
            .all_close(&Vector::from_vec(vec![0.2; 5]), 1e-6));

        assert!(p
            .probabilities(&[0.0, 0.0, 0.0, 0.0, 1.0])
            .all_close(&Vector::from_vec(vec![0.2; 5]), 1e-6));
    }
}
