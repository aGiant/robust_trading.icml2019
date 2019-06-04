use crate::core::*;
use crate::domains::Transition;
use crate::fa::{Parameterised, Approximator, VFunction};
use crate::geometry::{Matrix, MatrixView, MatrixViewMut};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExponentialTD<S, V> {
    pub v_func: V,
    pub start_state: S,

    pub alpha: Parameter,
    pub rho: Parameter,
}

impl<S, V> ExponentialTD<S, V> {
    pub fn new<T1, T2>(v_func: V, start_state: S, alpha: T1, rho: T2) -> Self
    where
        T1: Into<Parameter>,
        T2: Into<Parameter>,
    {
        ExponentialTD {
            v_func,
            start_state,

            alpha: alpha.into(),
            rho: rho.into(),
        }
    }
}

impl<S, V> Algorithm for ExponentialTD<S, V> {
    fn handle_terminal(&mut self) {
        self.alpha = self.alpha.step();
        self.rho = self.rho.step();
    }
}

impl<S, A, V: VFunction<S>> OnlineLearner<S, A> for ExponentialTD<S, V> {
    fn handle_transition(&mut self, t: &Transition<S, A>) {
        let phi_s = self.v_func.embed(t.from.state());

        let v_s = self.v_func.evaluate(&phi_s).unwrap();
        let v_ns = self.v_func.evaluate(&self.v_func.embed(t.to.state())).unwrap();
        let v_ss = self.v_func.evaluate(&self.v_func.embed(&self.start_state)).unwrap();

        let exp_rr = (self.rho.value() * t.reward).exp();
        let td_error = if v_ss.abs() < 1e-5 {
            exp_rr * v_ns - v_s
        } else {
            exp_rr / v_ss * v_ns - v_s
        };

        self.v_func.update(&phi_s, self.alpha * td_error).ok();
    }
}

impl<S, V: VFunction<S>> ValuePredictor<S> for ExponentialTD<S, V> {
    fn predict_v(&mut self, s: &S) -> f64 {
        self.v_func.evaluate(&self.v_func.embed(s)).unwrap()
    }
}

impl<S, A, V: VFunction<S>> ActionValuePredictor<S, A> for ExponentialTD<S, V> {}

impl<S, V: Parameterised> Parameterised for ExponentialTD<S, V> {
    fn weights(&self) -> Matrix<f64> {
        self.v_func.weights()
    }

    fn weights_view(&self) -> MatrixView<f64> {
        self.v_func.weights_view()
    }

    fn weights_view_mut(&mut self) -> MatrixViewMut<f64> {
        self.v_func.weights_view_mut()
    }
}
