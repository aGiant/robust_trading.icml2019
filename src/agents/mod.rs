extern crate bincode;
extern crate rsrl;

pub mod training;

use self::bincode::{deserialize_from, serialize_into};
use rsrl::{
    control::actor_critic::TDAC,
    fa::{
        LFA,
        TransformedLFA,
        Composable,
        basis::fixed::{Polynomial, Constant},
        transforms::Softplus,
    },
    geometry::{continuous::Interval, product::LinearSpace},
    policies::{gaussian::{self, Gaussian}, Beta, IPP},
    prediction::td::TD,
};
use std::{
    fs::File,
    io::{BufReader, BufWriter},
};

pub type Basis = Polynomial;
pub type Critic = TD<LFA<
    lfa::composition::Stack<Basis, Constant>,
    lfa::eval::ScalarFunction
>>;

pub type RP = gaussian::Gaussian<
    gaussian::mean::Scalar<LFA<
        lfa::composition::Stack<Basis, Constant>,
        lfa::eval::ScalarFunction,
    >>,
    gaussian::stddev::Scalar<TransformedLFA<
        lfa::composition::Stack<Basis, Constant>,
        lfa::eval::ScalarFunction,
        Softplus,
    >>,
>;
pub type Spread = gaussian::Gaussian<
    gaussian::mean::Scalar<TransformedLFA<
        lfa::composition::Stack<Basis, Constant>,
        lfa::eval::ScalarFunction,
        Softplus,
    >>,
    gaussian::stddev::Scalar<TransformedLFA<
        lfa::composition::Stack<Basis, Constant>,
        lfa::eval::ScalarFunction,
        Softplus,
    >>,
>;
pub type Drift = Beta<
    TransformedLFA<
        lfa::composition::Stack<Basis, Constant>,
        lfa::eval::ScalarFunction,
        Softplus,
    >,
>;

pub type Trader = TDAC<Critic, IPP<RP, Spread>>;
pub type Adversary = TDAC<Critic, Drift>;

// Trader:
pub fn build_trader(state_space: LinearSpace<Interval>, critic_lr: f64, policy_lr: f64) -> Trader {
    let basis = Basis::from_space(3, state_space).with_constant();
    let critic = Critic::new(LFA::scalar(basis.clone()), critic_lr, 1.0);
    let policy_rp = Gaussian::new(
        gaussian::mean::Scalar(LFA::scalar(basis.clone())),
        gaussian::stddev::Scalar(TransformedLFA::scalar(basis.clone(), Softplus)),
    );
    let policy_sp = Gaussian::new(
        gaussian::mean::Scalar(TransformedLFA::scalar(basis.clone(), Softplus)),
        gaussian::stddev::Scalar(TransformedLFA::scalar(basis.clone(), Softplus)),
    );
    let policy = IPP::new(policy_rp, policy_sp);

    Trader::new(
        critic,
        policy,
        policy_lr,
        1.0,
    )
}

/// Transform trader action
pub fn tta(a: (f64, f64)) -> [f64; 2] {
    [
        a.0 + a.1,
        a.1 - a.0
    ]
}

pub fn save_trader(agent: &Trader, path: String) {
    let mut writer = BufWriter::new(File::create(path).unwrap());

    serialize_into(&mut writer, &agent).ok();
}

pub fn load_trader(path: String) -> Trader {
    let reader = BufReader::new(File::open(path).unwrap());

    deserialize_from(reader).unwrap()
}

// Adversary:
pub fn build_adversary(state_space: LinearSpace<Interval>, critic_lr: f64, policy_lr: f64) -> Adversary {
    let basis = Basis::from_space(3, state_space).with_constant();
    let critic = Critic::new(LFA::scalar(basis.clone()), critic_lr, 1.0);
    let policy = Drift::new(
        TransformedLFA::scalar(basis.clone(), Softplus),
        TransformedLFA::scalar(basis, Softplus),
    );

    Adversary::new(
        critic,
        policy,
        policy_lr,
        1.0,
    )
}

pub fn save_adversary(agent: &Adversary, path: String) {
    let mut writer = BufWriter::new(File::create(path).unwrap());

    serialize_into(&mut writer, &agent).ok();
}

pub fn load_adversary(path: String) -> Adversary {
    let reader = BufReader::new(File::open(path).unwrap());

    deserialize_from(reader).unwrap()
}
