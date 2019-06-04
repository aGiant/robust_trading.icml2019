extern crate algo_hft;
extern crate lfa;
extern crate bincode;
extern crate clap;
extern crate rand;
extern crate rsrl;
extern crate csv;
extern crate serde;
#[macro_use]
extern crate serde_derive;

use algo_hft::{
    agents::{Trader, tta},
    env::Env,
};
use bincode::deserialize_from;
use clap::{App, Arg};
use rsrl::{
    domains::Domain,
    policies::Policy,
};
use std::{
    fs::File,
    io::{BufReader, stdout},
};

#[derive(Serialize)]
struct Record {
    pub time: f64,

    pub midprice: f64,

    pub ask_price: f64,
    pub ask_executed: bool,

    pub bid_price: f64,
    pub bid_executed: bool,


    pub inventory: f64,
}

fn generate_sample(mut trader: Trader) {
    let mut file_logger = csv::Writer::from_writer(stdout());

    let mut domain = Env::default();
    let mut a = tta(trader.policy.mpa(domain.emit().state()));

    macro_rules! log {
        () => {
            file_logger.serialize(Record {
                time: domain.dynamics.time,

                midprice: domain.dynamics.price,

                ask_price: domain.dynamics.price + a[0],
                ask_executed: domain.ask_executed,

                bid_price: domain.dynamics.price - a[1],
                bid_executed: domain.bid_executed,

                inventory: domain.inv,
            }).ok();
        }
    }

    log!();

    loop {
        let t = domain.step(a);

        a = tta(trader.policy.mpa(domain.emit().state()));

        log!();

        if t.terminated() {
            break
        }
    }

    file_logger.flush().ok();
}

fn main() {
    let matches = App::new("AS inventory strategy simulator")
        .arg(Arg::with_name("bin_path")
                .index(1)
                .required(true))
        .get_matches();

    let reader = BufReader::new(File::open(matches.value_of("bin_path").unwrap()).unwrap());

    generate_sample(deserialize_from(reader).unwrap());
}
