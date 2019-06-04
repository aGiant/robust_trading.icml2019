# robust_trading
## ICML 2019 Workshop on AI in Finance
### Applications and Infrastructure for Multi-Agent Learning

This repo stores the code used to generate the results presented in the paper
"Robust Trading via Adversarial Reinforcement Learning". The main scripts are
stored in src, including the actual experiments scripts (bin), environment
implementations (env) and RL agent types (agents). These will remain largely
unchanged in the future and make use of the
[rsrl](https://github.com/tspooner/rsrl) and
[lfa](https://github.com/tspooner/lfa) crates available in Rust.

Due to rapid development, local copies of `rsrl`, `lfa` and `rstat` have been
included in the repo with changes diverging from the main repo. Most of these
have since been integrated into the frameworks but some work is still required.
Once this has been finished, they will be removed from the repo and standard
versioning will resume. _At this point further documentation of this repo will
be added for your convenience_.
