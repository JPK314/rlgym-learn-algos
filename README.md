# RLGym-Learn-algos
A set of standard implementations of common algorithms for use with [rlgym-learn](https://github.com/JPK314/rlgym-learn)

## Features
- PPO agent controller
- Flexible metrics logging
- File-based configuration

## Installation
1. install RLGym via `pip install rlgym`. If you're here for Rocket League, you can use `pip install rlgym[rl-rlviser]` instead to get the RLGym API as well as the Rocket League / Sim submodules and [rlviser](https://github.com/VirxEC/rlviser) support. 
2. If you would like to use a GPU install [PyTorch with CUDA](https://pytorch.org/get-started/locally/)
3. Install rlgym-learn via `pip install rlgym-learn`
3. Install this project via `pip install rlgym-learn-algos`
4. If pip installing fails at first, install Rust by following the instructions [here](https://rustup.rs/)

## Usage
See the [RLGym website](https://rlgym.org/RLGym%20Learn/introduction/) for complete documentation and demonstration of functionality [COMING SOON]. For now, you can take a look at `quick_start_guide.py` and `speed_test.py` in the rlgym-learn repo to get a sense of what's going on.

## Credits
The code in the util folder comes from rlgym-ppo, written by Matthew Allen.