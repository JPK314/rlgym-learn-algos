# Changelog

All notable changes to this project will be documented in this file starting with version 0.3.0.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- added uv.lock file for python dependency management
- rlgym_learn_algos.logging.wandb imports will throw a ModuleNotFoundError if wandb is not installed

### Changed

- Config models containing generic other config models are now, as a pattern, validated inside the outer config model's before validation. This affects AgentController, ExperienceBuffer, TrajectoryProcessor, and MetricsLogger instances.
  - In order to support this, the `validate_config` method has been removed in favor of a `config_model` property.
- Moved Python code (the `rlgym_learn_algos` folder) to inside the `python` folder
- Moved rust-side module generation from `rlgym_learn_algos.rlgym_learn_algos` to `rlgym_learn_algos._rlgym_learn_algos` and modified internal module structure as well as re-exporting to main module (see below)
  - The rust `DerivedGAETrajectoryProcessorConfig` and `GAETrajectoryProcessor` classes are now defined in a submodule `rlgym_learn_algos._rlgym_learn_algos.ppo` and are re-exported in `rlgym_learn_algos.ppo` as `RustDerivedGAETrajectoryProcessorConfig` and `RustGAETrajectoryProcessor` respectively
- Logging subpackage has had wandb moved to a nested subpackage to make the additional dependency clear
- ExperienceBufferConfigModel's device field now defaults to "cpu" instead of automatically choosing cuda:0 if cuda is available
- PPOLearnerConfigModel's device field now defaults to "cpu" instead of automatically choosing cuda:0 if cuda is available
- The PPOAgentControllerConfigModel's `add_unix_timestamp` has been replaced with `run_suffix` which is an arbitrary factory function for a string to append to the end of the run name, for use in the checkpoint save folder and (if using wandb) in the wandb run.
- The DerivedMetricsLoggerConfig dataclass now stores the entire derived agent controller config in `derived_agent_controller_config`, and no longer stores additional derived config or the agent controller name.
  - The intended path for metrics loggers to derive configuration from agent controllers is now to have the metrics logger's constructor take a function handle that takes the `DerivedAgentControllerConfig` and returns the whatever additional derived config is needed/defined for that metrics logger type.
- The `WandbMetricsLogger` constructor now takes an additional method `additional_derived_config_factory` to map the `DerivedAgentControllerConfig` to `WandbAdditionalDerivedConfig`. An implementation of this method is exported by the wandb subpackage for the `PPOAgentController` class, called `ppo_additional_derived_config_factory`.
- `advantage_normalization` in the `PPOLearnerConfigModel` has been renamed to `advantage_standardization` to better reflect standard terminology.
- Generic type variables for config model classes no longer have Optional in the type bound. Instead Optional is placed on the type in the derived config dataclass.
- Fixed issue where PPOLearnerConfigModel and ExperienceBufferConfigModel default device "cpu" would stay as string after construction due to missing validate_default config

### Removed

- The PPOAgentController no longer hard codes the derivation of the additional derived config when the metrics logger is an instance of WandbMetricsLogger. See above for info on metrics logger config changes for more information.
