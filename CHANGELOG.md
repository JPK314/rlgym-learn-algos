# Changelog

All notable changes to this project will be documented in this file starting with version 0.3.0.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.1] - 2026-09-03

### Changed

- `PPOAgentController` and `PPOLearner` now use `non_blocking=True` only when the learner's device is not CPU in order to avoid strange asynchronous behavior when using CPU-only on Apple silicon.
- The field `log_prefix` in `PPOLearner` is now private (renamed to `_log_prefix`)

## [0.4.0] - 2026-08-29

### Added

- `MultiAgentController` and `MultiAgentSubcontroller` classes have been added in the `agent_controller` submodule to replace the multiple agent controller abstraction that used to be in rlgym-learn, with some enhancements. Notably:
  - The `choose_agents` and `get_action` methods have been updated to include information about which environment each agent id / obs is coming from.
  - The `choose_agents` method can now optionally return `None`, indicating that the agent subcontroller wants to submit actions for every agent_id in every environment it can.
  - The `PPOAgentController` class now inherits from `MultiAgentSubcontroller`, meaning it can be used interchangeably as either an `AgentController` or a `MultiAgentSubcontroller`.
  - Management of choosing env actions and assigning action choices for agents in environments performing a step env action to agent subcontrollers is now centralized in an overridable method in `MultiAgentController`. The default implementation calls reset if all agents in the environment are currently either terminated or truncated, and calls step otherwise, with no request to send state and no shared info setting. Note that the PPOAgentController now relies on this default implementation when used as a `MultiAgentSubcontroller`, which could be breaking if the `RLGym` environment is capable of having agents un-terminate or un-truncate after a previous step where they returned as terminated or truncated.
- A new abstraction `ActorCritic` has been created that combines the functionality of `Actor` and `Critic` to allow for shared parameters between the two. The `SeparateActorCritic` implementation wraps `Actor` and `Critic` instances into an `ActorCritic` instance.
- Added new config field `max_grad_norm` to `PPOLearnerConfigModel` which is used to clip the gradient norm when updating the actor and critic.
- Added new config parameter `reward_clip` instead of hard-coding a clip range of -10 to 10 for rewards during standardization. This parameter can be used independently of
- `obs_space` and `action_space` have been added to `DerivedExperienceBufferConfig`.
- Added `NumpyCircularBuffer` and `TensorCircularBuffer` abstractions for experience buffers.

### Changed

- Moved Python code (the `rlgym_learn_algos` folder) to inside the `python` folder
- Moved rust-side module generation from `rlgym_learn_algos.rlgym_learn_algos` to `rlgym_learn_algos._rlgym_learn_algos` and modified internal module structure as well as re-exporting to main module (see below)
  - The rust `DerivedGAETrajectoryProcessorConfig` and `GAETrajectoryProcessor` classes are now defined in a submodule `rlgym_learn_algos._rlgym_learn_algos.ppo` and are re-exported in `rlgym_learn_algos.ppo` as `RustDerivedGAETrajectoryProcessorConfig` and `RustGAETrajectoryProcessor` respectively
- Fixed issue where PPOLearnerConfigModel and ExperienceBufferConfigModel default device "cpu" would stay as string after construction due to missing validate_default config
- Update for rlgym-learn 2.0.0a1
  - The parameter `agent_choice_fn` in the constructor for `PPOAgentController` has had its type updated to reflect how rlgym-learn now supports making choices based on env id as well as agent id, and now defaults to None (meaning all are used).
  - Log probs are now stored in the `PPOAgentController` and are managed in the `get_actions` method.
  - The `get_env_actions` method for `PPOAgentController` now explicitly returns 0 processes to add on every invocation to comply with the new method signature
  - The `set_space_types` method now enforces that only one distinct value for `ActionSpaceType` and `ObsSpaceType` are returned across all environments now that all environments return their corresponding space types.
- Choices for AgentIDs in the `PPOAgentController` now only happen once per trajectory instead of every timestep
- The `PPOAgentController` now enforces that `recalculate_agent_id_every_step` is false in `ProcessConfig` as it breaks how the controller builds up trajectories
- `EnvTrajectories` has had the `agent_ids` parameter renamed to `env_agent_ids`, and the `agent_choice_fn` has been removed.
- `EnvTrajectories` now takes an additional parameter `controlled_agents` in the `add_steps` method to filter out any agent ids that had their actions chosen by another agent controller.
- In `PPOAgentController`, `natural_episode_length_mean`, `natural_episode_length_median`, and `percent_truncated` are no longer calculated due to ambiguity in what data to include or exclude from the statistics.
- `TrajectoryProcessor`, `Actor`, `Critic`, `MetricsLogger`, `DictMetricsLogger`, `BatchRewardTypeNumpyConverter`, and `ObsStandardizer` now are abstract base classes to properly force implementation of abstract methods for type checkers.
- `ExperienceBufferConfigModel`'s `device` field is now properly used as the device on which the experience buffer's tensor data is saved - if this is different from the learner device, the data will be moved to the learner's device as part of `get_all_batches_shuffled`.
- The checkpoint format for the `ExperienceBuffer` has changed, and the file extension has changed from `.pkl` to `.zip`. Starting a run with a checkpoint from the previous format will cause the experience buffer to initialize without any data in it (which is usually fine anyway).
- The `Actor` and `Critic` abstractions and implementations have been moved from the `ppo` submodule to the `ppo.actor_critic` submodule.
- Renamed `acts` parameter in `Actor` class's `get_backprop_data` method to `action_list` for naming consistency.
- `PPOAgentController` and `PPOLearner` now take an `actor_critic_factory` which returns an `ActorCritic` instance instead of two separate factories for `Actor` and `Critic`.
- `PPOLearner` now takes a dict `optimizer_named_parameter_group_kwargs` to manage learning rate and other optimizer parameters instead of having a simple `actor_lr` and `critic_lr`.
- `PPOAgentController` and `PPOLearner` now take an `optimizers_factory` which returns a `list[Optimizer]` using the `ActorCritic` returned by the `actor_critic_factory` in order to allow for custom optimizers.
- Logging of parameter counts has been moved to be the responsibility of the `actor_critic_factory` implementation. Logging of optimizer parameters (such as learning rate) has been moved to be the responsibility of the `optimizers_factory` implementation. There is a helper function `log_actor_critic_parameter_counts` to perform the logging for a separated `Actor` and `Critic`.
- Renamed `get_action` to `get_actions` in `Actor` and its subclasses.
- Renamed the field `controller_name` to `agent_controller_name` in `DerivedMetricsLoggerConfig` for consistency.
- Actor and Critic subclasses have added "dtype" kwarg to customize the dtype used for the model.
- Renamed field `standardize_returns` to `standardize_rewards` in `GAETrajectoryProcessorConfigModel` for clarity and fixed implementation to better reflect intent of standardization.
  - The original implementation of PPO used by OpenAI used a wrapper around the environment that computed something close to the standard deviation of returns and updated the reward in place. This implementation calculates the actual returns using unstandardized, unclipped rewards across all trajectories being added to the experience buffer and then standardizes all rewards by dividing by the standard deviation of these unstandardized, unclipped returns. This should maximize stability and accuracy.
  - When `standardize_rewards` is true, `max_returns_per_stats_increment` now uses a random sample (without replacement) of the unstandardized, unclipped returns to update the running stat that stores the standard deviation.
  - `standardize_returns` being set to true no longer hard codes a clip on all rewards to `[-10, 10]`. Instead a separate optional field `reward_clip` has been added (see Added section above).
- `max_returns_per_stats_increment` is now optional, and when set to None, all unstandardized, unclipped returns are used instead of just a random sample. Defaults to None (previously 150).
- Updated `Actor`, `Critic`, and `ActorCritic` abstract classes and implementations to optionally take a `Tensor` instead of a `list[ObsType]` or `list[ActionType]` where appropriate.
- `ContinuousActor` has been reworked entirely. It now uses tanh squishing and an affine transform to allow for arbitrary finite ranges in each output dimension, and no longer restricts the variance of the gaussian.
- The inner metrics logger checkpoint for `WandbMetricsLogger` is now saved in a folder `inner_metrics_logger` inside the folder for the `WandbMetricsLogger` checkpoint itself.
- Switched `ExperienceBuffer` and `NumpyExperienceBuffer` to use `TensorCircularBuffer` and `NumpyCircularBuffer` where appropriate.

### Removed

- `MultiDiscreteFF` was removed because it was not generically implemented and there is currently no demand for a generic implementation.
- `learner_device` has been removed from the `DerivedExperienceBufferConfig` dataclass.
- `torch_functions` and the `MapContinuousAction` class have been removed due to the `ContinuousActor` rework.

## [0.3.0] - 2026-06-03

### Added

- added uv.lock file for python dependency management
- rlgym_learn_algos.logging.wandb imports will throw a ModuleNotFoundError if wandb is not installed

### Changed

- Config models containing generic other config models are now, as a pattern, validated inside the outer config model's before validation. This affects AgentController, ExperienceBuffer, TrajectoryProcessor, and MetricsLogger instances.
  - In order to support this, the `validate_config` method has been removed in favor of a `config_model` property.
- Logging subpackage has had wandb moved to a nested subpackage to make the additional dependency clear
- ExperienceBufferConfigModel's device field now defaults to "cpu" instead of automatically choosing cuda:0 if cuda is available
- PPOLearnerConfigModel's device field now defaults to "cpu" instead of automatically choosing cuda:0 if cuda is available
- The PPOAgentControllerConfigModel's `add_unix_timestamp` has been replaced with `run_suffix` which is an arbitrary factory function for a string to append to the end of the run name, for use in the checkpoint save folder and (if using wandb) in the wandb run.
- The DerivedMetricsLoggerConfig dataclass now stores the entire derived agent controller config in `derived_agent_controller_config`, and no longer stores additional derived config or the agent controller name.
  - The intended path for metrics loggers to derive configuration from agent controllers is now to have the metrics logger's constructor take a function handle that takes the `DerivedAgentControllerConfig` and returns the whatever additional derived config is needed/defined for that metrics logger type.
- The `WandbMetricsLogger` constructor now takes an additional method `additional_derived_config_factory` to map the `DerivedAgentControllerConfig` to `WandbAdditionalDerivedConfig`. An implementation of this method is exported by the wandb subpackage for the `PPOAgentController` class, called `ppo_additional_derived_config_factory`.
- `advantage_normalization` in the `PPOLearnerConfigModel` has been renamed to `advantage_standardization` to better reflect standard terminology.
- Generic type variables for config model classes no longer have Optional in the type bound. Instead Optional is placed on the type in the derived config dataclass.

### Removed

- The PPOAgentController no longer hard codes the derivation of the additional derived config when the metrics logger is an instance of WandbMetricsLogger. See above for info on metrics logger config changes for more information.

## [0.2.6] - 2026-01-24
