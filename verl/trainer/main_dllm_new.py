# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
统一多轮 Agentic RL 入口，支持离散扩散语言模型（dLLM），如 SDAR、LLaDA。
rollout 由 SGLang dLLM 服务器完成（rollout.name=sglang）。

核心组件
  - Trainer : ``verl.trainer.ppo.ray_trainer_dllm_new.RayPPOTrainer``
  - Worker  : ``verl.workers.dllm_fsdp_workers.DLLMActorRolloutRefWorker``

Worker 根据 ``actor_rollout_ref.model.name`` 和 ``dllm_config.actor_type``
自动选择对应的 Actor 实现和 Rollout 后端，无需在此入口硬编码。

用法
----
::

    python -m verl.trainer.main_dllm_new \\
        +actor_rollout_ref.model.name=sdar \\
        actor_rollout_ref.rollout.name=sglang \\
        +actor_rollout_ref.dllm_config.actor_type=rl \\
        ...
"""

import os

import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.ppo.ray_trainer_dllm_new import RayPPOTrainer, ResourcePoolManager, Role
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config) -> None:
    # Check if Ray is not initialized
    if not ray.is_initialized():
        # Initialize Ray with a local cluster configuration
        # Set environment variables in the runtime environment to control tokenizer parallelism,
        # NCCL debug level, VLLM logging level, and allow runtime LoRA updating
        # `num_cpus` specifies the number of CPU cores Ray can use, obtained from the configuration
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        runtime_env_dict = OmegaConf.to_container(runtime_env, resolve=True)
        env_vars = runtime_env_dict.setdefault("env_vars", {})
        pythonpath = os.environ.get("PYTHONPATH", "")
        if pythonpath:
            env_vars["PYTHONPATH"] = pythonpath
        for _key in ("TORCHDYNAMO_DISABLE", "FLASHINFER_CACHE_DIR", "RAY_DEDUP_LOGS", "HYDRA_FULL_ERROR"):
            _val = os.environ.get(_key)
            if _val is not None:
                env_vars[_key] = _val
        runtime_env = OmegaConf.create(runtime_env_dict)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    runner = TaskRunner.remote()
    try:
        ray.get(runner.run.remote(config))
    except ray.exceptions.ActorUnavailableError as e:
        print(
            "\n[verl] TaskRunner actor died before returning (ActorUnavailableError). "
            "Common causes: (1) Ray raylet hit ulimit -n → epoll errno=24 / Too many open files; "
            "(2) OOM; (3) Python exception in actor. Check error.log for raylet lines. "
            "Try raising nofile in the job script: ulimit -n 1048576 (or RAY_NOFILE_LIMIT). "
            "Find actor traceback on nodes:\n"
            "  grep -rl Traceback /tmp/ray_*/logs/worker*.log 2>/dev/null | head -3 | xargs tail -80\n",
            flush=True,
        )
        raise e from None


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    def run(self, config):
        # print initial config
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        OmegaConf.resolve(config)

        # download the checkpoint from hdfs
        local_path = copy_to_local(config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False))

        from agent_system.environments import make_envs
        # When reward_manager=batch (e.g. MHQA), tools are loaded via multi_turn.tool_config_path
        # and no interactive environment is needed.
        reward_manager_name_early = config.reward_model.get("reward_manager", "episode")
        if reward_manager_name_early == "batch":
            envs, val_envs = None, None
        else:
            envs, val_envs = make_envs(config)

        # instantiate tokenizer
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        local_files_only = config.actor_rollout_ref.model.get("local_files_only", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code, local_files_only=local_files_only)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True, local_files_only=local_files_only)  # used for multimodal LLM, could be none

        # vllm early verify
        if config.actor_rollout_ref.rollout.name in ["vllm"]:
            from verl.utils.vllm_utils import is_version_ge

            if config.actor_rollout_ref.model.get("lora_rank", 0) > 0:
                if not is_version_ge(pkg="vllm", minver="0.7.3"):
                    raise NotImplementedError("PPO LoRA is not supported before vllm 0.7.3")

        # 选择 Worker 类
        # DLLMActorRolloutRefWorker 统一处理 sdar / llada / llada2 / dream：
        #   - _build_model_optimizer: 按 model.name 选 AutoModelForCausalLM / AutoModel
        #   - _build_rollout: 按 rollout.name 选 SGLangRollout / vLLMRollout
        #   - init_model: 按 dllm_config.actor_type（rl / distill）选对应的 DLLMDataParallelPPOActor
        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            assert config.critic.strategy in ["fsdp", "fsdp2"]
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.dllm_fsdp_workers import DLLMActorRolloutRefWorker
            from verl.workers.fsdp_workers import CriticWorker

            actor_rollout_cls = DLLMActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # we should adopt a multi-source reward function here
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # - finally, we combine all the rewards together
        # - The reward type depends on the tag of the data
        if config.reward_model.enable:
            if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # use reference model
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(actor_rollout_cls)
            mapping[Role.RefPolicy] = global_pool_id

        reward_manager_name = config.reward_model.get("reward_manager", "episode")
        if reward_manager_name == 'episode':
            from agent_system.reward_manager import EpisodeRewardManager
            reward_manager_cls = EpisodeRewardManager

            reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=0, normalize_by_length=False)

            # Note that we always use function-based RM for validation.
            # Allow overriding the number of examined batches via config to control logging verbosity.
            val_num_examine = config.reward_model.get("val_num_examine", 1)
            val_reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=val_num_examine, normalize_by_length=False)
        elif reward_manager_name == 'batch':
            # BatchRewardManager: used for non-interactive tasks like MHQA where reward is
            # computed from the final trajectory using a custom scoring function.
            from verl.trainer.ppo.reward import load_reward_manager
            reward_fn = load_reward_manager(config, tokenizer, num_examine=0,
                                            **config.reward_model.get("reward_kwargs", {}))
            val_num_examine = config.reward_model.get("val_num_examine", 1)
            val_reward_fn = load_reward_manager(config, tokenizer, num_examine=val_num_examine,
                                                **config.reward_model.get("reward_kwargs", {}))
        else:
            raise NotImplementedError(f"Unsupported reward_manager: {reward_manager_name}")

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        assert config.actor_rollout_ref.rollout.n == 1, "In verl, actor_rollout_ref.rollout.n>1 is for GRPO. In verl+env, we keep n=1, and achieve GRPO by env.rollout.n"

        from agent_system.multi_turn_rollout import TrajectoryCollector
        traj_collector = TrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor)

        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        train_sampler = create_rl_sampler(config.data, train_dataset)
        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
            traj_collector=traj_collector,
            envs=envs,
            val_envs=val_envs,
        )
        trainer.init_workers()
        trainer.fit()


def create_rl_dataset(data_paths, data_config, tokenizer, processor):
    """Create a dataset.

    Arguments:
        data_config: The data config.
        tokenizer (Tokenizer): The tokenizer.
        processor (Processor): The processor.

    Returns:
        dataset (Dataset): The dataset.
    """
    from torch.utils.data import Dataset

    from verl.utils.dataset.rl_dataset import RLHFDataset

    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        from verl.utils.import_utils import load_extern_type

        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
        if not issubclass(dataset_cls, Dataset):
            raise TypeError(f"The custom dataset class '{data_config.custom_cls.name}' from '{data_config.custom_cls.path}' must inherit from torch.utils.data.Dataset")
    else:
        dataset_cls = RLHFDataset
    print(f"Using dataset class: {dataset_cls.__name__}")

    dataset = dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
    )

    return dataset


def create_rl_sampler(data_config, dataset):
    """Create a sampler for the dataset.

    Arguments:
        data_config: The data config.
        dataset (Dataset): The dataset.

    Returns:
        sampler (Sampler): The sampler.
    """
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler

    # use sampler for better ckpt resume
    if data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(data_config.get("seed", 1))
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        sampler = SequentialSampler(data_source=dataset)

    return sampler


if __name__ == "__main__":
    main()
