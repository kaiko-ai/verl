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
import asyncio
import heapq
import logging
import os
import random
from abc import ABC, abstractmethod
from typing import Any, Optional

import hydra
import numpy as np
import ray
import torch
from cachetools import LRUCache
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict
from tensordict import TensorDict
from transformers import AutoProcessor, AutoTokenizer

from verl.protocol import DataProto
from verl.single_controller.ray.base import RayWorkerGroup
from verl.trainer.ppo.reward import load_reward_manager
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.import_utils import import_external_libs
from verl.utils.model import compute_position_id_with_mask
from verl.utils.rollout_trace import (RolloutTraceConfig, rollout_trace_attr,
                                      rollout_trace_op)
from verl.workers.rollout.async_server import TokenOutput, async_server_class

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AsyncLLMServerManager:
    """
    A class to manage multiple OpenAI compatible LLM servers. This class provides
    - Load balance: least requests load balancing
    - Sticky session: send multi-turn chat completions to same server for automatic prefix caching
    """

    def __init__(self, config: DictConfig, server_handles: list[ray.actor.ActorHandle], max_cache_size: int = 10000):
        """Initialize the AsyncLLMServerManager.

        Args:
            config (DictConfig): YAML config.
            server_handles (List[ray.actor.ActorHandle]): OpenAI compatible LLM server actor handles.
            max_cache_size (int, optional): max cache size for request_id to server mapping. Defaults to 10000.
        """
        self.config = config
        self.server_handles = server_handles
        random.shuffle(self.server_handles)

        # Least requests load balancing
        self.weighted_serveres = [[0, (hash(server), server)] for server in server_handles]
        heapq.heapify(self.weighted_serveres)

        # LRU cache to map request_id to server
        self.request_id_to_server = LRUCache(maxsize=max_cache_size)

    def _choose_server(self, request_id: str) -> ray.actor.ActorHandle:
        # TODO: implement server pressure awareness load balancing
        if request_id in self.request_id_to_server:
            return self.request_id_to_server[request_id]

        server = self.weighted_serveres[0][1][1]
        self.weighted_serveres[0][0] += 1
        heapq.heapreplace(self.weighted_serveres, self.weighted_serveres[0])
        self.request_id_to_server[request_id] = server
        return server

    @rollout_trace_op
    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
    ) -> TokenOutput:
        """Generate tokens from prompt ids.

        Args:
            request_id (str): request id for sticky session.
            prompt_ids (List[int]): List of prompt token ids.
            sampling_params (Dict[str, Any]): Sampling parameters for the chat completion.

        Returns:
            TokenOutput: token output
        """
        server = self._choose_server(request_id)
        output = await server.generate.remote(
            request_id=request_id,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            image_data=image_data,
        )
        return output


class AgentLoopMetrics(BaseModel):
    """Agent loop performance metrics."""

    generate_sequences: float = 0.0
    tool_calls: float = 0.0


class AgentLoopOutput(BaseModel):
    """Agent loop output."""

    prompt_ids: list[int]
    """Prompt token ids."""
    response_ids: list[int]
    """Response token ids including LLM generated token, tool response token."""
    response_mask: list[int]
    """Response mask, 1 for LLM generated token, 0 for tool response token."""
    response_logprobs: Optional[list[float]] = None
    """Log probabilities for the response tokens."""
    multi_modal_data: Optional[dict[str, Any]] = None
    """Multi-modal data for multi-modal tools."""
    reward_score: Optional[float] = None
    """Reward score for the trajectory."""
    num_turns: int = 0
    """Number of chat turns, including user, assistant, tool."""
    metrics: AgentLoopMetrics
    """Auxiliary performance metrics"""
    extra_fields: dict[str, Any] = {}
    """Extra fields for dynamic addition."""


class _InternalAgentLoopOutput(AgentLoopOutput):
    """Internal agent loop output with padded sequences."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt_ids: torch.Tensor
    """Padded prompt token ids."""
    response_ids: torch.Tensor
    """Padded response token ids."""
    input_ids: torch.Tensor
    """Padded input ids(prompt_ids + response_ids)."""
    position_ids: torch.Tensor
    """Padded position ids."""
    response_mask: torch.Tensor
    """Padded response mask."""
    attention_mask: torch.Tensor
    """Padded attention mask."""
    response_logprobs: Optional[torch.Tensor] = None
    """Padded log probabilities for the response tokens."""
    multi_modal_inputs: Optional[dict[str, torch.Tensor]] = None
    """Multi-modal inputs for processors (e.g., pixel_values, image_grid_thw)."""
    extra_fields: dict[str, Any] = {}
    """Extra fields for dynamic addition."""


# make hydra.utils.instantiate happy
class _DummyConfig:
    def __init__(self, config: DictConfig) -> None:
        self.config = config


class AgentLoopBase(ABC):
    """An agent loop takes a input message, chat with OpenAI compatible LLM server and interact with various
    environments."""

    _class_initialized = False

    def __init__(
        self,
        trainer_config: _DummyConfig,
        server_manager: AsyncLLMServerManager,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        **kwargs,
    ):
        """Initialize agent loop, each sample will have its own loop instance.

        Args:
            trainer_config (_DummyConfig): trainer config.
            server_manager (AsyncLLMServerManager): OpenAI compatible LLM server manager.
            tokenizer (AutoTokenizer): Tokenizer for tokenize messages.
            processor (AutoProcessor): Processor for process messages.
        """
        self.init_class(config=trainer_config.config, tokenizer=tokenizer, processor=processor, **kwargs)
        self.config = trainer_config.config
        self.server_manager = server_manager
        self.tokenizer = tokenizer
        self.processor = processor
        self.loop = asyncio.get_running_loop()

    @classmethod
    def init_class(cls, config: DictConfig, tokenizer: AutoTokenizer, processor: AutoProcessor, **kwargs):
        """This is used to do heavy initialization work that should shared across all instances. It's only called once.

        Args:
            config (DictConfig): trainer config.
            tokenizer (AutoTokenizer): Tokenizer for tokenize messages.
            processor (AutoProcessor): Processor for process multi_modal data.
            **kwargs: extra kwargs from config file passed in by `hydra.utils.instantiate`.
        """
        if cls._class_initialized:
            return
        cls._class_initialized = True

    @abstractmethod
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Run agent loop to interact with LLM server and environment.

        Args:
            sampling_params (Dict[str, Any]): LLM sampling params.
            **kwargs: dataset fields from `verl.utils.dataset.RLHFDataset`.

        Returns:
            AgentLoopOutput: Agent loop output.
        """
        raise NotImplementedError


"""Agent loop registry: key is agent_name, value is a dict of agent loop config
used by hydra.utils.instantiate to initialize agent loop instance.

https://hydra.cc/docs/advanced/instantiate_objects/overview/
"""
_agent_loop_registry: dict[str, dict] = {}


def register(agent_name: str):
    """Register agent loop class."""

    def decorator(subclass: type[AgentLoopBase]) -> type[AgentLoopBase]:
        fqdn = f"{subclass.__module__}.{subclass.__qualname__}"
        _agent_loop_registry[agent_name] = {"_target_": fqdn}
        return subclass

    return decorator


@ray.remote(num_cpus=1)
class RewardManagerWorker:
    """Reward manager worker to compute reward score asynchronously to overlap with agent loop."""

    def __init__(self, config: DictConfig, local_path: str) -> None:
        tokenizer = hf_tokenizer(local_path, trust_remote_code=True)
        self.reward_manager = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        self.loop = asyncio.get_event_loop()

    async def compute_score(self, output: AgentLoopOutput, kwargs: dict) -> dict:
        """Compute reward score for agent loop output.

        NOTE: Since `reward_manager.__call__` is blocking function, we run it in thread pool to
        compute multiple samples in parallel.

        Args:
            output (AgentLoopOutput): Agent loop output.
            kwargs (dict): Dataset fields from `verl.utils.dataset.RLHFDataset`.

        Returns:
            dict: Reward score and reward extra info.
        """
        prompts = torch.tensor(output.prompt_ids, dtype=torch.long).unsqueeze(0)
        responses = torch.tensor(output.response_ids, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.ones((1, prompts.shape[1] + responses.shape[1]), dtype=torch.long)
        batch = TensorDict(
            {
                "prompts": prompts,  # [1, prompt_length]
                "responses": responses,  # [1, response_length]
                "attention_mask": attention_mask,  # [1, prompt_length + response_length]
            },
            batch_size=1,
        )
        non_tensor_batch = {
            **{k: np.array([v]) for k, v in kwargs.items()},
            "__num_turns__": np.array([output.num_turns]),
        }
        data = DataProto(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
        )
        result = await self.loop.run_in_executor(
            None,
            self.reward_manager,
            data,
            True,  # return_dict
        )

        reward_score = result["reward_tensor"].sum(dim=-1).item()
        reward_extra_info = {k: v[0] for k, v in result.get("reward_extra_info", {}).items()}
        return {"reward_score": reward_score, "reward_extra_info": reward_extra_info}

@ray.remote
class AgentLoopWorker:
    """Agent loop worker (vLLM-only inference).
    - Uses a single chat template across tokenizer/processor.
    - No multimodal RoPE/index recomputation on the client; vLLM handles it.
    - Keeps useful QoL: per-item fanout, optional logprobs, trace init, clean postprocess.
    """

    def __init__(self, config: DictConfig, server_handles: list[ray.actor.ActorHandle]):
        self.config = config
        self.server_manager = AsyncLLMServerManager(config, server_handles)

        model_path = config.actor_rollout_ref.model.path
        self.model_name = "/".join(model_path.split("/")[-2:])
        local_path = copy_to_local(config.actor_rollout_ref.model.path)

        ext_libs = config.actor_rollout_ref.model.get("external_lib", [])
        import_external_libs(ext_libs)

        # Hugging Face artifacts
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=True)
        self.processor = hf_processor(local_path, trust_remote_code=True)

        # Register agent loop configs
        agent_loop_config_path = config.actor_rollout_ref.rollout.agent.agent_loop_config_path
        if agent_loop_config_path:
            agent_loop_configs = OmegaConf.load(agent_loop_config_path)
            for agent_loop_config in agent_loop_configs:
                _agent_loop_registry[agent_loop_config.name] = agent_loop_config

        # Unify chat template across processor & tokenizer (critical for vLLM parity)
        custom_chat_template = self.config.actor_rollout_ref.model.get("custom_chat_template", None)
        if custom_chat_template is not None:
            if self.processor is not None:
                self.processor.chat_template = custom_chat_template
                # if getattr(self.processor, "tokenizer", None) is not None:
                    # self.processor.tokenizer.chat_template = custom_chat_template
            self.tokenizer.chat_template = custom_chat_template

        # Tracing
        trace_config = self.config.actor_rollout_ref.rollout.get("trace", {})
        RolloutTraceConfig.init(
            self.config.trainer.project_name,
            self.config.trainer.experiment_name,
            trace_config.get("backend"),
            trace_config.get("token2text", False),
        )

    async def generate_sequences(self, batch: DataProto) -> DataProto:
        """Generate sequences from agent loops, collate to DataProto."""
        cfg = self.config.actor_rollout_ref.rollout
        sampling_params = dict(
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            repetition_penalty=1.0,
            logprobs=cfg.calculate_log_probs,  # vLLM can return token logprobs
        )

        # Validation overrides
        if batch.meta_info.get("validate", False):
            sampling_params["top_p"] = cfg.val_kwargs.top_p
            sampling_params["temperature"] = cfg.val_kwargs.temperature

        # Ensure agent_name exists
        if "agent_name" not in batch.non_tensor_batch:
            batch.non_tensor_batch["agent_name"] = np.array(["single_turn_agent"] * len(batch), dtype=object)

        # Trajectory info
        if "index" in batch.non_tensor_batch:
            index = batch.non_tensor_batch["index"]
        else:
            index = np.arange(len(batch))
        trajectory_info = await get_trajectory_info(
            batch.meta_info.get("global_steps", -1), index.tolist(), batch.meta_info.get("validate", False)
        )

        # Fan out one task per sample (preserves per-item kwargs)
        tasks = []
        for i in range(len(batch)):
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
            tasks.append(asyncio.create_task(self._run_agent_loop(sampling_params, trajectory_info[i], **kwargs)))
        outputs = await asyncio.gather(*tasks)

        return self._postprocess(outputs)

    async def _run_agent_loop(
        self,
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
        *,
        agent_name: str,
        **kwargs,
    ):
        """Instantiate configured agent loop and run it. vLLM does all inference."""
        with rollout_trace_attr(
            step=trajectory["step"],
            sample_index=trajectory["sample_index"],
            rollout_n=trajectory["rollout_n"],
            validate=trajectory["validate"],
            name="agent_loop",
        ):
            assert agent_name in _agent_loop_registry, (
                f"Agent loop {agent_name} not registered, registered: {_agent_loop_registry.keys()}"
            )

            agent_loop_config = _agent_loop_registry[agent_name]
            agent_loop = hydra.utils.instantiate(
                config=agent_loop_config,
                trainer_config=_DummyConfig(config=self.config),
                server_manager=self.server_manager,
                tokenizer=self.tokenizer,
                processor=self.processor,
            )

            # Runs end-to-end (tool calls + vLLM generation); returns AgentLoopOutput
            output = await agent_loop.run(sampling_params, **kwargs)

            # Return raw AgentLoopOutput; padding is handled centrally in _postprocess
            return output

    def _postprocess(self, inputs: list) -> DataProto:
        """Pad/stack outputs. position_ids are simple 1-D cumsum (vLLM ignores them during gen)."""
        # prompts (left pad)
        self.tokenizer.padding_side = "left"
        outs = self.tokenizer.pad(
            [{"input_ids": x.prompt_ids} for x in inputs],
            padding="max_length",
            max_length=self.config.actor_rollout_ref.rollout.prompt_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        prompt_ids, prompt_mask = outs["input_ids"], outs["attention_mask"]

        # responses (right pad)
        self.tokenizer.padding_side = "right"
        outs = self.tokenizer.pad(
            [{"input_ids": x.response_ids} for x in inputs],
            padding="max_length",
            max_length=self.config.actor_rollout_ref.rollout.response_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        response_ids, response_attn_mask = outs["input_ids"], outs["attention_mask"]

        # response_mask (right pad): 1 for model tokens, 0 for tool/obs/pad
        outs = self.tokenizer.pad(
            [{"input_ids": x.response_mask} for x in inputs],
            padding="max_length",
            max_length=self.config.actor_rollout_ref.rollout.response_length,
            return_tensors="pt",
            return_attention_mask=False,
        )
        response_mask = outs["input_ids"]
        assert response_ids.shape == response_mask.shape, \
            f"mismatch in response_ids and response_mask: {response_ids.shape} vs {response_mask.shape}"
        response_mask = response_mask * response_attn_mask

        # concat
        input_ids = torch.cat([prompt_ids, response_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, response_attn_mask], dim=1)

        # simple 1-D position ids (for exports/analysis only)
        position_ids = (attention_mask.cumsum(dim=1) - 1) * attention_mask

        # optional rollout logprobs if any sample provided them
        optional = {}
        if inputs and getattr(inputs[0], "response_logprobs", None) is not None:
            # right-pad to response_length per-sample if not already padded
            padded_logs = []
            tgt_len = self.config.actor_rollout_ref.rollout.response_length
            for x in inputs:
                logs = x.response_logprobs or []
                pad = [0.0] * max(0, tgt_len - len(logs))
                padded_logs.append(torch.tensor(logs + pad).unsqueeze(0))
            optional["rollout_log_probs"] = torch.cat(padded_logs, dim=0)

        batch = TensorDict(
            {
                "prompts": prompt_ids,          # [bsz, prompt_length]
                "responses": response_ids,      # [bsz, response_length]
                "response_mask": response_mask, # [bsz, response_length]
                "input_ids": input_ids,         # [bsz, prompt_length + response_length]
                "attention_mask": attention_mask,
                "position_ids": position_ids,   # [bsz, prompt_length + response_length]
                **optional,
            },
            batch_size=len(inputs),
        )

        # reward scores mapped to last real response token per sample (if provided upstream)
        scores = [getattr(x, "reward_score", None) for x in inputs]
        if scores and all(s is not None for s in scores):
            prompt_len = prompt_ids.size(1)
            last_resp_idx = attention_mask[:, prompt_len:].sum(dim=1) - 1
            rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)
            rm_scores[torch.arange(response_mask.size(0)), last_resp_idx] = torch.tensor(scores, dtype=torch.float32)
            batch["rm_scores"] = rm_scores

        # non-tensor batch (turn counts, metrics, and any extra_fields)
        non_tensor = {
            "__num_turns__": np.array([getattr(x, "num_turns", 1) for x in inputs], dtype=np.int32),
        }
        metrics = [x.metrics.model_dump() for x in inputs]
        # flatten extra_fields into arrays
        all_extra_keys = set(k for x in inputs for k in getattr(x, "extra_fields", {}))
        for k in all_extra_keys:
            non_tensor[k] = np.array([x.extra_fields.get(k) for x in inputs], dtype=object)

        return DataProto(
            batch=batch,
            non_tensor_batch=non_tensor,
            meta_info={"metrics": metrics, "reward_extra_keys": []},
        )

    
async def get_trajectory_info(step, index, validate):
    """Get trajectory info.

    Args:
        step (int): global steps in the trainer.
        index (list): form datastore extra_info.index column.
        validate (bool): whether is a validate step.

    Returns:
        list: trajectory.
    """
    trajectory_info = []
    rollout_n = 0
    for i in range(len(index)):
        if i > 0 and index[i - 1] == index[i]:
            rollout_n += 1
        else:
            rollout_n = 0
        trajectory_info.append({"step": step, "sample_index": index[i], "rollout_n": rollout_n, "validate": validate})
    return trajectory_info


class AgentLoopManager:
    """Agent loop manager that manages a group of agent loop workers."""

    def __init__(self, config: DictConfig, worker_group: RayWorkerGroup):
        """Initialize agent loop manager.

        Args:
            config (DictConfig): trainer config.
            worker_group (RayWorkerGroup): ActorRolloutRef worker group.
        """
        self.config = config
        self.worker_group = worker_group

        self._initialize_llm_servers()
        self._init_agent_loop_workers()

        # Initially we're in sleep mode.
        self.sleep()

    def _initialize_llm_servers(self):
        self.rollout_tp_size = self.config.actor_rollout_ref.rollout.tensor_model_parallel_size
        self.rollout_dp_size = self.worker_group.world_size // self.rollout_tp_size

        workers_info = ray.get(
            [
                worker.__ray_call__.remote(lambda self: ray.get_runtime_context().get_node_id())
                for worker in self.worker_group.workers
            ]
        )
        self.rollout_dp_size = self.worker_group.world_size // self.rollout_tp_size
        # Store the node IDs for the servers
        self.server_node_ids = [workers_info[i * self.rollout_tp_size] for i in range(self.rollout_dp_size)]
        assert len(workers_info) == self.worker_group.world_size

        self.async_llm_servers = [None] * self.rollout_dp_size
        self.server_addresses = [None] * self.rollout_dp_size

        if self.config.actor_rollout_ref.rollout.agent.custom_async_server:
            server_class = async_server_class(
                rollout_backend=self.config.actor_rollout_ref.rollout.name,
                rollout_backend_module=self.config.actor_rollout_ref.rollout.agent.custom_async_server.path,
                rollout_backend_class=self.config.actor_rollout_ref.rollout.agent.custom_async_server.name,
            )
        else:
            server_class = async_server_class(rollout_backend=self.config.actor_rollout_ref.rollout.name)

        # Start all server instances, restart if address already in use.
        unready_dp_ranks = set(range(self.rollout_dp_size))
        while len(unready_dp_ranks) > 0:
            servers = {
                rollout_dp_rank: server_class.options(
                    # make sure AsyncvLLMServer colocates with its corresponding workers
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=workers_info[rollout_dp_rank * self.rollout_tp_size],
                        soft=False,
                    ),
                    name=f"async_llm_server_{rollout_dp_rank}",
                ).remote(self.config, self.rollout_dp_size, rollout_dp_rank, self.worker_group.name_prefix)
                for rollout_dp_rank in unready_dp_ranks
            }

            for rollout_dp_rank, server in servers.items():
                try:
                    address = ray.get(server.get_server_address.remote())
                    self.server_addresses[rollout_dp_rank] = address
                    self.async_llm_servers[rollout_dp_rank] = server
                    unready_dp_ranks.remove(rollout_dp_rank)
                except Exception:
                    ray.kill(server)
                    print(f"rollout server {rollout_dp_rank} failed, maybe address already in use, restarting...")

        # All server instances are ready, init AsyncLLM engine.
        ray.get([server.init_engine.remote() for server in self.async_llm_servers])

    def _init_agent_loop_workers(self):
        self.agent_loop_workers = []
        num_workers = self.config.actor_rollout_ref.rollout.agent.num_workers
        num_server_nodes = len(self.server_node_ids)
        for i in range(num_workers):
            # Round-robin scheduling over the all nodes
            node_id = self.server_node_ids[i % num_server_nodes]
            self.agent_loop_workers.append(
                AgentLoopWorker.options(
                    name=f"agent_loop_worker_{i}",
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_id, soft=True
                    ),
                ).remote(self.config, self.async_llm_servers)
            )

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Split input batch and dispatch to agent loop workers.

        Args:
            prompts (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
        """
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.wake_up()
        chunkes = prompts.chunk(len(self.agent_loop_workers))
        outputs = ray.get(
            [
                worker.generate_sequences.remote(chunk)
                for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=True)
            ]
        )
        output = DataProto.concat(outputs)
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.sleep()

        # calculate performance metrics
        metrics = [output.meta_info.pop("metrics") for output in outputs]  # List[List[Dict[str, str]]]
        timing = self._performance_metrics(metrics, output)

        output.meta_info = {"timing": timing, **outputs[0].meta_info}
        return output

    def _performance_metrics(self, metrics: list[list[dict[str, str]]], output: DataProto) -> dict[str, float]:
        timing = {}
        t_generate_sequences = np.array([metric["generate_sequences"] for chunk in metrics for metric in chunk])
        t_tool_calls = np.array([metric["tool_calls"] for chunk in metrics for metric in chunk])
        timing["agent_loop/generate_sequences/min"] = t_generate_sequences.min()
        timing["agent_loop/generate_sequences/max"] = t_generate_sequences.max()
        timing["agent_loop/generate_sequences/mean"] = t_generate_sequences.mean()
        timing["agent_loop/tool_calls/min"] = t_tool_calls.min()
        timing["agent_loop/tool_calls/max"] = t_tool_calls.max()
        timing["agent_loop/tool_calls/mean"] = t_tool_calls.mean()

        # batch sequence generation is bounded by the slowest sample
        slowest = np.argmax(t_generate_sequences + t_tool_calls)
        attention_mask = output.batch["attention_mask"][slowest]
        prompt_length = output.batch["prompts"].shape[1]
        timing["agent_loop/slowest/generate_sequences"] = t_generate_sequences[slowest]
        timing["agent_loop/slowest/tool_calls"] = t_tool_calls[slowest]
        timing["agent_loop/slowest/prompt_length"] = attention_mask[:prompt_length].sum().item()
        timing["agent_loop/slowest/response_length"] = attention_mask[prompt_length:].sum().item()

        return timing

    def wake_up(self):
        """Wake up all rollout server instances."""
        ray.get([server.wake_up.remote() for server in self.async_llm_servers])

    def sleep(self):
        """Sleep all rollout server instances."""
        ray.get([server.sleep.remote() for server in self.async_llm_servers])
