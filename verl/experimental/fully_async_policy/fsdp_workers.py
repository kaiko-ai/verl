# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright 2025 Meituan Ltd. and/or its affiliates
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

import logging
import os
import time

import torch
import torch.distributed
from omegaconf import DictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl.experimental.fully_async_policy.base_detach_sync import BaseDetachNcclSync
from verl.experimental.fully_async_policy.fsdp2_utils import fsdp2_sharded_load_from_cpu, fsdp2_sharded_save_to_cpu
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import (
    get_device_name,
    get_torch_device,
)
from verl.utils.fsdp_utils import (
    collect_lora_params,
    fsdp_version,
    load_fsdp_model_to_gpu,
    offload_fsdp_model_to_cpu,
)
from verl.utils.import_utils import import_external_libs
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.profiler import log_gpu_memory_usage
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker, CriticWorker
from verl.workers.rollout import get_rollout_class

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()

__all__ = ["DetachActorWorker", "DetachAsyncRolloutWorker", "CriticWorker"]


class DetachNcclSync(BaseDetachNcclSync, AsyncActorRolloutRefWorker):
    def __init__(self, config: DictConfig, role: str):
        BaseDetachNcclSync.__init__(self, config, role)
        AsyncActorRolloutRefWorker.__init__(self, config, role)

    def _get_actor_params(self):
        pass

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def sync_rollout_weights(self, sync_group_name="actor_rollout"):
        assert (self._is_actor or self._is_rollout) and not self.config.hybrid_engine
        assert hasattr(self, "_weights_info") and self._weights_info is not None

        if self._is_actor and self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        params = self._get_actor_params() if self._is_actor else None
        rollout_name = self.config.rollout.name

        inference_model = None
        # Determine inference model for direct weight loading (non-ServerAdapter rollouts).
        # ServerAdapter rollouts (both pure rollout and actor+rollout) use IPC-based sync instead.
        if self._is_rollout:
            if rollout_name == "vllm":
                from verl.workers.rollout.vllm_rollout.vllm_rollout import ServerAdapter as VllmServerAdapter

                if isinstance(self.rollout, VllmServerAdapter):
                    # ServerAdapter — use IPC-based weight sync (works for both pure rollout and actor+rollout)
                    inference_model = None
                elif not self._is_actor:
                    # Non-ServerAdapter pure rollout — load weights directly into inference engine
                    inference_model = BaseDetachNcclSync.get_inference_model(self.rollout)

                    from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader

                    patch_vllm_moe_model_weight_loader(inference_model)
            elif rollout_name == "sglang" and not self._is_actor:
                inference_model = self.rollout._engine
                # For ServerAdapter, _engine might be None and needs async initialization
                if inference_model is None:
                    # Initialize the server adapter engine
                    print("[sync_rollout_weights] Initialize server adapter engine")

                    async def init_engine():
                        if hasattr(self.rollout, "_init_server_adapter"):
                            await self.rollout._init_server_adapter()
                        else:
                            print("[sync_rollout_weights] No _init_server_adapter method found")
                        return self.rollout._engine

                    inference_model = self._run_async_safely(init_engine())
                    # For ServerAdapter, only TP rank 0 initializes the engine
                    # TP rank != 0 can safely have inference_model as None
                    from verl.workers.rollout.sglang_rollout.sglang_rollout import ServerAdapter

                    is_server_adapter = isinstance(self.rollout, ServerAdapter)
                    is_non_tp_rank = False
                    if (
                        is_server_adapter
                        and hasattr(self.rollout, "device_mesh")
                        and self.rollout.device_mesh is not None
                    ):
                        try:
                            is_non_tp_rank = self.rollout.device_mesh["infer_tp"].get_local_rank() != 0
                        except Exception:
                            pass

                    if inference_model is None and not (is_server_adapter and is_non_tp_rank):
                        raise RuntimeError(
                            f"Failed to initialize rollout engine. "
                            f"rollout type: {type(self.rollout)}, "
                            f"has _init_server_adapter: {hasattr(self.rollout, '_init_server_adapter')}"
                        )

        if rollout_name == "sglang" and self._is_rollout and not self._is_actor:
            self._sync_sglang_weights(inference_model, params, sync_group_name)
        elif rollout_name == "vllm" and self._is_rollout and inference_model is None:
            self._sync_vllm_weights_via_server_adapter(params, sync_group_name)
        else:
            self._sync_vllm_weights(inference_model, params, sync_group_name)

        if self._is_actor and self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        get_torch_device().empty_cache()

    def _sync_vllm_weights_via_server_adapter(self, params, sync_group_name):
        """Sync weights when rollout uses ServerAdapter (async disaggregated mode).

        Streams weights from NCCL broadcast directly to ServerAdapter.update_weights()
        via a thread-safe queue, avoiding accumulating all weights in GPU memory.
        Weights are cast to bf16 before IPC transfer since the vLLM model uses bf16
        and individual fp32 tensors (e.g. embed_tokens) can exceed the IPC bucket size.

        The IPC buffer and handle are pre-created in the main thread because
        reduce_tensor() (CUDA IPC handle creation) is not thread-safe — calling it
        repeatedly from the background event loop thread causes SIGSEGV in
        torch::GetNewRefCountedSentData after several syncs.
        """
        import asyncio
        import queue as queue_module

        from ray.util.collective import collective

        # Pre-create IPC buffer and handle in the main thread where CUDA IPC is safe.
        ipc_buffer = None
        ipc_handle = None
        if self._is_rollout and not self.rollout.use_shm:
            from torch.multiprocessing.reductions import reduce_tensor

            bucket_size = int(self.rollout.config.checkpoint_engine.update_weights_bucket_megabytes) << 20
            ipc_buffer = torch.empty(bucket_size, dtype=torch.uint8, device=f"{device_name}:0")
            ipc_handle = reduce_tensor(ipc_buffer)

        weight_queue = queue_module.Queue(maxsize=2)

        def streaming_weight_generator():
            while True:
                item = weight_queue.get()
                if item is None:
                    return
                yield item

        async def run_ipc_update():
            peft_config = getattr(self, "_peft_config", None)
            await self.rollout.update_weights(
                streaming_weight_generator(),
                ipc_buffer=ipc_buffer,
                ipc_handle=ipc_handle,
                peft_config=peft_config,
                base_sync_done=bool(peft_config),
            )

        if self._is_rollout:
            ipc_future = asyncio.run_coroutine_threadsafe(run_ipc_update(), self._bg_loop)

        for key, shape, dtype in self._weights_info:
            tensor = torch.empty(shape, dtype=dtype, device=get_torch_device().current_device())
            if self._is_actor:
                assert key in params
                origin_data = params[key]
                if hasattr(origin_data, "full_tensor"):
                    origin_data = origin_data.full_tensor()
                if torch.distributed.get_rank() == 0:
                    tensor.copy_(origin_data)
            collective.broadcast(tensor, src_rank=0, group_name=sync_group_name)
            if self._is_rollout:
                if tensor.dtype == torch.float32:
                    tensor = tensor.to(torch.bfloat16)
                weight_queue.put((key, tensor))

        if self._is_rollout:
            weight_queue.put(None)
            ipc_future.result()

    def cache_actor_weights_to_cpu(self):
        self.cpu_named_params = {}
        if self._is_actor:
            params = self._get_actor_params()
            local_rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()

            for tensor_idx, (key, _, _) in enumerate(self._weights_info):
                origin_data = params[key]
                if hasattr(origin_data, "full_tensor"):
                    origin_data = origin_data.full_tensor()

                if tensor_idx % world_size == local_rank:
                    self.cpu_named_params[key] = origin_data.to("cpu", non_blocking=True)
            get_torch_device().synchronize()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def sync_rollout_weights_by_checkpoint(self, sync_group_name="actor_rollout"):
        assert (self._is_actor or self._is_rollout) and not self.config.hybrid_engine
        assert hasattr(self, "_weights_info") and self._weights_info is not None

        # Load model to GPU
        load_start_time = time.time()
        if self._is_actor and self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        load_duration = time.time() - load_start_time

        from ray.util.collective import collective

        # Cache actor weights to CPU and measure the time taken
        cache_start_time = time.time()
        self.cache_actor_weights_to_cpu()
        cache_end_time = time.time()
        cache_duration = cache_end_time - cache_start_time

        # Register the cached weights into the checkpoint engine
        self.checkpoint_engine.register_checkpoint(self._weights_info, self.cpu_named_params)
        register_end_time = time.time()
        register_duration = register_end_time - cache_end_time
        self.cpu_named_params = {}

        collective.barrier(group_name=sync_group_name)
        update_start_time = time.time()

        inference_model = None
        if self._is_rollout and not self._is_actor:
            inference_model = BaseDetachNcclSync.get_inference_model(self.rollout)
            from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader

            patch_vllm_moe_model_weight_loader(inference_model)

        # Update the checkpoint with the inference model and broadcast weights
        self.checkpoint_engine.update_checkpoint(
            inference_model=inference_model,
            group_name=sync_group_name,
            overlap_broadcast_and_consume=self.config.checkpoint_engine.overlap_broadcast_and_consume,
        )

        update_end_time = time.time()
        update_duration = update_end_time - update_start_time

        offload_start_time = time.time()
        if self._is_actor and self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        offload_duration = time.time() - offload_start_time

        print(
            f"sync_rollout_weights_by_checkpoint finish!, rank:{torch.distributed.get_rank()},"
            f" is_actor:{self._is_actor}, is_rollout:{self._is_rollout},"
            f" total cost:{update_end_time - cache_start_time} seconds, while cache cost {cache_duration} seconds, "
            f" register cost {register_duration} seconds, update cost {update_duration} seconds"
        )

        if self._is_actor and self._is_offload_param:
            print(
                f"sync_rollout_weights_by_checkpoint load model to gpu cost {load_duration} seconds,"
                f" offload model to cpu cost {offload_duration} seconds"
            )


class DetachActorWorker(DetachNcclSync):
    def __init__(self, config: DictConfig, role: str):
        print("[DetachAsyncRolloutWorker] Initializing via DetachNcclSync...")
        DetachNcclSync.__init__(self, config, role)

    def _get_actor_params(self):
        assert self._is_actor
        from verl.utils.model import convert_weight_keys

        peft_model = getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
        if hasattr(peft_model, "peft_config"):
            from dataclasses import asdict

            peft_cfg = peft_model.peft_config.get("default", None)
            self._peft_config = asdict(peft_cfg) if peft_cfg else None
            params = collect_lora_params(
                module=self.actor_module_fsdp,
                layered_summon=False,
                base_sync_done=True,
            )
        else:
            self._peft_config = None
            params = self.actor_module_fsdp.state_dict()

        params = convert_weight_keys(
            params, getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
        )
        return params

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_actor_weights_info(self):
        assert self._is_actor
        if hasattr(self, "_weights_info"):
            return self._weights_info, getattr(self, "_peft_config", None)
        if fsdp_version(self.actor_module_fsdp) == 1:
            from torch.distributed.fsdp.api import ShardedStateDictConfig, StateDictType

            FSDP.set_state_dict_type(
                self.actor_module_fsdp,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )
        params = self._get_actor_params()
        ret = []
        for key, tensor in params.items():
            ret.append((key, tensor.size(), tensor.dtype))
        self._weights_info = ret
        return ret, self._peft_config

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_model_to_cpu(self, n):
        if not hasattr(self, "cpu_saved_models"):
            self.cpu_saved_models = {}
        self.cpu_saved_models[n] = fsdp2_sharded_save_to_cpu(self.actor_module_fsdp)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def restore_model_from_cpu(self, n):
        if n in self.cpu_saved_models:
            cpu_sharded_state, global_spec = self.cpu_saved_models[n]
            fsdp2_sharded_load_from_cpu(self.actor_module_fsdp, cpu_sharded_state, global_spec)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def clear_cpu_model(self, n):
        if n in self.cpu_saved_models:
            del self.cpu_saved_models[n]


class DetachAsyncRolloutWorker(DetachNcclSync):
    def __init__(self, config: DictConfig, role: str):
        print(f"[DetachAsyncRolloutWorker] {DetachAsyncRolloutWorker.__mro__}")
        DetachNcclSync.__init__(self, config, role)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_actor_weights_info(self, weights_info, peft_config=None):
        assert self._is_rollout
        self._weights_info = weights_info
        self._peft_config = peft_config

    # overwrite the following two functions in AsyncActorRolloutRefWorker so that no FSDP module
    # is created for the rollout worker, which takes up gpu unnecessarily
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))

        # Initialize QAT config before _build_model_optimizer
        self._init_qat_config()
        self._build_rollout(trust_remote_code=self.config.model.get("trust_remote_code", False))
        # Free PyTorch's cached-but-unused CUDA memory after init so that
        # vLLM MP Executor workers (separate processes) see it as available.
        aggressive_empty_cache(force_sync=True)
        log_gpu_memory_usage("After init_model empty_cache", logger=logger)

    def _build_rollout(self, trust_remote_code=False):
        from torch.distributed.device_mesh import init_device_mesh

        # 1. parse rollout and huggingface model config
        rollout_config: RolloutConfig = omega_conf_to_dataclass(self.config.rollout)
        model_config: HFModelConfig = omega_conf_to_dataclass(self.config.model, dataclass_type=HFModelConfig)
        self.model_config = model_config
        # 2. build rollout device mesh
        infer_tp = self.config.rollout.tensor_model_parallel_size * self.config.rollout.data_parallel_size
        infer_pp = self.config.rollout.pipeline_model_parallel_size
        infer_world_size = infer_tp * infer_pp
        dp = self.world_size // infer_world_size
        assert self.world_size % infer_world_size == 0, (
            f"rollout world_size: {self.world_size} is not divisible by infer_world_size: {infer_world_size}"
        )
        rollout_device_mesh = init_device_mesh(
            device_name, mesh_shape=(dp, infer_tp, infer_pp), mesh_dim_names=["dp", "infer_tp", "infer_pp"]
        )
        rollout_name = self.config.rollout.name

        self.rollout_device_mesh = rollout_device_mesh

        if rollout_name == "hf":
            self._register_dispatch_collect_info("rollout", dp_rank=self.rank, is_collect=True)
        else:
            is_collect = (
                rollout_device_mesh["infer_tp"].get_local_rank() == 0
                and rollout_device_mesh["infer_pp"].get_local_rank() == 0
            )
            self._register_dispatch_collect_info(
                "rollout", dp_rank=rollout_device_mesh["dp"].get_local_rank(), is_collect=is_collect
            )

        # 4. build rollout model
        log_gpu_memory_usage(f"Before building {self.config.rollout.name} rollout", logger=logger)
        self.rollout = get_rollout_class(rollout_config.name, rollout_config.mode)(
            config=rollout_config, model_config=model_config, device_mesh=rollout_device_mesh
        )
        log_gpu_memory_usage(f"After building {self.config.rollout.name} rollout", logger=logger)
        # used for LoRA
        self.base_sync_done: bool = "dummy" not in self.config.rollout.load_format
        self.layered_summon = self.config.rollout.get("layered_summon", False)
