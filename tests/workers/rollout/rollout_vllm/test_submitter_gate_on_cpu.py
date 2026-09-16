# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""GPU-free tests for the vLLMHttpServer submission gate.

vLLM's pause stops requests being scheduled but still accepts them. A request
admitted between abort_all_requests() and resume_generation() is parked in the
scheduler's waiting queue and masked out of the drain's liveness check, so
wait_for_requests_to_drain() cannot return. These tests pin the ordering that
makes such an admission impossible.
"""

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("ray")
pytest.importorskip("vllm")

from verl.workers.rollout.replica import RolloutMode
from verl.workers.rollout.vllm_rollout import vllm_async_server


class _FakeEngine:
    """Records the state of the gate at the moment the engine is paused."""

    def __init__(self):
        self.output_processor = SimpleNamespace(request_states={})
        self.server = None
        self.pause_calls = 0
        self.resume_calls = 0
        self.admitting_at_pause = None

    async def pause_generation(self, **kwargs):
        self.pause_calls += 1
        self.admitting_at_pause = self.server._admitting

    async def resume_generation(self):
        self.resume_calls += 1


def _make_server(node_rank: int = 0, rollout_mode: RolloutMode = RolloutMode.HYBRID):
    server = object.__new__(vllm_async_server.vLLMHttpServer)
    server.node_rank = node_rank
    server.rollout_mode = rollout_mode
    server.config = SimpleNamespace(free_cache_engine=True)
    server.global_steps = 7
    server.engine = _FakeEngine()
    server.engine.server = server
    server._submission_paused = False
    server._admitting = 0
    server._resume_event = asyncio.Event()
    server._resume_event.set()
    server._rejecting = False
    server.sleep_hybrid_calls = 0

    async def _sleep_hybrid():
        server.sleep_hybrid_calls += 1

    server._sleep_hybrid = _sleep_hybrid
    return server


def test_abort_does_not_pause_until_inflight_admissions_land():
    async def main():
        server = _make_server()
        server._admitting = 1  # a turn is past the gate but not yet in the engine

        abort = asyncio.create_task(server.abort_all_requests())
        await asyncio.sleep(0.05)

        assert server._submission_paused is True, "gate must close before the barrier runs"
        assert not abort.done(), "abort must not proceed while an admission is in flight"
        assert server.engine.pause_calls == 0, "engine paused while an admission was in flight"

        server._admitting = 0  # the in-flight admission reaches the engine
        await asyncio.wait_for(abort, timeout=5)

        assert server.engine.pause_calls == 1
        assert server.engine.admitting_at_pause == 0

    asyncio.run(main())


def test_submission_parks_while_gate_closed_and_wakes_on_resume():
    async def main():
        server = _make_server()
        await server.abort_all_requests()
        assert server._submission_paused is True

        task = asyncio.create_task(server._park_until_admitted("r1"))
        await asyncio.sleep(0.05)
        assert not task.done(), "submission must park while the gate is closed"
        assert server._admitting == 0

        await server.resume_generation()
        assert await asyncio.wait_for(task, timeout=5) is None
        assert server._admitting == 1

    asyncio.run(main())


def test_sleep_behind_closed_gate_rejects_parked_and_new_requests():
    async def main():
        server = _make_server()
        await server.abort_all_requests()

        parked = asyncio.create_task(server._park_until_admitted("parked"))
        await asyncio.sleep(0.05)
        assert not parked.done()

        await server.sleep()
        assert server.sleep_hybrid_calls == 1

        output = await asyncio.wait_for(parked, timeout=5)
        assert output.stop_reason == "aborted", "parked request must fail over instead of waiting for validation"
        assert output.token_ids == []
        assert output.extra_fields["global_steps"] == 7

        late = await asyncio.wait_for(server._park_until_admitted("late"), timeout=5)
        assert late.stop_reason == "aborted", "requests arriving while asleep must be rejected immediately"
        assert server._admitting == 0, "rejected requests never count as admissions"
        assert server._submission_paused is True, "gate stays closed until resume_generation"

    asyncio.run(main())


def test_sleep_with_open_gate_does_not_reject():
    async def main():
        server = _make_server()
        await server.sleep()

        assert server._rejecting is False
        assert await server._park_until_admitted("r1") is None
        assert server._admitting == 1

    asyncio.run(main())


def test_resume_after_sleep_clears_rejection():
    async def main():
        server = _make_server()
        await server.abort_all_requests()
        await server.sleep()
        assert server._rejecting is True

        await server.resume_generation()

        assert server._rejecting is False
        assert server._submission_paused is False
        assert await server._park_until_admitted("r1") is None
        assert server._admitting == 1

    asyncio.run(main())


def test_resume_reopens_gate_on_non_head_server():
    async def main():
        server = _make_server(node_rank=1)
        server._submission_paused = True
        server._resume_event.clear()

        await server.resume_generation()

        assert server._submission_paused is False, "non-head server stays gated forever"
        assert server._resume_event.is_set()
        assert server.engine.resume_calls == 0, "only node rank 0 drives the engine"

    asyncio.run(main())


def test_resume_on_head_server_also_resumes_engine():
    async def main():
        server = _make_server(node_rank=0)
        await server.abort_all_requests()

        await server.resume_generation()

        assert server._submission_paused is False
        assert server._resume_event.is_set()
        assert server.engine.resume_calls == 1

    asyncio.run(main())


def test_barrier_times_out_instead_of_hanging(monkeypatch):
    # raising=False: this asserts the barrier cannot deadlock, not that the constant exists.
    monkeypatch.setattr(vllm_async_server, "_GATE_BARRIER_TIMEOUT_S", 0.05, raising=False)

    async def main():
        server = _make_server()
        server._admitting = 1  # never clears

        await asyncio.wait_for(server.abort_all_requests(), timeout=5)

        assert server.engine.pause_calls == 1, "barrier must proceed rather than deadlock"

    asyncio.run(main())
