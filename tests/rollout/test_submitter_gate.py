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
"""Deterministic asyncio model of the vLLMHttpServer submitter gate.

This mirrors the gate/barrier logic added to
``verl/workers/rollout/vllm_rollout/vllm_async_server.py`` (generate /
abort_all_requests / resume_generation). It does NOT import the real server
(that needs Ray + vLLM + a full config); it validates the concurrency
invariants that make the weight-sync drain hang impossible:

  1. while paused, a turn parks and never admits;
  2. the abort barrier does not snapshot until in-flight admissions clear;
  3. NO admission's add_request lands after the abort snapshot (the orphan
     that pins dp_running=True), across many concurrent turns racing an abort;
  4. the _admitting counter always balances back to 0;
  5. resume wakes parked turns.

If the gate logic here diverges from the server, keep them in sync; the real
integration proof is the instrumented cluster run.
"""
import asyncio

import pytest


class _FakeEngine:
    """Records the ordering invariant: an admission (add_request) must never be
    sent after the abort snapshot was taken."""

    def __init__(self) -> None:
        self.snapshot_taken = False
        self.violations: list[str] = []
        self.admitted_ids: list[str] = []

    async def add_request_send(self, request_id: str) -> None:
        # Simulate scheduling + the ZMQ enqueue of add_request.
        await asyncio.sleep(0)
        if self.snapshot_taken:
            self.violations.append(request_id)  # orphan: admitted after snapshot
        self.admitted_ids.append(request_id)

    async def take_snapshot(self) -> None:
        # Models pause_generation()'s finish_requests(None) abort snapshot.
        self.snapshot_taken = True


class _GateServer:
    """Faithful copy of the gate/barrier logic under test."""

    def __init__(self) -> None:
        self.engine = _FakeEngine()
        self._submission_paused = False
        self._admitting = 0
        self._resume_event = asyncio.Event()
        self._resume_event.set()

    async def generate(self, request_id: str) -> None:
        # --- gate (mirrors generate()) ---
        while self._submission_paused:
            await self._resume_event.wait()
        self._admitting += 1  # NO await between the check above and here
        admitted = False
        try:
            await self.engine.add_request_send(request_id)  # first output => admitted
            admitted = True
            self._admitting -= 1
            await asyncio.sleep(0)  # stream remaining outputs
        finally:
            if not admitted:
                self._admitting -= 1

    async def abort_all_requests(self) -> None:
        # --- barrier (mirrors abort_all_requests()) ---
        self._submission_paused = True
        self._resume_event.clear()
        t0 = 0
        while self._admitting > 0:
            t0 += 1
            if t0 > 100_000:
                raise AssertionError("barrier did not drain")
            await asyncio.sleep(0)
        await self.engine.take_snapshot()

    async def resume_generation(self) -> None:
        self._submission_paused = False
        self._resume_event.set()


@pytest.mark.asyncio
async def test_paused_turn_parks_then_resumes() -> None:
    s = _GateServer()
    s._submission_paused = True
    s._resume_event.clear()
    task = asyncio.create_task(s.generate("r1"))
    await asyncio.sleep(0.02)
    assert not task.done(), "turn should park while paused"
    assert s.engine.admitted_ids == [], "parked turn must not admit"
    await s.resume_generation()
    await task
    assert s.engine.admitted_ids == ["r1"]
    assert s._admitting == 0


@pytest.mark.asyncio
async def test_barrier_waits_for_inflight_admission() -> None:
    s = _GateServer()
    s._admitting = 1  # simulate a turn already past the gate, mid-admission
    abort = asyncio.create_task(s.abort_all_requests())
    await asyncio.sleep(0.02)
    assert not abort.done(), "barrier must wait while _admitting > 0"
    assert not s.engine.snapshot_taken
    s._admitting = 0  # the in-flight admission completes
    await abort
    assert s.engine.snapshot_taken


@pytest.mark.asyncio
async def test_no_admission_after_snapshot_under_concurrency() -> None:
    # Many turns race an abort fired at various interleavings; none may orphan.
    for delay_ticks in range(0, 12):
        s = _GateServer()
        turns = [asyncio.create_task(s.generate(f"r{i}")) for i in range(32)]

        async def fire_abort() -> None:
            for _ in range(delay_ticks):
                await asyncio.sleep(0)
            await s.abort_all_requests()

        await asyncio.gather(fire_abort(), *turns, return_exceptions=False)
        assert s.engine.violations == [], (
            f"admission after snapshot (delay_ticks={delay_ticks}): {s.engine.violations}"
        )
        assert s._admitting == 0, f"counter imbalance: {s._admitting}"


@pytest.mark.asyncio
async def test_counter_balances_with_zero_output_turn() -> None:
    s = _GateServer()

    async def raise_after_gate(request_id: str) -> None:
        while s._submission_paused:
            await s._resume_event.wait()
        s._admitting += 1
        admitted = False
        try:
            raise RuntimeError("engine died before first output")
        finally:
            if not admitted:
                s._admitting -= 1

    with pytest.raises(RuntimeError):
        await raise_after_gate("r1")
    assert s._admitting == 0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and asyncio.iscoroutinefunction(fn):
            asyncio.run(fn())
            print(f"PASS {name}")
    print("ALL PASS")
