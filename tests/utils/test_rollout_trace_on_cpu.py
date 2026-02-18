# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from verl.utils.rollout_trace import RolloutTraceConfig, rollout_trace_attach_conversation, rollout_trace_attr, rollout_trace_op


@pytest.fixture(autouse=True)
def reset_rollout_trace_config_singleton():
    """Fixture to reset the RolloutTraceConfig singleton before each test."""
    RolloutTraceConfig.reset()


@pytest.fixture
def mock_weave_client():
    """Mocks the weave module and its client, yielding the mock client."""
    mock_weave = MagicMock()
    mock_client = MagicMock()
    mock_call = MagicMock()
    mock_client.create_call.return_value = mock_call
    mock_weave.init.return_value = mock_client

    # Also mock the call_context if it's used internally by the decorator
    mock_weave.trace.context.call_context.return_value = MagicMock()

    with patch.dict(sys.modules, {"weave": mock_weave, "weave.trace.context": mock_weave.trace.context}):
        yield mock_client


class TracedClass:
    @rollout_trace_op
    # @weave.op
    # @mlflow.trace
    async def my_method(self, a, b="default"):
        return f"result: {a}, {b}"

    @rollout_trace_op
    # @weave.op
    # @mlflow.trace
    async def middle_method(self, a, b="default"):
        await self.my_method("test_a1", b="test_b1")
        return f"result: {a}, {b}"

    @rollout_trace_op
    # @mlflow.trace
    async def my_method_with_exception(self):
        raise ValueError("Test Exception")

    async def upper_method(self):
        await self.my_method("test_a0", b="test_b0")
        await self.middle_method("test_a2", b="test_b2")
        return True


class UntracedClass:
    @rollout_trace_op
    async def my_method(self, x):
        return x * 2


async def test_rollout_trace_on_untraced_class():
    """Tests that the decorator works correctly when no backend is configured."""
    instance = UntracedClass()
    assert await instance.my_method(10) == 20


async def test_rollout_trace_with_tracer(mock_weave_client):
    """Tests that the decorator calls the tracer's methods correctly."""
    RolloutTraceConfig.init(project_name="my-project", experiment_name="my-experiment", backend="weave")
    instance = TracedClass()
    assert RolloutTraceConfig.get_client() is mock_weave_client

    result = await instance.my_method("test_a", b="test_b")

    assert result == "result: test_a, test_b"
    mock_weave_client.create_call.assert_called_once()
    call_kwargs = mock_weave_client.create_call.call_args.kwargs
    assert call_kwargs["op"] == "TracedClass.my_method"
    expected_inputs = {"a": "test_a", "b": "test_b"}
    assert call_kwargs["inputs"] == expected_inputs

    mock_call = mock_weave_client.create_call.return_value
    mock_weave_client.finish_call.assert_called_once_with(mock_call, output=result)


async def test_rollout_trace_with_exception(mock_weave_client):
    """Tests that `finish` is called with the exception when one is raised."""
    RolloutTraceConfig.init(project_name="my-project", experiment_name="my-experiment", backend="weave")
    instance = TracedClass()

    with pytest.raises(ValueError, match="Test Exception"):
        await instance.my_method_with_exception()

    mock_weave_client.create_call.assert_called_once()
    mock_call = mock_weave_client.create_call.return_value
    mock_weave_client.finish_call.assert_called_once()

    # Check that finish_call was called with the exception
    args, kwargs = mock_weave_client.finish_call.call_args
    assert args[0] == mock_call
    assert "exception" in kwargs
    assert isinstance(kwargs["exception"], ValueError)


async def test_rollout_trace_with_dummy_backend(mock_weave_client):
    """Tests that the tracer is not called when the backend is 'dummy'."""
    RolloutTraceConfig.init(project_name="my-project", experiment_name="my-experiment", backend="dummy")
    instance = TracedClass()

    await instance.my_method("test_a")

    mock_weave_client.create_call.assert_not_called()


async def test_trace_disabled_with_trace_false(mock_weave_client):
    """Tests that tracing is disabled when trace=False."""
    RolloutTraceConfig.init(
        project_name="my-project",
        experiment_name="my-experiment",
        backend="weave",
    )
    instance = TracedClass()

    assert RolloutTraceConfig.get_backend() == "weave"

    with rollout_trace_attr(step=1, sample_index=0, rollout_n=0, trace=False):
        result = await instance.my_method("test_a", b="test_b")
        assert result == "result: test_a, test_b"

    # No tracing should have occurred
    mock_weave_client.create_call.assert_not_called()

    # Verify that tracing works again with trace=True (default)
    with rollout_trace_attr(step=1, sample_index=0, rollout_n=0):
        result = await instance.my_method("test_a", b="test_b")
        assert result == "result: test_a, test_b"

    assert mock_weave_client.create_call.call_count == 1


async def test_trace_false_disables_nested_trace_ops(mock_weave_client):
    """Tests that trace=False disables all nested @rollout_trace_op calls."""
    RolloutTraceConfig.init(
        project_name="my-project",
        experiment_name="my-experiment",
        backend="weave",
    )
    instance = TracedClass()

    with rollout_trace_attr(step=1, sample_index=0, rollout_n=0, trace=False):
        # Call upper_method which internally calls my_method and middle_method
        # All of these are decorated with @rollout_trace_op
        result = await instance.upper_method()
        assert result is True

    # No tracing should have occurred for any of the nested calls
    mock_weave_client.create_call.assert_not_called()

    with rollout_trace_attr(step=1, sample_index=0, rollout_n=0):
        result = await instance.my_method("test_a", b="test_b")
        assert result == "result: test_a, test_b"

    assert mock_weave_client.create_call.call_count == 1


async def test_trace_enabled_restored_after_exception(mock_weave_client):
    """Tests that trace state is restored even if an exception occurs when trace=False."""
    RolloutTraceConfig.init(
        project_name="my-project",
        experiment_name="my-experiment",
        backend="weave",
    )
    instance = TracedClass()

    assert RolloutTraceConfig.get_backend() == "weave"

    # Use trace=False and raise an exception
    try:
        with rollout_trace_attr(step=1, sample_index=0, rollout_n=0, trace=False):
            raise RuntimeError("Test exception with trace disabled")
    except RuntimeError:
        pass

    with rollout_trace_attr(step=1, sample_index=0, rollout_n=0):
        result = await instance.my_method("test_a", b="test_b")
        assert result == "result: test_a, test_b"

    assert mock_weave_client.create_call.call_count == 1


@pytest.mark.skipif(
    os.environ.get("RUN_WEAVE_INTEGRATION_TESTS", "false").lower() != "true",
    reason="Skipping weave integration test. Set RUN_WEAVE_INTEGRATION_TESTS=true to run.",
)
async def test_rollout_trace_with_real_weave_backend():
    """Integration test with a real weave backend."""

    # This assumes that the weave environment (e.g., project) is configured
    RolloutTraceConfig.init(project_name="my-project", experiment_name="my-experiment", backend="weave")

    instance = TracedClass()

    with rollout_trace_attr(step=1, sample_index=2, rollout_n=3):
        await instance.upper_method()

    with pytest.raises(ValueError, match="Test Exception"):
        await instance.my_method_with_exception()

    print("\nWeave integration test ran successfully. Check your weave project for the trace.")


@pytest.mark.skipif(
    os.environ.get("RUN_MLFLOW_INTEGRATION_TESTS", "false").lower() != "true",
    reason="Skipping mlflow integration test. Set RUN_MLFLOW_INTEGRATION_TESTS=true to run.",
)
async def test_rollout_trace_with_real_mlflow_backend():
    """Integration test with a real mlflow backend."""

    # This assumes that the mlflow environment (e.g., project) is configured
    RolloutTraceConfig.init(project_name="my-project", experiment_name="my-experiment", backend="mlflow")

    instance = TracedClass()

    with rollout_trace_attr(step=1, sample_index=2, rollout_n=3, name="agent_run"):
        assert await instance.upper_method()

    print("\nMlflow integration test ran successfully. Check your mlflow project for the trace.")


@pytest.fixture
def mock_arize_tracer():
    """Mocks the arize.otel and opentelemetry modules, yielding (mock_tracer, mock_span)."""
    mock_arize_otel = MagicMock()
    mock_otel = MagicMock()
    mock_tracer = MagicMock()
    mock_span = MagicMock()
    mock_span.__enter__ = MagicMock(return_value=mock_span)
    mock_span.__exit__ = MagicMock(return_value=False)
    mock_tracer.start_as_current_span.return_value = mock_span
    mock_otel.trace.get_tracer.return_value = mock_tracer
    mock_otel.trace.Status = MagicMock()
    mock_otel.trace.StatusCode = MagicMock()
    mock_otel.trace.StatusCode.OK = "OK"
    mock_otel.trace.StatusCode.ERROR = "ERROR"

    with patch.dict(
        sys.modules,
        {
            "arize": MagicMock(),
            "arize.otel": mock_arize_otel,
            "opentelemetry": mock_otel,
            "opentelemetry.trace": mock_otel.trace,
        },
    ):
        with patch.dict(os.environ, {"ARIZE_SPACE_ID": "test-space", "ARIZE_API_KEY": "test-key"}):
            yield mock_tracer, mock_span


async def test_rollout_trace_with_arize_tracer(mock_arize_tracer):
    """Tests that the Arize backend creates spans without input/output dumps."""
    mock_tracer, mock_span = mock_arize_tracer
    RolloutTraceConfig.init(project_name="my-project", experiment_name="my-experiment", backend="arize")
    instance = TracedClass()

    result = await instance.my_method("test_a", b="test_b")

    assert result == "result: test_a, test_b"
    mock_tracer.start_as_current_span.assert_called_once()
    call_kwargs = mock_tracer.start_as_current_span.call_args.kwargs
    assert call_kwargs["name"] == "TracedClass.my_method"
    # Child span should only have span kind, not input/output dumps
    assert call_kwargs["attributes"] == {"openinference.span.kind": "CHAIN"}
    # No output.value with raw token dump
    output_calls = [c for c in mock_span.set_attribute.call_args_list if c[0][0] == "output.value"]
    assert len(output_calls) == 0


async def test_rollout_trace_arize_with_exception(mock_arize_tracer):
    """Tests that the Arize backend records exceptions on spans."""
    mock_tracer, mock_span = mock_arize_tracer
    RolloutTraceConfig.init(project_name="my-project", experiment_name="my-experiment", backend="arize")
    instance = TracedClass()

    with pytest.raises(ValueError, match="Test Exception"):
        await instance.my_method_with_exception()

    mock_tracer.start_as_current_span.assert_called_once()
    mock_span.record_exception.assert_called_once()
    assert isinstance(mock_span.record_exception.call_args[0][0], ValueError)
    mock_span.set_status.assert_called_once()


async def test_trace_disabled_with_arize(mock_arize_tracer):
    """Tests that trace=False disables Arize tracing."""
    mock_tracer, mock_span = mock_arize_tracer
    RolloutTraceConfig.init(project_name="my-project", experiment_name="my-experiment", backend="arize")
    instance = TracedClass()

    with rollout_trace_attr(step=1, sample_index=0, rollout_n=0, trace=False):
        result = await instance.my_method("test_a", b="test_b")
        assert result == "result: test_a, test_b"

    mock_tracer.start_as_current_span.assert_not_called()


@pytest.mark.skipif(
    os.environ.get("RUN_ARIZE_INTEGRATION_TESTS", "false").lower() != "true",
    reason="Skipping Arize integration test. Set RUN_ARIZE_INTEGRATION_TESTS=true to run.",
)
async def test_rollout_trace_with_real_arize_backend():
    """Integration test with a real Arize backend."""

    RolloutTraceConfig.init(project_name="my-project", experiment_name="my-experiment", backend="arize")

    instance = TracedClass()

    with rollout_trace_attr(step=1, sample_index=2, rollout_n=3):
        await instance.upper_method()

    with pytest.raises(ValueError, match="Test Exception"):
        await instance.my_method_with_exception()

    print("\nArize integration test ran successfully. Check your Arize project for the trace.")


class FakePILImage:
    """Minimal duck-type stand-in for a PIL Image."""

    def __init__(self, width=100, height=80):
        self._size = (width, height)

    @property
    def size(self):
        return self._size

    def resize(self, size, resample=None):
        return FakePILImage(width=size[0], height=size[1])

    def save(self, fp, format=None):
        # Write a tiny valid payload so base64 is non-empty
        fp.write(b"\x89PNG_FAKE")

    class Resampling:
        LANCZOS = 1


def _build_realistic_messages():
    """Build a realistic multi-turn conversation."""
    return [
        {
            "role": "system",
            "content": "You are a slide-level analysis agent. Use tools to examine regions.",
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this slide for abnormal tissue."},
                {"type": "image"},
            ],
        },
        {"role": "assistant", "content": "I'll examine the tissue. Let me use extract_patch."},
        {
            "role": "tool",
            "content": [
                {"type": "text", "text": "Extracted patch at (100, 200), size 512x512"},
                {"type": "image"},
            ],
        },
        {
            "role": "tool",
            "content": [
                {"type": "text", "text": "Segmentation results for region A"},
                {"type": "image"},
                {"type": "image"},
            ],
        },
        {"role": "tool", "content": "Invalid tool call: missing argument 'x'"},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "This is the last turn, please make the final analysis.",
                }
            ],
        },
    ]


# ============================================================================
# Auto-attach conversation tests
# ============================================================================


@pytest.fixture
def real_otel_pipeline():
    """Creates a real OpenInference tracing pipeline with in-memory span export."""
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from openinference.instrumentation import TracerProvider as OITracerProvider
    from openinference.instrumentation.config import TraceConfig

    exporter = InMemorySpanExporter()

    config = TraceConfig(base64_image_max_length=200_000)
    provider = OITracerProvider(config=config)
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    tracer = provider.get_tracer("test-tracer")

    rc = RolloutTraceConfig.get_instance()
    rc.backend = "arize"
    rc.client = tracer
    rc._initialized = True

    def get_exported_spans():
        provider.force_flush()
        return exporter.get_finished_spans()

    yield tracer, get_exported_spans

    provider.shutdown()


async def test_auto_attach_conversation():
    """End-to-end: result.trace_conversation triggers auto-attach in the decorator."""
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from openinference.instrumentation import TracerProvider as OITracerProvider
    from openinference.instrumentation.config import TraceConfig

    exporter = InMemorySpanExporter()
    config = TraceConfig(base64_image_max_length=200_000)
    provider = OITracerProvider(config=config)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test-auto-attach")

    rc = RolloutTraceConfig.get_instance()
    rc.backend = "arize"
    rc.client = tracer
    rc._initialized = True

    messages = _build_realistic_messages()
    images = [FakePILImage(512, 512) for _ in range(4)]

    class FakeResult:
        def __init__(self):
            self.trace_conversation = messages
            self.multi_modal_data = {"image": images}
            self.reward_score = 0.85
            self.num_turns = 5
            self.prompt_ids = [1, 2]
            self.response_ids = [3, 4]
            self.response_mask = [1, 1]
            self.extra_fields = {}
            self.metrics = {}

    class AgentUnderTest:
        @rollout_trace_op
        async def run(self):
            return FakeResult()

    agent = AgentUnderTest()
    result = await agent.run()

    provider.force_flush()
    exported = exporter.get_finished_spans()
    assert len(exported) >= 1

    attrs = dict(exported[0].attributes)

    # Verify span kind was set to AGENT by auto-attach
    assert attrs.get("openinference.span.kind") == "AGENT"

    # Verify messages were attached
    assert attrs.get("llm.input_messages.0.message.role") == "system"
    assert "llm.input_messages.0.message.content" in attrs
    assert attrs.get("llm.input_messages.1.message.role") == "user"
    assert attrs.get("llm.input_messages.2.message.role") == "assistant"
    assert attrs.get("llm.input_messages.3.message.role") == "tool"

    # Verify images were attached (4 image placeholders in messages)
    image_url_keys = [k for k in attrs if k.endswith(".image.image.url")]
    assert len(image_url_keys) == 4

    # trace_conversation should be cleared after decorator
    assert result.trace_conversation is None

    provider.shutdown()


async def test_auto_attach_noop_when_none():
    """No trace_conversation on result → no error, no conversation attributes."""
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from openinference.instrumentation import TracerProvider as OITracerProvider

    exporter = InMemorySpanExporter()
    provider = OITracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test-noop")

    rc = RolloutTraceConfig.get_instance()
    rc.backend = "arize"
    rc.client = tracer
    rc._initialized = True

    class FakeResult:
        def __init__(self):
            self.reward_score = 0.5
            self.num_turns = 1
            self.prompt_ids = [1]
            self.response_ids = [2]
            self.response_mask = [1]
            self.extra_fields = {}
            self.metrics = {}

    class AgentUnderTest:
        @rollout_trace_op
        async def run(self):
            return FakeResult()

    agent = AgentUnderTest()
    await agent.run()

    provider.force_flush()
    exported = exporter.get_finished_spans()
    assert len(exported) == 1

    attrs = dict(exported[0].attributes)
    msg_keys = [k for k in attrs if k.startswith("llm.input_messages")]
    assert len(msg_keys) == 0

    provider.shutdown()


async def test_auto_attach_clears_field():
    """Verify result.trace_conversation is None after the decorator returns."""
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from openinference.instrumentation import TracerProvider as OITracerProvider
    from openinference.instrumentation.config import TraceConfig

    exporter = InMemorySpanExporter()
    config = TraceConfig(base64_image_max_length=200_000)
    provider = OITracerProvider(config=config)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test-clear")

    rc = RolloutTraceConfig.get_instance()
    rc.backend = "arize"
    rc.client = tracer
    rc._initialized = True

    class FakeResult:
        def __init__(self):
            self.trace_conversation = [{"role": "user", "content": "hello"}]
            self.multi_modal_data = {}
            self.reward_score = 0.0
            self.num_turns = 1
            self.prompt_ids = [1]
            self.response_ids = [2]
            self.response_mask = [1]
            self.extra_fields = {}
            self.metrics = {}

    class AgentUnderTest:
        @rollout_trace_op
        async def run(self):
            return FakeResult()

    agent = AgentUnderTest()
    result = await agent.run()

    assert result.trace_conversation is None

    provider.shutdown()


async def test_auto_attach_uses_arize_config():
    """Verify image_format/max_dimension come from arize_config, span_kind is always AGENT."""
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from openinference.instrumentation import TracerProvider as OITracerProvider
    from openinference.instrumentation.config import TraceConfig

    exporter = InMemorySpanExporter()
    config = TraceConfig(base64_image_max_length=200_000)
    provider = OITracerProvider(config=config)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test-arize-config")

    rc = RolloutTraceConfig.get_instance()
    rc.backend = "arize"
    rc.client = tracer
    rc._initialized = True
    rc.arize_config = {"image_format": "jpeg", "max_dimension": 256}

    saved_sizes = []

    class TrackingImage(FakePILImage):
        def resize(self, size, resample=None):
            return TrackingImage(width=size[0], height=size[1])

        def save(self, fp, format=None):
            saved_sizes.append((self.size, format))
            fp.write(b"\x89PNG_FAKE")

    class FakeResult:
        def __init__(self):
            self.trace_conversation = [
                {"role": "user", "content": [{"type": "image"}]},
            ]
            self.multi_modal_data = {"image": [TrackingImage(width=1024, height=512)]}
            self.reward_score = 0.0
            self.num_turns = 1
            self.prompt_ids = [1]
            self.response_ids = [2]
            self.response_mask = [1]
            self.extra_fields = {}
            self.metrics = {}

    class AgentUnderTest:
        @rollout_trace_op
        async def run(self):
            return FakeResult()

    agent = AgentUnderTest()
    await agent.run()

    provider.force_flush()
    exported = exporter.get_finished_spans()
    attrs = dict(exported[0].attributes)

    # span_kind is always AGENT
    assert attrs.get("openinference.span.kind") == "AGENT"

    # Image was resized to max_dimension=256 (1024x512 → 256x128)
    assert saved_sizes[0][0] == (256, 128)
    # Image format from arize_config
    assert saved_sizes[0][1] == "JPEG"

    # Image URL uses jpeg format
    img_key = "llm.input_messages.0.message.contents.0.message_content.image.image.url"
    assert attrs[img_key].startswith("data:image/jpeg;base64,")

    provider.shutdown()


def test_trace_conversation_excluded_from_model_dump():
    """A pydantic Field(exclude=True) should not appear in model_dump()."""
    from typing import Any, Optional

    from pydantic import BaseModel, Field

    class MinimalOutput(BaseModel):
        prompt_ids: list[int]
        response_ids: list[int]
        trace_conversation: Optional[list[dict[str, Any]]] = Field(default=None, exclude=True)

    output = MinimalOutput(prompt_ids=[1, 2], response_ids=[3, 4])
    output.trace_conversation = [{"role": "user", "content": "hello"}]

    dumped = output.model_dump()
    assert "trace_conversation" not in dumped
    assert "prompt_ids" in dumped


async def test_root_span_metadata_from_rollout_trace_op():
    """Verifies that rollout_trace_op sets scalar inputs on the root (rollout_trace_attr) span."""
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from openinference.instrumentation import TracerProvider as OITracerProvider

    exporter = InMemorySpanExporter()
    provider = OITracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test-root-metadata")

    rc = RolloutTraceConfig.get_instance()
    rc.backend = "arize"
    rc.client = tracer
    rc._initialized = True

    class FakeResult:
        def __init__(self):
            self.reward_score = 0.85
            self.num_turns = 7
            self.prompt_ids = [1, 2, 3]
            self.response_ids = [4, 5, 6]
            self.response_mask = [1, 1, 1]
            self.response_logprobs = [-0.1, -0.2]
            self.multi_modal_data = {}
            self.extra_fields = {"turn_scores": [0.1, 0.2]}
            self.metrics = {"latency": 1.5}

    class AgentUnderTest:
        @rollout_trace_op
        async def run(self, sampling_params, **kwargs):
            return FakeResult()

    agent = AgentUnderTest()

    with rollout_trace_attr(
        step=5, sample_index=42, rollout_n=3, validate=False, name="agent_loop",
    ):
        await agent.run(
            sampling_params={"temperature": 0.7},
            wsi_name="TCGA-TEST-001",
            data_source="test_dataset",
            primary_site_bb="Breast",
            index=207,
            extra_info={"wsi_id": "TCGA-TEST-001", "slide_level_task": True, "reward_groups": ["a", "b"]},
            messages=[{"role": "system", "content": "hello"}],
            multi_modal_inputs={"image": [1, 2, 3]},
        )

    provider.force_flush()
    exported = exporter.get_finished_spans()
    assert len(exported) == 2, f"Expected 2 spans (child + root), got {len(exported)}"

    # Spans are exported child-first, root-last
    child_attrs = dict(exported[0].attributes)
    root_attrs = dict(exported[1].attributes)

    # Root span should have the original rollout_trace_attr attributes
    assert root_attrs.get("step") == "5"
    assert root_attrs.get("sample_index") == "42"

    # Root span should also have scalar kwargs from rollout_trace_op
    assert root_attrs.get("wsi_name") == "TCGA-TEST-001"
    assert root_attrs.get("data_source") == "test_dataset"
    assert root_attrs.get("primary_site_bb") == "Breast"
    assert root_attrs.get("index") == "207"

    # extra_info scalars should be flattened one level
    assert root_attrs.get("extra_info.wsi_id") == "TCGA-TEST-001"
    assert root_attrs.get("extra_info.slide_level_task") == "True"

    # Non-scalar extra_info values should be skipped
    assert "extra_info.reward_groups" not in root_attrs

    # Skipped keys should not appear on root span
    assert "messages" not in root_attrs
    assert "multi_modal_inputs" not in root_attrs
    assert "sampling_params" not in root_attrs

    # Root span should have scalar result fields (reward_score, num_turns)
    assert root_attrs.get("reward_score") == "0.85"
    assert root_attrs.get("num_turns") == "7"

    # Raw tensor fields from result should NOT appear on root span
    assert "prompt_ids" not in root_attrs
    assert "response_ids" not in root_attrs
    assert "response_mask" not in root_attrs
    assert "response_logprobs" not in root_attrs
    assert "extra_fields" not in root_attrs
    assert "metrics" not in root_attrs

    # Child span should NOT have input/output dumps
    assert "input.value" not in child_attrs
    assert "output.value" not in child_attrs

    provider.shutdown()
