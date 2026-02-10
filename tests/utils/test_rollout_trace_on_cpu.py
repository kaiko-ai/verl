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

    # with pytest.raises(ValueError, match="Test Exception"):
    #     await instance.my_method_with_exception()

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
    """Tests that the Arize backend creates spans with inputs/outputs."""
    mock_tracer, mock_span = mock_arize_tracer
    RolloutTraceConfig.init(project_name="my-project", experiment_name="my-experiment", backend="arize")
    instance = TracedClass()

    result = await instance.my_method("test_a", b="test_b")

    assert result == "result: test_a, test_b"
    mock_tracer.start_as_current_span.assert_called_once()
    call_kwargs = mock_tracer.start_as_current_span.call_args.kwargs
    assert call_kwargs["name"] == "TracedClass.my_method"
    assert "input.value" in call_kwargs["attributes"]
    mock_span.set_attribute.assert_any_call("output.value", str(result))


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


def _setup_mock_span(mock_arize_tracer):
    """Helper to configure mock span for attach_conversation tests."""
    import sys

    mock_tracer, mock_span = mock_arize_tracer
    mock_span.is_recording.return_value = True
    mock_otel = sys.modules["opentelemetry"]
    mock_otel.trace.get_current_span.return_value = mock_span
    return mock_span


async def test_attach_conversation_structured_messages(mock_arize_tracer):
    """Tests that a multi-turn conversation is serialized with roles, text, and images."""
    mock_span = _setup_mock_span(mock_arize_tracer)
    RolloutTraceConfig.init(project_name="my-project", experiment_name="my-experiment", backend="arize")

    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": [{"type": "text", "text": "Analyze"}, {"type": "image"}]},
        {"role": "assistant", "content": "I see tissue."},
        {"role": "tool", "content": [{"type": "text", "text": "Patch result"}, {"type": "image"}]},
    ]
    images = [FakePILImage(), FakePILImage()]
    rollout_trace_attach_conversation(messages, images, image_format="png")

    calls = {args[0]: args[1] for args, _ in mock_span.set_attribute.call_args_list}

    # Message 0: system with string content
    assert calls["llm.input_messages.0.message.role"] == "system"
    assert calls["llm.input_messages.0.message.content"] == "You are helpful."

    # Message 1: user with text + image
    assert calls["llm.input_messages.1.message.role"] == "user"
    assert calls["llm.input_messages.1.message.contents.0.message_content.type"] == "text"
    assert calls["llm.input_messages.1.message.contents.0.message_content.text"] == "Analyze"
    assert calls["llm.input_messages.1.message.contents.1.message_content.type"] == "image"
    assert calls["llm.input_messages.1.message.contents.1.message_content.image.image.url"].startswith(
        "data:image/png;base64,"
    )

    # Message 2: assistant with string content
    assert calls["llm.input_messages.2.message.role"] == "assistant"
    assert calls["llm.input_messages.2.message.content"] == "I see tissue."

    # Message 3: tool with text + image
    assert calls["llm.input_messages.3.message.role"] == "tool"
    assert calls["llm.input_messages.3.message.contents.0.message_content.type"] == "text"
    assert calls["llm.input_messages.3.message.contents.1.message_content.type"] == "image"


async def test_attach_conversation_sets_span_kind(mock_arize_tracer):
    """Tests that span_kind parameter overrides openinference.span.kind attribute."""
    mock_span = _setup_mock_span(mock_arize_tracer)
    RolloutTraceConfig.init(project_name="my-project", experiment_name="my-experiment", backend="arize")

    messages = [{"role": "user", "content": "hello"}]
    rollout_trace_attach_conversation(messages, span_kind="AGENT")

    calls = {args[0]: args[1] for args, _ in mock_span.set_attribute.call_args_list}
    assert calls["openinference.span.kind"] == "AGENT"


async def test_attach_conversation_noop_without_backend():
    """Tests that attach_conversation is a no-op when no backend is configured."""
    messages = [{"role": "user", "content": "hello"}]
    rollout_trace_attach_conversation(messages, [FakePILImage()])


async def test_attach_conversation_noop_with_weave(mock_weave_client):
    """Tests that attach_conversation is a no-op when Weave backend is configured."""
    RolloutTraceConfig.init(project_name="my-project", experiment_name="my-experiment", backend="weave")
    messages = [{"role": "user", "content": "hello"}]
    rollout_trace_attach_conversation(messages, [FakePILImage()])


async def test_attach_conversation_with_resize(mock_arize_tracer):
    """Tests that images are resized when max_dimension is set."""
    mock_span = _setup_mock_span(mock_arize_tracer)
    RolloutTraceConfig.init(project_name="my-project", experiment_name="my-experiment", backend="arize")

    saved_sizes = []

    class TrackingImage(FakePILImage):
        def resize(self, size, resample=None):
            return TrackingImage(width=size[0], height=size[1])

        def save(self, fp, format=None):
            saved_sizes.append(self.size)
            fp.write(b"\x89PNG_FAKE")

    messages = [{"role": "user", "content": [{"type": "image"}]}]
    rollout_trace_attach_conversation(messages, [TrackingImage(width=1024, height=512)], max_dimension=256)

    assert saved_sizes[0] == (256, 128)


async def test_attach_conversation_respects_trace_disabled(mock_arize_tracer):
    """Tests that attach_conversation is a no-op when tracing is disabled."""
    mock_span = _setup_mock_span(mock_arize_tracer)
    RolloutTraceConfig.init(project_name="my-project", experiment_name="my-experiment", backend="arize")

    messages = [{"role": "user", "content": [{"type": "image"}]}]

    with rollout_trace_attr(step=1, sample_index=0, rollout_n=0, trace=False):
        rollout_trace_attach_conversation(messages, [FakePILImage()])

    mock_span.set_attribute.assert_not_called()
