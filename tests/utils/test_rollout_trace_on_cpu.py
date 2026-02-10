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
    assert calls["llm.input_messages.1.message.content"] == "Analyze"
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
    assert calls["llm.input_messages.3.message.content"] == "Patch result"
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


# ============================================================================
# Diagnostic tests — verify attribute correctness before deploying
# ============================================================================

def _build_realistic_kaiko_messages():
    """Build a realistic multi-turn conversation matching kaiko_tool_agent output."""
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
        # After first generation — assistant message is a string (decoded tokens)
        {"role": "assistant", "content": "I'll examine the tissue. Let me use extract_patch."},
        # Tool response with image
        {
            "role": "tool",
            "content": [
                {"type": "text", "text": "Extracted patch at (100, 200), size 512x512"},
                {"type": "image"},
            ],
        },
        # Tool response with multiple images
        {
            "role": "tool",
            "content": [
                {"type": "text", "text": "Segmentation results for region A"},
                {"type": "image"},
                {"type": "image"},
            ],
        },
        # Tool response without images — plain string
        {"role": "tool", "content": "Invalid tool call: missing argument 'x'"},
        # Last-turn message
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


def _collect_attributes(mock_span):
    """Collect all set_attribute calls as a dict."""
    return {args[0]: args[1] for args, _ in mock_span.set_attribute.call_args_list}


async def test_diagnostic_realistic_kaiko_conversation(mock_arize_tracer):
    """Full realistic multi-turn conversation — dumps all attributes for inspection."""
    mock_span = _setup_mock_span(mock_arize_tracer)
    RolloutTraceConfig.init(project_name="p", experiment_name="e", backend="arize")

    messages = _build_realistic_kaiko_messages()
    images = [FakePILImage(w, h) for w, h in [(512, 512), (512, 512), (256, 256), (256, 256)]]

    rollout_trace_attach_conversation(
        messages, images, image_format="jpeg", max_dimension=512, span_kind="AGENT"
    )

    attrs = _collect_attributes(mock_span)

    # --- Span kind ---
    assert attrs["openinference.span.kind"] == "AGENT"

    # --- Message 0: system (string content) ---
    assert attrs["llm.input_messages.0.message.role"] == "system"
    assert attrs["llm.input_messages.0.message.content"] == (
        "You are a slide-level analysis agent. Use tools to examine regions."
    )

    # --- Message 1: user with text + 1 image ---
    assert attrs["llm.input_messages.1.message.role"] == "user"
    assert attrs["llm.input_messages.1.message.content"] == (
        "Analyze this slide for abnormal tissue."
    )
    assert attrs["llm.input_messages.1.message.contents.0.message_content.type"] == "text"
    assert attrs["llm.input_messages.1.message.contents.0.message_content.text"] == (
        "Analyze this slide for abnormal tissue."
    )
    assert attrs["llm.input_messages.1.message.contents.1.message_content.type"] == "image"
    img_url = attrs["llm.input_messages.1.message.contents.1.message_content.image.image.url"]
    assert img_url.startswith("data:image/jpeg;base64,"), f"Bad image URL prefix: {img_url[:40]}"

    # --- Message 2: assistant (string content) ---
    assert attrs["llm.input_messages.2.message.role"] == "assistant"
    assert attrs["llm.input_messages.2.message.content"] == (
        "I'll examine the tissue. Let me use extract_patch."
    )

    # --- Message 3: tool with text + 1 image ---
    assert attrs["llm.input_messages.3.message.role"] == "tool"
    assert attrs["llm.input_messages.3.message.contents.0.message_content.type"] == "text"
    assert attrs["llm.input_messages.3.message.contents.0.message_content.text"] == (
        "Extracted patch at (100, 200), size 512x512"
    )
    assert attrs["llm.input_messages.3.message.contents.1.message_content.type"] == "image"
    # message.content also set for Arize text display
    assert attrs["llm.input_messages.3.message.content"] == (
        "Extracted patch at (100, 200), size 512x512"
    )

    # --- Message 4: tool with text + 2 images ---
    assert attrs["llm.input_messages.4.message.role"] == "tool"
    assert attrs["llm.input_messages.4.message.contents.0.message_content.type"] == "text"
    assert attrs["llm.input_messages.4.message.contents.1.message_content.type"] == "image"
    assert attrs["llm.input_messages.4.message.contents.2.message_content.type"] == "image"
    assert attrs["llm.input_messages.4.message.content"] == "Segmentation results for region A"

    # --- Message 5: tool (plain string, no images) ---
    assert attrs["llm.input_messages.5.message.role"] == "tool"
    assert attrs["llm.input_messages.5.message.content"] == (
        "Invalid tool call: missing argument 'x'"
    )

    # --- Message 6: user text-only list content ---
    assert attrs["llm.input_messages.6.message.role"] == "user"
    assert attrs["llm.input_messages.6.message.contents.0.message_content.type"] == "text"
    assert attrs["llm.input_messages.6.message.content"] == (
        "This is the last turn, please make the final analysis."
    )

    # All 4 images consumed
    image_url_keys = [k for k in attrs if k.endswith(".image.image.url")]
    assert len(image_url_keys) == 4, f"Expected 4 image URLs, got {len(image_url_keys)}: {image_url_keys}"


async def test_diagnostic_list_content_also_sets_message_content(mock_arize_tracer):
    """Verifies that list content sets BOTH message.content AND message.contents.*.

    Arize reads message.content for text display. Without it, structured messages
    show empty text. We now set message.content as concatenated text from all
    text items alongside the structured message.contents attributes.
    """
    mock_span = _setup_mock_span(mock_arize_tracer)
    RolloutTraceConfig.init(project_name="p", experiment_name="e", backend="arize")

    messages = [
        {"role": "tool", "content": [
            {"type": "text", "text": "Patch at (100, 200)"},
            {"type": "image"},
            {"type": "text", "text": "Region analysis complete"},
        ]},
    ]
    rollout_trace_attach_conversation(messages, [FakePILImage()])

    attrs = _collect_attributes(mock_span)

    # Role is set
    assert attrs["llm.input_messages.0.message.role"] == "tool"

    # Structured content is set
    assert attrs["llm.input_messages.0.message.contents.0.message_content.type"] == "text"
    assert attrs["llm.input_messages.0.message.contents.0.message_content.text"] == "Patch at (100, 200)"
    assert attrs["llm.input_messages.0.message.contents.1.message_content.type"] == "image"
    assert attrs["llm.input_messages.0.message.contents.2.message_content.type"] == "text"

    # FIX: message.content IS set — concatenated text for Arize display
    assert attrs["llm.input_messages.0.message.content"] == (
        "Patch at (100, 200)\nRegion analysis complete"
    )


async def test_diagnostic_image_count_mismatch_fewer_images(mock_arize_tracer):
    """Tests behavior when fewer images than placeholders — images silently skipped."""
    mock_span = _setup_mock_span(mock_arize_tracer)
    RolloutTraceConfig.init(project_name="p", experiment_name="e", backend="arize")

    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "image"}]},
    ]
    # Only 1 image for 3 placeholders
    rollout_trace_attach_conversation(messages, [FakePILImage()])

    attrs = _collect_attributes(mock_span)
    image_url_keys = [k for k in attrs if k.endswith(".image.image.url")]

    # Only 1 image attached, 2 placeholders silently skipped
    assert len(image_url_keys) == 1, f"Expected 1 image, got {len(image_url_keys)}"


async def test_diagnostic_image_count_mismatch_more_images(mock_arize_tracer):
    """Tests behavior when more images than placeholders — extra images ignored."""
    mock_span = _setup_mock_span(mock_arize_tracer)
    RolloutTraceConfig.init(project_name="p", experiment_name="e", backend="arize")

    messages = [
        {"role": "user", "content": [{"type": "image"}]},
    ]
    # 3 images for 1 placeholder
    rollout_trace_attach_conversation(messages, [FakePILImage(), FakePILImage(), FakePILImage()])

    attrs = _collect_attributes(mock_span)
    image_url_keys = [k for k in attrs if k.endswith(".image.image.url")]

    # Only 1 image attached (matching the placeholder)
    assert len(image_url_keys) == 1, f"Expected 1 image, got {len(image_url_keys)}"


async def test_diagnostic_base64_image_size():
    """Measures actual base64 image size to verify it fits within OPENINFERENCE limits."""
    from verl.utils.rollout_trace import _encode_image

    class RealisticImage(FakePILImage):
        """Writes a payload similar in size to a 512x512 JPEG (~30-80KB)."""

        def save(self, fp, format=None):
            # Simulate ~50KB JPEG
            fp.write(b"\xff\xd8" + b"\x00" * 50_000 + b"\xff\xd9")

    data_uri = _encode_image(RealisticImage(512, 512), image_format="jpeg")
    assert data_uri is not None

    # base64 of 50KB ≈ 66K chars + prefix
    uri_len = len(data_uri)
    prefix = "data:image/jpeg;base64,"
    assert data_uri.startswith(prefix)

    # Should be well under 200K (our env var limit)
    assert uri_len < 200_000, f"Image URI is {uri_len} chars — exceeds 200K limit!"
    # Should exceed old default 32K limit (this was the REDACTED bug)
    assert uri_len > 32_000, f"Image URI is {uri_len} chars — under old 32K default"

    print(f"\n  Base64 image URI length: {uri_len:,} chars (limit: 200,000)")


async def test_diagnostic_openinference_mask_passes_our_attributes():
    """Uses the REAL OpenInference TraceConfig.mask() to verify our keys aren't dropped.

    This catches cases where mask() returns None for our attribute keys,
    which would cause OpenInferenceSpan to silently drop them.
    """
    from openinference.instrumentation.config import TraceConfig

    # Default config (all hide_* = False), with our custom image max length
    config = TraceConfig(base64_image_max_length=200_000)

    test_keys_and_values = [
        ("openinference.span.kind", "AGENT"),
        ("llm.input_messages.0.message.role", "system"),
        ("llm.input_messages.0.message.content", "You are helpful."),
        ("llm.input_messages.1.message.role", "user"),
        ("llm.input_messages.1.message.contents.0.message_content.type", "text"),
        ("llm.input_messages.1.message.contents.0.message_content.text", "Analyze this"),
        ("llm.input_messages.1.message.contents.1.message_content.type", "image"),
        (
            "llm.input_messages.1.message.contents.1.message_content.image.image.url",
            "data:image/jpeg;base64,/9j/AAAA",  # short fake base64
        ),
        ("llm.input_messages.2.message.role", "tool"),
        ("llm.input_messages.2.message.contents.0.message_content.type", "text"),
        ("llm.input_messages.2.message.contents.0.message_content.text", "Patch result"),
    ]

    dropped = []
    for key, value in test_keys_and_values:
        masked = config.mask(key, value)
        if masked is None:
            dropped.append(key)

    assert not dropped, f"TraceConfig.mask() dropped these keys: {dropped}"


async def test_diagnostic_openinference_mask_redacts_large_images():
    """Verifies that the mask REDACTS images over the size limit."""
    from openinference.instrumentation.config import TraceConfig

    # With default 32K limit (simulating missing env var)
    config_default = TraceConfig(base64_image_max_length=32_000)

    # Simulate a 50KB image base64 (≈67K chars)
    import base64

    fake_data = b"\xff" * 50_000
    b64 = base64.b64encode(fake_data).decode()
    data_uri = f"data:image/jpeg;base64,{b64}"

    key = "llm.input_messages.0.message.contents.0.message_content.image.image.url"
    masked = config_default.mask(key, data_uri)
    assert masked == "__REDACTED__", (
        f"Expected REDACTED with 32K limit, got: {str(masked)[:60]}..."
    )

    # With our 200K limit — should pass through
    config_custom = TraceConfig(base64_image_max_length=200_000)
    masked = config_custom.mask(key, data_uri)
    assert masked == data_uri, "Image should NOT be redacted with 200K limit"


async def test_diagnostic_openinference_mask_with_env_var():
    """Verifies that OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH env var is read by TraceConfig."""
    from openinference.instrumentation.config import TraceConfig

    import base64

    fake_data = b"\xff" * 50_000
    b64 = base64.b64encode(fake_data).decode()
    data_uri = f"data:image/jpeg;base64,{b64}"
    key = "llm.input_messages.0.message.contents.0.message_content.image.image.url"

    # Without env var — uses default 32K, should REDACT
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH", None)
        config_no_env = TraceConfig()
        assert config_no_env.base64_image_max_length == 32_000
        assert config_no_env.mask(key, data_uri) == "__REDACTED__"

    # With env var set to 200K — should pass
    with patch.dict(os.environ, {"OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH": "200000"}):
        config_with_env = TraceConfig()
        assert config_with_env.base64_image_max_length == 200_000
        assert config_with_env.mask(key, data_uri) == data_uri
