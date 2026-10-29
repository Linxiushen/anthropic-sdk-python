"""Offline SDK stream/decode benchmark. Run unchanged before and after a patch."""

from __future__ import annotations

import json
import hashlib
import platform
import statistics
import sys
import time
from pathlib import Path

import anyio
import httpx2
import pydantic

import anthropic
import anthropic._models
from anthropic._models import construct_type
from anthropic.types import RawContentBlockDeltaEvent, RawMessageStreamEvent


REPEATS = 7
DELTAS = 3000
DECODES = 12000
DELTA = {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "x"}}


def payload() -> bytes:
    events = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_offline_benchmark",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-6",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 7, "output_tokens": 0},
            },
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        *[DELTA] * DELTAS,
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": {"output_tokens": DELTAS}},
        {"type": "message_stop"},
    ]
    return "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events).encode()


def describe(seconds: list[float], count: int) -> dict:
    return {"seconds": seconds, "median_seconds": statistics.median(seconds), "median_us_per_event": statistics.median(seconds) * 1e6 / count}


def decode() -> dict:
    for _ in range(100):
        construct_type(type_=RawMessageStreamEvent, value=DELTA)
    elapsed = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        for _ in range(DECODES):
            event = construct_type(type_=RawMessageStreamEvent, value=DELTA)
        elapsed.append(time.perf_counter() - start)
        assert isinstance(event, RawContentBlockDeltaEvent)
        assert event.delta.type == "text_delta" and event.delta.text == "x"
    return describe(elapsed, DECODES)


def large_events() -> dict:
    tool = {
        "type": "tool_use", "id": "toolu_offline", "name": "analyze_rows",
        "input": {"rows": [{"id": i, "scores": [i, i + 1, i + 2], "meta": {"name": str(i)}} for i in range(1000)]},
    }
    events = {
        "content_block_start": {"type": "content_block_start", "index": 0, "content_block": tool},
        "message_start": {"type": "message_start", "message": {
            "id": "msg_large", "type": "message", "role": "assistant", "model": "claude-sonnet-4-6",
            "content": [tool], "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": 7, "output_tokens": 0},
        }},
    }
    results = {}
    for name, event in events.items():
        elapsed = []
        for _ in range(REPEATS):
            start = time.perf_counter()
            for _ in range(2000):
                parsed = construct_type(type_=RawMessageStreamEvent, value=event)
            elapsed.append(time.perf_counter() - start)
        assert parsed.type == name
        results[name] = describe(elapsed, 2000)
    return results


async def streams() -> dict:
    wire = payload()
    requests = 0

    def respond(request: httpx2.Request) -> httpx2.Response:
        nonlocal requests
        requests += 1
        assert request.url.path == "/v1/messages"
        body = json.loads(request.content)
        assert body["stream"] is True
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=wire)

    results = {}
    async with anthropic.AsyncAnthropic(
        api_key="offline-test-only",
        base_url="https://offline.invalid",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(respond)),
    ) as client:
        for name, resource in [("stable", client.messages), ("beta", client.beta.messages)]:
            elapsed = []
            for iteration in range(REPEATS + 1):
                start = time.perf_counter()
                async with resource.stream(model="claude-sonnet-4-6", max_tokens=DELTAS, messages=[{"role": "user", "content": "offline"}]) as stream:
                    message = await stream.get_final_message()
                seconds = time.perf_counter() - start
                assert message.content[0].type == "text" and message.content[0].text == "x" * DELTAS
                assert message.usage.input_tokens == 7 and message.usage.output_tokens == DELTAS
                assert message.stop_reason == "end_turn"
                if iteration:
                    elapsed.append(seconds)
            results[name] = describe(elapsed, DELTAS + 5)
            results[name]["final_message"] = message.to_dict()
    results["requests"] = requests
    return results


cold_start = time.perf_counter()
cold_event = construct_type(type_=RawMessageStreamEvent, value=DELTA)
cold_seconds = time.perf_counter() - cold_start
assert isinstance(cold_event, RawContentBlockDeltaEvent)

result = {
    "python": sys.version,
    "platform": platform.platform(),
    "pydantic": pydantic.__version__,
    "anthropic": anthropic.__version__,
    "models_file": anthropic._models.__file__,
    "models_sha256": hashlib.sha256(Path(anthropic._models.__file__).read_bytes()).hexdigest(),
    "repeats": REPEATS,
    "delta_events_per_stream": DELTAS,
    "decode_calls_per_repeat": DECODES,
    "cold_first_decode_seconds": cold_seconds,
    "decode": decode(),
    "large_events": large_events(),
    "streams": anyio.run(streams),
}
Path(sys.argv[1]).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"pydantic": result["pydantic"], "decode_us": result["decode"]["median_us_per_event"], "stable_stream_us": result["streams"]["stable"]["median_us_per_event"], "beta_stream_us": result["streams"]["beta"]["median_us_per_event"]}, indent=2))
