"""Strictly bounded OpenAI vision fallback for failed calibrated ReID."""

from __future__ import annotations

import json
import http.client
import socket
from dataclasses import dataclass
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class VlmIdentityDecision:
    same_identity: bool
    confidence: float
    reason: str = ""


class OpenAIVisionIdentityComparator:
    """Compare two detector-indicated entities through the Responses API."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gpt-4o-mini-2024-07-18",
        timeout_seconds: float = 15.0,
        transport: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("OpenAI API key is required")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self._transport = transport or self._request

    def compare(
        self,
        *,
        entity_kind: str,
        left_image_url: str,
        right_image_url: str,
        left_local_entity_id: str = "",
        right_local_entity_id: str = "",
    ) -> VlmIdentityDecision:
        payload = {
            "model": self.model,
            "max_output_tokens": 120,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                f"Determine whether the indicated {entity_kind} in "
                                "these two full camera frames is the same physical "
                                "identity. The solid yellow rectangle is the raw "
                                "detector box and may be clipped, delayed, too small, "
                                "or slightly misaligned. The larger dashed cyan "
                                "rectangle is an approximate search region. Treat both "
                                "as pointers, not hard object boundaries, and inspect "
                                "the entire frame for the most plausible nearby target. "
                                "Compare stable appearance cues such as body style, "
                                "color, proportions, windows, wheels, trim, markings, "
                                "and visible damage. Account for viewpoint, scale, "
                                "lighting, motion blur, and partial occlusion. Do not "
                                "claim a match from color alone. Set confidence "
                                "to the estimated probability that the targets are "
                                "the same physical identity, not confidence in the "
                                "boolean answer. Set same_identity=true when that "
                                "probability is at least 0.5; the caller applies its "
                                "configured acceptance threshold afterward. "
                                "Return only "
                                'JSON: {\"same_identity\": boolean, '
                                '\"confidence\": number from 0 to 1, '
                                '\"reason\": string}.'
                            ),
                        },
                        {
                            "type": "input_image",
                            "image_url": left_image_url,
                            "detail": "high",
                        },
                        {
                            "type": "input_image",
                            "image_url": right_image_url,
                            "detail": "high",
                        },
                    ],
                }
            ],
        }
        response = self._transport(payload)
        text = _response_text(response)
        document = _json_object(text)
        return VlmIdentityDecision(
            same_identity=bool(document.get("same_identity", False)),
            confidence=max(0.0, min(1.0, float(document.get("confidence", 0.0)))),
            reason=str(document.get("reason", ""))[:240],
        )

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        last_error: Exception | None = None
        # Docker's embedded DNS can briefly fail while a large replay bundle
        # starts dozens of services.  A single lookup failure must not turn a
        # valid identity fallback into a semantic false negative.  Keep the
        # retry count small and inside the existing bounded VLM invocation.
        for attempt in range(5):
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                if isinstance(exc, HTTPError) and 400 <= exc.code < 500:
                    break
                if attempt < 4:
                    time.sleep(min(2**attempt, 8))
        raise RuntimeError(f"OpenAI VLM ReID request failed: {last_error}") from last_error


class RemoteVisionIdentityComparator:
    """Secret-free client for the dedicated cloud-side VLM proxy."""

    def __init__(
        self,
        *,
        endpoint: str,
        timeout_seconds: float = 20.0,
        transport: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        if not endpoint.startswith(("http://", "https://", "unix://")):
            raise ValueError("VLM proxy endpoint must be HTTP(S) or a Unix socket")
        self._unix_socket = (
            endpoint.removeprefix("unix://") if endpoint.startswith("unix://") else None
        )
        if self._unix_socket is not None:
            if not self._unix_socket.startswith("/"):
                raise ValueError("VLM proxy Unix socket path must be absolute")
            self.endpoint = "/v1/compare"
        else:
            self.endpoint = endpoint.rstrip("/") + "/v1/compare"
        self.timeout_seconds = timeout_seconds
        self.model = "hosted-vlm-proxy"
        self._transport = transport or self._request
        self._run_id = "unscoped"
        self._invocation = 0

    def set_run_id(self, run_id: str | None) -> None:
        self._run_id = str(run_id or "unscoped")
        self._invocation = 0

    def compare(
        self,
        *,
        entity_kind: str,
        left_image_url: str,
        right_image_url: str,
        left_local_entity_id: str = "",
        right_local_entity_id: str = "",
    ) -> VlmIdentityDecision:
        self._invocation += 1
        response = self._transport(
            {
                "schema_version": "fable.hosted_vlm_request.v1",
                "run_id": self._run_id,
                "invocation_id": f"{self._run_id}:{self._invocation}",
                "entity_kind": entity_kind,
                "left_image_url": left_image_url,
                "right_image_url": right_image_url,
                "left_local_entity_id": left_local_entity_id,
                "right_local_entity_id": right_local_entity_id,
            }
        )
        if response.get("schema_version") != "fable.hosted_vlm_response.v1":
            raise RuntimeError("hosted VLM proxy returned an unsupported schema")
        return VlmIdentityDecision(
            same_identity=bool(response.get("same_identity", False)),
            confidence=max(0.0, min(1.0, float(response.get("confidence", 0)))),
            reason=str(response.get("reason", ""))[:240],
        )

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._unix_socket is not None:
            return self._request_unix(payload)
        request = Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(5):
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                if isinstance(exc, HTTPError) and 400 <= exc.code < 500:
                    break
                if attempt < 4:
                    time.sleep(min(2**attempt, 8))
        raise RuntimeError(f"hosted VLM proxy request failed: {last_error}") from last_error

    def _request_unix(self, payload: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        body = json.dumps(payload).encode("utf-8")
        for attempt in range(5):
            connection = _UnixHTTPConnection(
                self._unix_socket,
                timeout=self.timeout_seconds,
            )
            try:
                connection.request(
                    "POST",
                    self.endpoint,
                    body=body,
                    headers={"Content-Type": "application/json"},
                )
                response = connection.getresponse()
                document = json.loads(response.read().decode("utf-8"))
                if response.status >= 400:
                    raise RuntimeError(
                        f"hosted VLM proxy returned HTTP {response.status}: {document}"
                    )
                return document
            except (OSError, http.client.HTTPException, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt < 4:
                    time.sleep(min(2**attempt, 8))
            finally:
                connection.close()
        raise RuntimeError(f"hosted VLM proxy request failed: {last_error}") from last_error


class _UnixHTTPConnection(http.client.HTTPConnection):
    """Minimal HTTP client transported over an allowlisted Unix socket."""

    def __init__(self, socket_path: str, *, timeout: float) -> None:
        super().__init__("localhost", timeout=timeout)
        self.socket_path = socket_path

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.socket_path)

def _response_text(response: dict[str, Any]) -> str:
    for output in response.get("output", ()):
        if not isinstance(output, dict):
            continue
        for content in output.get("content", ()):
            if isinstance(content, dict) and content.get("type") == "output_text":
                return str(content.get("text") or "")
    raise RuntimeError("OpenAI VLM ReID response contained no output text")


def _json_object(value: str) -> dict[str, Any]:
    text = value.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    document = json.loads(text)
    if not isinstance(document, dict):
        raise RuntimeError("OpenAI VLM ReID response was not a JSON object")
    return document
