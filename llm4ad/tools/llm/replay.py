# SPDX-License-Identifier: BSD-2-Clause
"""Opt-in raw LLM recording and strict offline response replay.

These adapters preserve the existing :class:`llm4ad.base.LLM` interface. Exact
search-trajectory replay requires deterministic evaluation and serial sampling.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from threading import Condition, Lock
import time

from llm4ad.base import LLM
from llm4ad.base.errors import ReplayMismatchError, ReplayExhaustedError, ReplayBackendError
from llm4ad.tools.reproducibility import redact


def _request(prompt, args, kwargs):
    # Round-trip to a JSON value: unsupported objects fail rather than being
    # converted to repr(), which may hide nondeterminism or expose credentials.
    value = {"prompt": prompt, "args": args, "kwargs": kwargs}
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return redact(json.loads(encoded)), hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _stored_hash(request):
    encoded = json.dumps(request, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class RecordingLLM(LLM):
    """Wrap an LLM and durably record requests/responses in a new JSONL file.

    Args:
        llm: Backend implementing ``draw_sample`` and ``close``.
        path: A new file; existing recordings are never overwritten or appended.
        model_metadata: Explicit model/revision/generation metadata, without keys.
        run_metadata: Optional task, seed, dataset and budget configuration.
    """
    def __init__(self, llm: LLM, path, *, model_metadata=None, run_metadata=None):
        super().__init__(do_auto_trim=llm.do_auto_trim, debug_mode=llm.debug_mode)
        if ((model_metadata is not None and not isinstance(model_metadata, dict))
                or (run_metadata is not None and not isinstance(run_metadata, dict))):
            raise TypeError("Model and run metadata must be dictionaries")
        header = {"schema_version": 2, "type": "header", "backend": type(llm).__name__,
                  "do_auto_trim": llm.do_auto_trim, "model": redact(model_metadata or {}),
                  "run": redact(run_metadata or {})}
        # Validate before creating a file that could otherwise be left empty.
        json.dumps(header, allow_nan=False)
        self._llm = llm
        self._condition = Condition()
        self._sequence = 0
        self._active = 0
        self._closing = False
        self.path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        self._stream = os.fdopen(descriptor, "w", encoding="utf-8")
        try:
            self._append(header)
        except BaseException:
            self._stream.close()
            Path(path).unlink()
            raise

    def _append(self, record):
        self._stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        self._stream.flush()
        os.fsync(self._stream.fileno())

    def draw_sample(self, prompt, *args, **kwargs):
        request, request_hash = _request(prompt, args, kwargs)
        with self._condition:
            if self._closing or self._stream.closed:
                raise ValueError("RecordingLLM is closed")
            sequence = self._sequence
            self._sequence += 1
            self._active += 1
        started = time.monotonic()
        record = {"type": "sample", "sequence": sequence,
                  "request": request, "request_sha256": request_hash,
                  "stored_request_sha256": _stored_hash(request)}
        try:
            self._llm.debug_mode = self.debug_mode
            response = self._llm.draw_sample(prompt, *args, **kwargs)
            if not isinstance(response, str):
                raise TypeError("The LLM backend must return a string")
            record["response"] = response
            return response
        except BaseException as error:
            # Exception text may embed credentials; record its type only.
            record["error_type"] = type(error).__name__
            raise
        finally:
            record["elapsed_seconds"] = time.monotonic() - started
            with self._condition:
                try:
                    self._append(record)
                finally:
                    self._active -= 1
                    self._condition.notify_all()

    def close(self):
        """Close the journal and underlying backend after outstanding calls finish."""
        with self._condition:
            if self._stream.closed:
                return
            self._closing = True
            while self._active:
                self._condition.wait()
            if self._stream.closed:
                return
            try:
                self._append({"type": "footer", "sample_count": self._sequence})
            finally:
                self._stream.close()
                self._llm.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class ReplayLLM(LLM):
    """Replay a complete recording in request-start order, without any API client.

    Changed prompts/arguments, corrupt journals and exhausted recordings raise
    explicit errors. Replay never silently falls back to online generation.
    """
    def __init__(self, path):
        with open(path, encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream if line.strip()]
        if (not records or not isinstance(records[0], dict) or records[0].get("type") != "header"
                or type(records[0].get("schema_version")) is not int
                or records[0]["schema_version"] not in (1, 2)):
            raise ValueError("Unsupported or missing recording header")
        self.schema_version = records[0]["schema_version"]
        if not isinstance(records[0].get("do_auto_trim"), bool):
            raise ValueError("Invalid recording trim setting")
        super().__init__(do_auto_trim=records[0]["do_auto_trim"])
        self.model_metadata = records[0].get("model", {})
        self.run_metadata = records[0].get("run", {})
        if not isinstance(self.model_metadata, dict) or not isinstance(self.run_metadata, dict):
            raise ValueError("Invalid recording metadata")
        samples = records[1:]
        if self.schema_version == 2:
            footer = samples[-1] if samples else None
            if (not isinstance(footer, dict) or footer.get("type") != "footer"
                    or type(footer.get("sample_count")) is not int
                    or footer["sample_count"] != len(samples) - 1):
                raise ValueError("Recording is incomplete or has an invalid footer")
            samples = samples[:-1]
        if any(not isinstance(row, dict) or type(row.get("sequence")) is not int for row in samples):
            raise ValueError("Invalid sample sequence")
        self._records = sorted(samples, key=lambda row: row["sequence"])
        for sequence, record in enumerate(self._records):
            if record.get("type") != "sample" or record["sequence"] != sequence:
                raise ValueError("Recording contains missing or duplicate samples")
            request = record.get("request")
            if (not isinstance(request, dict) or set(request) != {"prompt", "args", "kwargs"}
                    or not isinstance(request["args"], list) or not isinstance(request["kwargs"], dict)):
                raise ValueError("Invalid recorded request")
            stored_hash = record.get('stored_request_sha256') if self.schema_version == 2 else record.get('request_sha256')
            if _stored_hash(request) != stored_hash:
                raise ValueError("Recording request checksum mismatch")
            fingerprint = record.get('request_sha256')
            if (not isinstance(fingerprint, str) or len(fingerprint) != 64
                    or any(c not in '0123456789abcdef' for c in fingerprint)):
                raise ValueError("Invalid request fingerprint")
            if ("response" in record) == ("error_type" in record):
                raise ValueError("Sample must contain either a response or an error")
            if "response" in record and not isinstance(record["response"], str):
                raise ValueError("Recorded response must be a string")
            if "error_type" in record and not isinstance(record["error_type"], str):
                raise ValueError("Recorded error type must be a string")
        self._position = 0
        self._lock = Lock()

    @property
    def remaining(self):
        """Number of unconsumed requests."""
        return len(self._records) - self._position

    def draw_sample(self, prompt, *args, **kwargs):
        request, request_hash = _request(prompt, args, kwargs)
        if self.schema_version == 1:
            request_hash = _stored_hash(request)
        with self._lock:
            if self._position >= len(self._records):
                raise ReplayExhaustedError("LLM recording exhausted")
            record = self._records[self._position]
            if request_hash != record["request_sha256"] or request != record["request"]:
                raise ReplayMismatchError(f"Replay request mismatch at sample {self._position}")
            self._position += 1
            if "error_type" in record:
                raise ReplayBackendError(f"Recorded backend failure: {record['error_type']}")
            return record["response"]
