"""Exact-argument, one-use approval grants for sensitive local actions."""

from __future__ import annotations

import hashlib
import json
import secrets
import threading
import time
import copy
from dataclasses import dataclass
from typing import Any


class ApprovalError(PermissionError):
    """Raised when an approval is absent, expired, changed, or already used."""


@dataclass(frozen=True)
class ApprovalRequest:
    """Display-safe request description for the approval UI."""

    request_id: str
    tool_name: str
    arguments: dict[str, Any]
    expires_in_seconds: float


@dataclass(frozen=True)
class _ApprovalGrant:
    tool_name: str
    argument_digest: str
    expires_at: float


class ApprovalManager:
    """Issue and consume short-lived approvals for exact normalized arguments."""

    def __init__(
        self,
        *,
        ttl_seconds: float = 120.0,
        clock: Any = time.monotonic,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("Approval TTL must be greater than zero.")
        self.ttl_seconds = ttl_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._grants: dict[str, _ApprovalGrant] = {}
        self._requests: dict[str, _ApprovalGrant] = {}

    def prepare_request(
        self, tool_name: str, normalized_arguments: dict[str, Any]
    ) -> ApprovalRequest:
        """Create an expiring confirmation request, but not an execution token."""
        if not tool_name.strip():
            raise ValueError("tool_name cannot be empty.")
        request_id = secrets.token_urlsafe(32)
        grant = _ApprovalGrant(
            tool_name=tool_name,
            argument_digest=self._digest(normalized_arguments),
            expires_at=self._clock() + self.ttl_seconds,
        )
        with self._lock:
            self._discard_expired()
            self._requests[request_id] = grant
        return ApprovalRequest(
            request_id=request_id,
            tool_name=tool_name,
            arguments=copy.deepcopy(normalized_arguments),
            expires_in_seconds=self.ttl_seconds,
        )

    def confirm_request(self, request_id: str) -> str:
        """Issue a one-use token for the exact stored request after user confirmation."""
        with self._lock:
            grant = self._requests.pop(request_id, None)
            if grant is None:
                raise ApprovalError("Approval request is missing or expired.")
            if self._clock() >= grant.expires_at:
                raise ApprovalError("The approval request has expired.")
            token = secrets.token_urlsafe(32)
            self._grants[token] = _ApprovalGrant(
                tool_name=grant.tool_name,
                argument_digest=grant.argument_digest,
                expires_at=self._clock() + self.ttl_seconds,
            )
            return token

    def consume(
        self,
        token: str,
        tool_name: str,
        normalized_arguments: dict[str, Any],
    ) -> None:
        """Validate and consume a matching approval grant exactly once."""
        with self._lock:
            grant = self._grants.pop(token, None)
            if grant is None:
                raise ApprovalError("A valid user approval is required.")
            if self._clock() >= grant.expires_at:
                raise ApprovalError("The user approval has expired.")
            if grant.tool_name != tool_name:
                raise ApprovalError(
                    "The approval is for a different operation."
                )
            if not secrets.compare_digest(
                grant.argument_digest, self._digest(normalized_arguments)
            ):
                raise ApprovalError(
                    "Operation arguments changed after user approval."
                )

    def _discard_expired(self) -> None:
        now = self._clock()
        for token in [
            token
            for token, grant in self._grants.items()
            if now >= grant.expires_at
        ]:
            self._grants.pop(token, None)
        for request_id in [
            request_id
            for request_id, grant in self._requests.items()
            if now >= grant.expires_at
        ]:
            self._requests.pop(request_id, None)

    @staticmethod
    def _digest(arguments: dict[str, Any]) -> str:
        canonical = json.dumps(
            arguments,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
