"""In-process reference implementation of :class:`ApprovalStore`.

**This implementation INTENTIONALLY VIOLATES the persistence + cross-instance
contract.** It exists for two reasons:

1. Documenting the gap: the contract test suite at
   ``tests/tools/test_approval_store_contract.py`` runs against this store
   to demonstrate which invariants in-process storage cannot satisfy.
   Persistence + cross-instance-atomicity tests will fail. That failure
   is intentional and is the proof that the contract is meaningful.

2. Process-local testing: when a test genuinely doesn't care about
   persistence (e.g. unit-testing the gateway's call into ``submit``),
   instantiating an ``InMemoryApprovalStore`` is cheaper than spinning
   up SQLite.

**Do NOT use in production.** Wire ``tools.approval_store_sqlite.SqliteApprovalStore``
(added in a follow-up commit) for any real gateway.
"""

from __future__ import annotations

import threading
import time
from dataclasses import replace
from typing import Optional, Tuple

from tools.approval_store import (
    ApprovalProposal,
    ApprovalStore,
    ApprovalStoreError,
)


class InMemoryApprovalStore:
    """Thread-safe dict-backed approval store. Process-bound by design.

    Each instance has its own internal dict — two instances DO NOT share
    state, even if you give them the same identifier. This is the
    deliberate failure mode for cross-instance contract tests.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proposals: dict[str, ApprovalProposal] = {}
        # (verb, target) → (expires_at, reason, created_at).
        # In-memory mirror of gateway_approval_lockouts.
        self._lockouts: dict[Tuple[str, str], Tuple[float, str, float]] = {}

    # ----- Lifecycle -----

    def submit(self, proposal: ApprovalProposal) -> None:
        with self._lock:
            if proposal.approval_id in self._proposals:
                raise ValueError(
                    f"approval_id collision: {proposal.approval_id!r} "
                    "already exists in this store instance"
                )
            self._proposals[proposal.approval_id] = proposal

    def get(self, approval_id: str) -> Optional[ApprovalProposal]:
        with self._lock:
            return self._proposals.get(approval_id)

    def consume(self, approval_id: str, *, consumed_by: str,
                now: Optional[float] = None) -> Optional[ApprovalProposal]:
        ts = now if now is not None else time.time()
        with self._lock:
            proposal = self._proposals.get(approval_id)
            if proposal is None:
                return None
            if proposal.status != "pending":
                return None
            if proposal.is_expired_at(ts):
                self._proposals[approval_id] = proposal.with_status("expired")
                return None
            new = proposal.with_status(
                "consumed", consumed_by=consumed_by, consumed_at=ts,
            )
            self._proposals[approval_id] = new
            return new

    def deny(self, approval_id: str, *, denied_by: str,
             reason: Optional[str] = None,
             now: Optional[float] = None) -> bool:
        ts = now if now is not None else time.time()
        with self._lock:
            proposal = self._proposals.get(approval_id)
            if proposal is None or proposal.status not in {"pending", "pending_confirm"}:
                return False
            if proposal.is_expired_at(ts):
                # Already past TTL — call it expired, not denied. If
                # was pending_confirm, stamp the expiry reason.
                expired_reason = (
                    "expired_no_confirm"
                    if proposal.status == "pending_confirm"
                    else None
                )
                self._proposals[approval_id] = replace(
                    proposal, status="expired",
                    terminal_reason=expired_reason,
                )
                return False
            self._proposals[approval_id] = replace(
                proposal, status="denied",
                consumed_by=denied_by, consumed_at=ts,
                terminal_reason=reason,
            )
            return True

    def mark_post_consume(self, approval_id: str, *, executed: bool,
                          reason: Optional[str] = None,
                          now: Optional[float] = None) -> bool:
        ts = now if now is not None else time.time()
        new_status = "executed" if executed else "blocked_after_consume"
        with self._lock:
            proposal = self._proposals.get(approval_id)
            if proposal is None or proposal.status != "consumed":
                return False
            from dataclasses import replace
            self._proposals[approval_id] = replace(
                proposal,
                execution_status=new_status,
                execution_reason=reason,
                execution_recorded_at=ts,
            )
            return True

    def expire_due(self, now: Optional[float] = None) -> int:
        ts = now if now is not None else time.time()
        count = 0
        with self._lock:
            for aid, proposal in list(self._proposals.items()):
                if proposal.is_expired_at(ts):
                    if proposal.status == "pending":
                        self._proposals[aid] = proposal.with_status("expired")
                        count += 1
                    elif proposal.status == "pending_confirm":
                        self._proposals[aid] = replace(
                            proposal, status="expired",
                            terminal_reason="expired_no_confirm",
                        )
                        count += 1
        return count

    # ----- Tier 2 critical: text-confirm transitions -----

    def arm_text_confirm(self, approval_id: str, *,
                         event_id: str,
                         origin_server_ts_ms: int,
                         now: Optional[float] = None) -> bool:
        ts = now if now is not None else time.time()
        with self._lock:
            proposal = self._proposals.get(approval_id)
            if proposal is None or proposal.status != "pending":
                return False
            if proposal.is_expired_at(ts):
                self._proposals[approval_id] = proposal.with_status("expired")
                return False
            self._proposals[approval_id] = replace(
                proposal, status="pending_confirm",
                approval_event_id=event_id,
                approval_event_ts_ms=origin_server_ts_ms,
            )
            return True

    def register_invalid_attempt(self, approval_id: str, *,
                                 limit: int,
                                 now: Optional[float] = None,
                                 ) -> Tuple[int, bool]:
        with self._lock:
            proposal = self._proposals.get(approval_id)
            if proposal is None or proposal.status != "pending_confirm":
                return (0, False)
            new_count = proposal.invalid_attempts + 1
            self._proposals[approval_id] = replace(
                proposal, invalid_attempts=new_count,
            )
            return (new_count, new_count >= limit)

    def mark_blocked(self, approval_id: str, *,
                     reason: str,
                     now: Optional[float] = None) -> bool:
        ts = now if now is not None else time.time()
        with self._lock:
            proposal = self._proposals.get(approval_id)
            if proposal is None or proposal.status not in {"pending", "pending_confirm"}:
                return False
            self._proposals[approval_id] = replace(
                proposal, status="blocked",
                terminal_reason=reason,
                consumed_at=ts,
            )
            return True

    def consume_confirmed(self, approval_id: str, *,
                          consumed_by: str,
                          now: Optional[float] = None
                          ) -> Optional[ApprovalProposal]:
        ts = now if now is not None else time.time()
        with self._lock:
            proposal = self._proposals.get(approval_id)
            if proposal is None or proposal.status != "pending_confirm":
                return None
            if proposal.is_expired_at(ts):
                self._proposals[approval_id] = replace(
                    proposal, status="expired",
                    terminal_reason="expired_no_confirm",
                )
                return None
            new = replace(
                proposal, status="consumed",
                consumed_by=consumed_by, consumed_at=ts,
            )
            self._proposals[approval_id] = new
            return new

    # ----- Tier 2 critical: lockout CRUD -----

    def check_lockout(self, verb: str, target: str, *,
                      now: Optional[float] = None
                      ) -> Optional[Tuple[float, str]]:
        ts = now if now is not None else time.time()
        with self._lock:
            row = self._lockouts.get((verb, target))
            if row is None:
                return None
            expires_at, reason, _created_at = row
            if expires_at <= ts:
                return None
            return (expires_at, reason)

    def set_lockout(self, verb: str, target: str, *,
                    expires_at: float,
                    reason: str,
                    now: Optional[float] = None) -> None:
        ts = now if now is not None else time.time()
        with self._lock:
            existing = self._lockouts.get((verb, target))
            # Never shorten an active lockout — keep the later expiry.
            final_expires = (
                max(existing[0], expires_at) if existing else expires_at
            )
            self._lockouts[(verb, target)] = (final_expires, reason, ts)

    def clear_lockout(self, verb: str, target: str) -> bool:
        with self._lock:
            return self._lockouts.pop((verb, target), None) is not None


# Make a runtime-checkable conformance assertion explicit:
assert isinstance(InMemoryApprovalStore(), ApprovalStore), (
    "InMemoryApprovalStore must satisfy ApprovalStore Protocol"
)
