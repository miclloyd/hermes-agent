"""Approval storage abstraction for gateway-side command approvals.

This module defines the storage contract that any approval-store implementation
must satisfy to guarantee the security invariants documented at:

  - .plans/hermes-gateway-approval-safety.md  (if present)
  - or the inline summary below.

Invariants (non-negotiable; tested in tests/tools/test_approval_store_contract.py):

1. **Pinned policy** — risk/policy decision is computed and persisted at
   *proposal creation* time. Approval execution must consume the pinned
   payload; it must never re-classify risk from the command string.

2. **Process-durable** — a proposal submitted by one gateway process must be
   visible (and consumable) by another gateway process. Pure in-process
   storage does NOT satisfy this contract.

3. **Atomic consume, exactly once** — two concurrent ``consume(approval_id)``
   calls — same process or different processes — must result in *exactly
   one* returning the proposal. The other must return ``None``. Best-effort
   JSON+sidecar-lockfile schemes do NOT satisfy this; a real transaction
   (e.g. SQLite ``BEGIN IMMEDIATE``) is required.

4. **No silent downgrade** — a denied proposal cannot later be consumed.
   An expired proposal cannot be consumed. A consumed proposal cannot be
   consumed again.

Implementations live in sibling modules:

  - ``tools.approval_store_memory``: in-process reference (FAILS the
    persistence + cross-instance tests by design; useful only for
    process-local testing / documenting the contract gap).
  - ``tools.approval_store_sqlite``: production transactional impl backed
    by the existing Hermes ``state.db``.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field, replace
from typing import Optional, Protocol, Tuple, runtime_checkable


# ---------------------------------------------------------------------------
# Proposal model — what gets persisted per pending approval.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ApprovalProposal:
    """One pending (or terminal) command-approval proposal.

    Frozen because the persisted record is immutable once submitted —
    state transitions create a new dataclass instance via :func:`replace`
    rather than mutating in place. The store is responsible for
    durable serialization of these instances.

    Fields are grouped by purpose to make the pinned-policy distinction
    obvious to reviewers.
    """

    # --- Identity & lifecycle ---
    approval_id: str
    created_at: float                  # epoch seconds
    expires_at: Optional[float] = None
    # pending → (pending_confirm if tier2_critical) → consumed | denied | expired | blocked
    status: str = "pending"
    consumed_at: Optional[float] = None
    consumed_by: Optional[str] = None
    # Granular audit reason for terminal transitions. Populated on
    # denied/expired/blocked. ``status`` alone cannot distinguish e.g.
    # "user typed DENY" from "user typed wrong nonce 3x" — both are
    # non-consume terminal but mean very different things in incident
    # review. See _TERMINAL_REASONS for the enum.
    terminal_reason: Optional[str] = None

    # --- Requester context ---
    session_key: str = ""
    requester: Optional[str] = None    # platform user id / handle

    # --- Command to execute ---
    command: str = ""
    cwd: Optional[str] = None
    backend: Optional[str] = None      # e.g. "bash", "execute_code"
    env_overrides: dict = field(default_factory=dict)

    # --- Pinned policy (frozen at proposal creation, see invariant 1) ---
    risk_level: str = "low"            # low|medium|high|tier2_critical
    risk_reason: str = ""
    policy_decision: str = "needs_approval"   # allow|needs_approval|deny
    policy_version: Optional[str] = None
    requires_explicit_approval: bool = True
    default_decision: str = "deny"     # MUST be "deny" for high-risk / tier2_critical

    # --- UX context ---
    diff_text: Optional[str] = None
    diff_summary: Optional[str] = None
    display_text: str = ""             # exact text shown to user (or reconstruct hints)

    # --- Tier 2 critical (text-confirm) pinned policy ---
    # When ``requires_text_confirm`` is True, the approval cannot be
    # consumed by /approve alone; the requester must type a normalized
    # phrase ``<VERB> <TARGET> <NONCE>`` into the same channel. The
    # phrase parts are pinned at submit; only ``approval_event_*``
    # (the Matrix server's identifiers for the approval message) are
    # attached later via the store's ``arm_text_confirm`` transition.
    requires_text_confirm: bool = False
    nonce: Optional[str] = None        # 4-digit string; pinned at submit
    verb: Optional[str] = None         # e.g. "restart-tier2"; pinned at submit
    target: Optional[str] = None       # e.g. "vaultwarden"; pinned at submit

    # --- Tier 2 mutable post-submit columns ---
    # Set by ``arm_text_confirm`` once the gateway has sent the approval
    # message and the Matrix server has returned event_id + origin_ts.
    # TTL is anchored to ``approval_event_ts_ms`` (server clock), not
    # to ``created_at`` (gateway clock) — clients see the server time.
    approval_event_id: Optional[str] = None
    approval_event_ts_ms: Optional[int] = None
    # Incremented by ``register_invalid_attempt`` on each wrong phrase.
    # ``mark_blocked(reason='invalid_confirm_limit')`` is triggered when
    # this hits the limit; the proposal does NOT auto-consume on typo.
    invalid_attempts: int = 0

    # --- Post-consume execution outcome (audit-distinct from consume) ---
    # status='consumed' means "user clicked /approve and the store row was
    # atomically transitioned". It does NOT mean the command actually ran.
    # The Phase 3 runtime guard may block execution AFTER consume, and the
    # waiter may be gone at gateway restart. These fields record that
    # distinction so retrospective incident review can answer "did this
    # approval lead to an execution?".
    execution_status: str = "not_started"  # not_started | executed | blocked_after_consume
    execution_reason: Optional[str] = None   # e.g. 'phase3_runtime_stricter', 'orphan_no_waiter'
    execution_recorded_at: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate pinned-policy completeness at construction time.

        Per spec hardening: a proposal that lacks the fields the executor
        needs to make a safety decision MUST NOT exist. Failing early —
        before submit — surfaces malformed proposals at the call-site
        instead of pushing the failure into the consume/execute path.

        Defense in depth: the execution-thread Phase 3 guard still
        treats unknown risk_level as max-risk (ranks at 999), so even
        if this validation is bypassed, runtime guard catches it.
        """
        errors = []
        if not self.approval_id:
            errors.append("approval_id is required (non-empty string)")
        if not self.created_at or self.created_at <= 0:
            errors.append("created_at must be > 0 (epoch seconds)")
        if not self.session_key:
            errors.append("session_key is required (non-empty)")
        if not self.command:
            errors.append("command is required (non-empty)")
        if self.risk_level not in {"low", "medium", "high", "tier2_critical"}:
            errors.append(
                f"risk_level must be one of low/medium/high/tier2_critical, "
                f"got {self.risk_level!r}"
            )
        if not self.risk_reason:
            errors.append("risk_reason is required (non-empty)")
        if self.policy_decision not in {"allow", "needs_approval", "deny"}:
            errors.append(
                f"policy_decision must be one of allow/needs_approval/deny, "
                f"got {self.policy_decision!r}"
            )
        if self.default_decision not in {"allow", "deny"}:
            errors.append(
                f"default_decision must be 'allow' or 'deny', "
                f"got {self.default_decision!r}"
            )
        # High-risk and tier2_critical MUST default to deny — non-negotiable.
        if self.risk_level in {"high", "tier2_critical"} and self.default_decision != "deny":
            errors.append(
                f"{self.risk_level} proposals MUST have default_decision='deny', "
                f"got {self.default_decision!r}"
            )

        # Tier 2 critical implies text-confirm; text-confirm implies
        # pinned phrase parts. Without these the orchestrator cannot
        # build or match the phrase, so submit must refuse the proposal.
        if self.risk_level == "tier2_critical" and not self.requires_text_confirm:
            errors.append(
                "tier2_critical risk_level requires requires_text_confirm=True"
            )
        if self.requires_text_confirm:
            if not self.nonce:
                errors.append(
                    "requires_text_confirm=True requires non-empty nonce"
                )
            elif not (self.nonce.isdigit() and len(self.nonce) == 4):
                errors.append(
                    f"nonce must be a 4-digit string, got {self.nonce!r}"
                )
            if not self.verb:
                errors.append(
                    "requires_text_confirm=True requires non-empty verb"
                )
            if not self.target:
                errors.append(
                    "requires_text_confirm=True requires non-empty target"
                )

        if errors:
            raise ValueError(
                "ApprovalProposal validation failed:\n  - " +
                "\n  - ".join(errors)
            )

    def is_terminal(self) -> bool:
        """True once the proposal has reached any non-pending state.

        ``pending_confirm`` is NOT terminal — the proposal is still
        actively waiting for the requester to type the phrase.
        """
        return self.status in {"approved", "denied", "expired", "consumed", "blocked"}

    def is_expired_at(self, now: float) -> bool:
        return self.expires_at is not None and self.expires_at <= now

    def with_status(self, status: str, *, consumed_by: Optional[str] = None,
                    consumed_at: Optional[float] = None) -> "ApprovalProposal":
        return replace(
            self,
            status=status,
            consumed_by=consumed_by if consumed_by is not None else self.consumed_by,
            consumed_at=consumed_at if consumed_at is not None else self.consumed_at,
        )

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Storage contract.
# ---------------------------------------------------------------------------


class ApprovalStoreError(Exception):
    """Raised on store-internal failures (DB unavailable, corrupt payload, etc.).

    Per fail-closed invariant, callers MUST treat raised errors as
    *no execution*. They must NEVER catch this and silently proceed.
    """


@runtime_checkable
class ApprovalStore(Protocol):
    """Storage contract for command approval proposals.

    Implementations must guarantee the four invariants in this module's
    docstring. The contract test suite at
    ``tests/tools/test_approval_store_contract.py`` enforces these.
    """

    # ----- Lifecycle -----

    def submit(self, proposal: ApprovalProposal) -> None:
        """Persist *proposal* in ``pending`` status.

        Must be durable across process restarts. Must be visible to other
        store instances pointing at the same backing storage.

        Raises:
          ApprovalStoreError: on storage failure (the caller must treat
            this as 'proposal not accepted' — DO NOT execute the command).
          ValueError: if ``proposal.approval_id`` collides with an existing row.
        """
        ...

    def get(self, approval_id: str) -> Optional[ApprovalProposal]:
        """Return the proposal with that id, or ``None`` if missing.

        Read-only. Does not mutate state. Returns the proposal regardless
        of current status (caller is responsible for checking ``.status``
        if it cares about pending-only).
        """
        ...

    def consume(self, approval_id: str, *, consumed_by: str,
                now: Optional[float] = None) -> Optional[ApprovalProposal]:
        """Atomically transition pending → consumed and return the proposal.

        This is the **only** way to authorize execution. Implementations
        must guarantee:

        - Returns the *pinned* proposal (caller executes from this payload,
          never re-classifies).
        - Returns ``None`` if the proposal is missing, already consumed,
          denied, or expired (per ``expires_at`` vs ``now``).
        - When called concurrently from two threads or two processes on
          the same ``approval_id``, exactly one call returns a proposal;
          the other returns ``None``.

        After a successful consume, ``consumed_at`` and ``consumed_by``
        are set on the persisted row.
        """
        ...

    def deny(self, approval_id: str, *, denied_by: str,
             reason: Optional[str] = None,
             now: Optional[float] = None) -> bool:
        """Atomically transition (pending OR pending_confirm) → denied.

        Accepts both statuses so a /deny or ❌ reaction can terminate a
        tier-2-critical approval that's currently waiting for the text
        phrase, not just one that's pending the initial /approve.

        Args:
            denied_by: platform user id of the denier.
            reason: optional ``terminal_reason`` — typically
                ``'denied_explicit'`` when the deny was an active user
                action (vs an automatic termination handled elsewhere).
                Pass ``None`` to leave terminal_reason NULL.

        Returns True if the transition happened. Returns False if the
        proposal was already terminal or missing. A denied proposal
        cannot subsequently be consumed.
        """
        ...

    def expire_due(self, now: Optional[float] = None) -> int:
        """Mark all open proposals (pending OR pending_confirm) whose
        ``expires_at`` ≤ ``now`` as expired.

        For pending_confirm rows the transition also sets
        ``terminal_reason='expired_no_confirm'`` so incident review can
        distinguish "user never typed anything" from "user typed wrong
        3 times then we blocked them" (which uses ``mark_blocked``).

        Returns the number of rows transitioned. Implementations may run
        this opportunistically (lazy expiry inside ``consume``) instead of
        as a background job — but ``consume`` of an expired proposal MUST
        always return ``None`` regardless of whether this method has been
        called.
        """
        ...

    def mark_post_consume(self, approval_id: str, *, executed: bool,
                          reason: Optional[str] = None,
                          now: Optional[float] = None) -> bool:
        """Record the post-consume execution outcome.

        Called after ``consume`` succeeded and the caller knows whether
        the command actually executed (``executed=True``) or was blocked
        by a runtime guard / orphan / other post-consume failure
        (``executed=False`` + reason).

        Returns True if the row was updated, False if the approval_id is
        missing or not in 'consumed' status. Idempotent within the same
        outcome; calling twice with different outcomes overwrites (last
        writer wins) but in practice this should only ever be called
        once per consume.

        This is the audit signal that distinguishes "user approved
        AND command ran" from "user approved BUT execution was blocked".
        ``status`` stays 'consumed' either way — consume IS exactly-once.
        ``execution_status`` carries the outcome.
        """
        ...

    # ----- Tier 2 critical: text-confirm transitions -----
    #
    # These methods are atomic state transitions only. All policy/UX
    # logic (phrase normalisation, requester check, clock skew, TTL,
    # Matrix server-ts vs local-now sanity) lives in the orchestrator
    # in ``tools.approval``. The store's job is to be the consistent
    # source of truth, not to know what a nonce is for.

    def arm_text_confirm(self, approval_id: str, *,
                         event_id: str,
                         origin_server_ts_ms: int,
                         now: Optional[float] = None) -> bool:
        """Atomically transition pending → pending_confirm, attaching
        the Matrix server's identifiers for the approval message.

        Called by the gateway right after the approval prompt was sent
        and the platform (e.g. Matrix) returned ``event_id`` and
        ``origin_server_ts``. The TTL/clock-skew checks performed by the
        orchestrator are anchored to ``origin_server_ts_ms``; the row
        records it once and never updates again.

        Returns True if the transition happened (proposal was in
        ``pending`` and not expired). Returns False if missing, already
        armed (idempotent re-call returns False, not True — the caller
        can read the row to see current state), denied, expired, or
        consumed. The gateway treats False as "approval already
        terminal — do not wait for confirm".
        """
        ...

    def register_invalid_attempt(self, approval_id: str, *,
                                 limit: int,
                                 now: Optional[float] = None
                                 ) -> Tuple[int, bool]:
        """Atomically increment invalid_attempts on a pending_confirm row.

        Returns ``(new_count, exceeded)`` where ``exceeded`` is True iff
        ``new_count >= limit``. On exceeded, the caller is expected to
        call ``mark_blocked(reason='invalid_confirm_limit')`` plus
        ``set_lockout(verb, target, …)``. Sequencing is up to the
        orchestrator; the store does not auto-cascade so backends can
        keep transactions short.

        Returns ``(0, False)`` if the proposal is missing or not in
        ``pending_confirm`` (e.g. already consumed, denied, expired).
        This is fail-safe: the orchestrator routes the False back to the
        user as "approval not awaiting confirm".
        """
        ...

    def mark_blocked(self, approval_id: str, *,
                     reason: str,
                     now: Optional[float] = None) -> bool:
        """Atomically transition pending_confirm → blocked, setting
        ``terminal_reason``.

        Reason must be a member of :data:`_TERMINAL_REASONS` (caller's
        responsibility — the store does not validate; orchestrator owns
        the enum). Returns True if the transition happened, False if
        missing or already terminal.
        """
        ...

    def consume_confirmed(self, approval_id: str, *,
                          consumed_by: str,
                          now: Optional[float] = None
                          ) -> Optional[ApprovalProposal]:
        """Atomically transition pending_confirm → consumed.

        Tier-2-critical analogue of :meth:`consume` — same exactly-once
        guarantee, but the WHERE clause matches ``status='pending_confirm'``
        instead of ``'pending'``. The orchestrator only calls this AFTER
        it has verified the phrase, requester, nonce, and clock skew.

        Returns the consumed proposal on success, or ``None`` if the row
        is missing / in the wrong status / past TTL.
        """
        ...

    # ----- Tier 2 critical: per-(verb,target) lockout CRUD -----

    def check_lockout(self, verb: str, target: str, *,
                      now: Optional[float] = None
                      ) -> Optional[Tuple[float, str]]:
        """Return ``(expires_at, reason)`` if (verb, target) is locked
        out as of ``now``, else ``None``.

        Expired lockouts return ``None`` even if the row still exists —
        the store may opportunistically clear it but is not required to.
        The orchestrator calls this BEFORE creating a new approval; a
        non-None return is a hard fail-closed.
        """
        ...

    def set_lockout(self, verb: str, target: str, *,
                    expires_at: float,
                    reason: str,
                    now: Optional[float] = None) -> None:
        """Upsert a lockout row. If (verb, target) already has a row,
        the later expiry wins — never shorten an active lockout
        accidentally.

        Reason is opaque (typically ``invalid_confirm_limit``); the
        orchestrator decides what to write.
        """
        ...

    def clear_lockout(self, verb: str, target: str) -> bool:
        """Remove the lockout row for (verb, target).

        Returns True if a row was removed, False if none existed.
        Intended for operator recovery (``hermes-ctl clear-lockout``)
        and for test setup/teardown — NOT for routine post-TTL cleanup,
        which is implicit in ``check_lockout``'s expiry check.
        """
        ...


def now() -> float:
    """Default epoch-seconds clock. Tests override via ``now`` argument."""
    return time.time()


# ---------------------------------------------------------------------------
# Tier 2 critical state machine constants.
# ---------------------------------------------------------------------------

# Reason codes for terminal_reason. Stored as opaque strings on the
# proposal row; this set is the authoritative enum. Adding a new reason
# requires updating both this set and the auditor's reason-renderer.
_TERMINAL_REASONS = frozenset({
    # denied:
    "denied_explicit",          # ❌ reaction, DENY phrase, or /deny <id>
    # expired:
    "expired_no_confirm",       # pending_confirm passed TTL without confirm
    # blocked:
    "invalid_confirm_limit",    # too many wrong phrase attempts
    "clock_skew",               # |server_ts - local_now| exceeded threshold
    "lockout_violation",        # tried to submit while (verb,target) locked
    "store_failed",             # store could not persist (existing reason)
})


# Status values that allow consume/deny transitions (the "open" set).
# pending_confirm is open in addition to pending: the approval is armed
# and waiting for the text phrase, but a /deny or ❌ reaction may still
# terminate it.
_OPEN_STATUSES = frozenset({"pending", "pending_confirm"})
