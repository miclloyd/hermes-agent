"""End-to-end scenarios for the tier-2-critical text-confirm flow.

Exercises the orchestrator + store layers together with realistic
proposals, arming, and confirmation paths. Sister-file
``test_tier2_helpers.py`` covers pure helpers and the ``_await``
branches; this file covers the actual user-visible flows:

  - happy: arm + correct phrase → consumed
  - typo: register_invalid_attempt, no consume, status stays
  - 3 typos: blocked + (verb, target) lockout set
  - lockout: new submit on (verb, target) refused; other verb/target free
  - wrong sender: ignored_sender_mismatch, no state change
  - clock skew: blocked_clock_skew, proposal terminal
  - explicit deny: ❌-reaction-equivalent (process_explicit_deny)
  - explicit deny via DENY phrase: process_explicit_deny with source='text'
  - TTL expired: expired_no_confirm via expire_due
  - TTL anchored to origin_server_ts, not gateway clock

The Matrix-adapter routing (parse phrase → process_text_confirm) is
covered separately; here we call the orchestrator directly with
realistic args.
"""

from __future__ import annotations

import time
from typing import Optional

import pytest

from tools.approval import (
    TIER2_CLOCK_SKEW_THRESHOLD_MS,
    TIER2_INVALID_ATTEMPT_LIMIT,
    TIER2_LOCKOUT_SECONDS,
    TIER2_TTL_SECONDS,
    _await_gateway_decision,
    classify_tier2_critical,
    generate_tier2_nonce,
    process_explicit_deny,
    process_text_confirm,
    register_gateway_notify,
    set_default_approval_store,
    unregister_gateway_notify,
)
from tools.approval_store import ApprovalProposal
from tools.approval_store_memory import InMemoryApprovalStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_proposal(
    approval_id: str = "appr-e2e",
    *,
    session_key: str = "s-e2e",
    requester: str = "@requester:example",
    verb: str = "restart-tier2",
    target: str = "vaultwarden",
    nonce: str = "4821",
    created: Optional[float] = None,
    ttl_seconds: int = TIER2_TTL_SECONDS,
) -> ApprovalProposal:
    if created is None:
        created = time.time()
    return ApprovalProposal(
        approval_id=approval_id,
        created_at=created,
        expires_at=created + ttl_seconds,
        session_key=session_key,
        requester=requester,
        command=f"sudo hermes-ctl {verb} {target}",
        cwd="/",
        backend="bash",
        risk_level="tier2_critical",
        risk_reason="tier2-critical pilot",
        policy_decision="needs_approval",
        requires_explicit_approval=True,
        default_decision="deny",
        display_text="…",
        requires_text_confirm=True,
        nonce=nonce,
        verb=verb,
        target=target,
    )


@pytest.fixture
def store():
    s = InMemoryApprovalStore()
    prev = None
    from tools import approval as _appr
    prev = _appr.get_default_approval_store()
    set_default_approval_store(s)
    yield s
    set_default_approval_store(prev)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_correct_phrase_consumes(self, store):
        proposal = _build_proposal()
        store.submit(proposal)
        server_ts_ms = int(time.time() * 1000)
        store.arm_text_confirm(
            "appr-e2e",
            event_id="$evt-arm",
            origin_server_ts_ms=server_ts_ms,
        )

        outcome = process_text_confirm(
            "s-e2e", "appr-e2e",
            phrase="restart-tier2 vaultwarden 4821",
            confirmer="@requester:example",
            server_event_id="$evt-confirm",
            server_ts_ms=server_ts_ms + 5_000,
        )
        assert outcome["outcome"] == "consumed"

        loaded = store.get("appr-e2e")
        assert loaded.status == "consumed"

    def test_correct_phrase_normalized_case_whitespace(self, store):
        store.submit(_build_proposal())
        store.arm_text_confirm(
            "appr-e2e",
            event_id="$evt-arm",
            origin_server_ts_ms=int(time.time() * 1000),
        )

        outcome = process_text_confirm(
            "s-e2e", "appr-e2e",
            phrase="   RESTART-TIER2    vaultwarden   4821  ",
            confirmer="@requester:example",
            server_event_id="$evt",
            server_ts_ms=int(time.time() * 1000),
        )
        assert outcome["outcome"] == "consumed"


# ---------------------------------------------------------------------------
# Invalid attempt counting + lockout cascade
# ---------------------------------------------------------------------------


class TestInvalidAttemptsAndLockout:
    def test_wrong_phrase_does_not_consume(self, store):
        store.submit(_build_proposal())
        server_ts_ms = int(time.time() * 1000)
        store.arm_text_confirm(
            "appr-e2e",
            event_id="$evt-arm",
            origin_server_ts_ms=server_ts_ms,
        )

        outcome = process_text_confirm(
            "s-e2e", "appr-e2e",
            phrase="restart-tier2 vaultwarden 9999",   # wrong nonce
            confirmer="@requester:example",
            server_event_id="$evt",
            server_ts_ms=server_ts_ms + 1_000,
        )
        assert outcome["outcome"] == "invalid_attempt"
        assert outcome["invalid_attempts"] == 1
        assert outcome["phrase_reason"] == "nonce_mismatch"

        loaded = store.get("appr-e2e")
        assert loaded.status == "pending_confirm"
        assert loaded.invalid_attempts == 1

    def test_three_invalid_attempts_blocks_and_locks_out(self, store):
        store.submit(_build_proposal())
        server_ts_ms = int(time.time() * 1000)
        store.arm_text_confirm(
            "appr-e2e",
            event_id="$evt-arm",
            origin_server_ts_ms=server_ts_ms,
        )

        for _ in range(TIER2_INVALID_ATTEMPT_LIMIT - 1):
            outcome = process_text_confirm(
                "s-e2e", "appr-e2e",
                phrase="restart-tier2 vaultwarden 0000",
                confirmer="@requester:example",
                server_event_id="$evt",
                server_ts_ms=server_ts_ms + 1_000,
            )
            assert outcome["outcome"] == "invalid_attempt"

        # Limit-th attempt → blocked
        outcome = process_text_confirm(
            "s-e2e", "appr-e2e",
            phrase="restart-tier2 vaultwarden 0000",
            confirmer="@requester:example",
            server_event_id="$evt",
            server_ts_ms=server_ts_ms + 1_000,
        )
        assert outcome["outcome"] == "blocked_invalid_confirm_limit"
        assert outcome["invalid_attempts"] == TIER2_INVALID_ATTEMPT_LIMIT

        loaded = store.get("appr-e2e")
        assert loaded.status == "blocked"
        assert loaded.terminal_reason == "invalid_confirm_limit"

        # Lockout set on (verb, target)
        hit = store.check_lockout("restart-tier2", "vaultwarden")
        assert hit is not None
        expires_at, reason = hit
        assert reason == "invalid_confirm_limit"
        assert expires_at - time.time() <= TIER2_LOCKOUT_SECONDS
        assert expires_at - time.time() > TIER2_LOCKOUT_SECONDS - 5

    def test_subsequent_confirms_on_blocked_row_return_not_found(self, store):
        """After invalid_confirm_limit, the row is terminal. Late
        confirms with the correct phrase do not salvage it."""
        store.submit(_build_proposal())
        store.arm_text_confirm(
            "appr-e2e",
            event_id="$evt-arm",
            origin_server_ts_ms=int(time.time() * 1000),
        )
        for _ in range(TIER2_INVALID_ATTEMPT_LIMIT):
            process_text_confirm(
                "s-e2e", "appr-e2e",
                phrase="restart-tier2 vaultwarden 0000",
                confirmer="@requester:example",
                server_event_id="$evt",
                server_ts_ms=int(time.time() * 1000),
            )

        late = process_text_confirm(
            "s-e2e", "appr-e2e",
            phrase="restart-tier2 vaultwarden 4821",  # correct now
            confirmer="@requester:example",
            server_event_id="$evt",
            server_ts_ms=int(time.time() * 1000),
        )
        assert late["outcome"] == "not_found"


# ---------------------------------------------------------------------------
# Lockout scoping: only (verb, target) is blocked
# ---------------------------------------------------------------------------


class TestLockoutScope:
    @pytest.fixture(autouse=True)
    def _setup_session(self, store):
        self.store = store
        self.session_key = "lockout-scope"
        register_gateway_notify(self.session_key, lambda data: None)
        yield
        unregister_gateway_notify(self.session_key)

    def _approval_data(self, command: str):
        return {
            "command": command,
            "description": "tier2 pilot",
            "pattern_key": "hermes-ctl",
            "pattern_keys": ["hermes-ctl"],
            "requester": "@requester:example",
            "cwd": "/",
            "backend": "bash",
        }

    def test_lockout_blocks_same_verb_and_target(self):
        self.store.set_lockout(
            "restart-tier2", "vaultwarden",
            expires_at=time.time() + 300,
            reason="invalid_confirm_limit",
        )
        result = _await_gateway_decision(
            session_key=self.session_key,
            notify_cb=lambda d: {"event_id": "$x", "origin_server_ts_ms": 1},
            approval_data=self._approval_data(
                "sudo hermes-ctl restart-tier2 vaultwarden",
            ),
            surface="test",
        )
        assert result.get("tier2_lockout") is True
        assert self.store._proposals == {}

    def test_lockout_does_not_block_other_target(self):
        """Lockout on (restart-tier2, vaultwarden) MUST NOT block
        (restart-tier2, cloudflared). Currently cloudflared is not in
        the allowlist so it doesn't take the tier-2 path at all; we
        verify the lockout API independently."""
        self.store.set_lockout(
            "restart-tier2", "vaultwarden",
            expires_at=time.time() + 300,
            reason="invalid_confirm_limit",
        )
        # API-level: check_lockout on different target returns None
        assert self.store.check_lockout(
            "restart-tier2", "cloudflared",
        ) is None

    def test_lockout_does_not_block_other_verb(self):
        """Lockout on (restart-tier2, vaultwarden) MUST NOT block
        (diag, vaultwarden)."""
        self.store.set_lockout(
            "restart-tier2", "vaultwarden",
            expires_at=time.time() + 300,
            reason="invalid_confirm_limit",
        )
        assert self.store.check_lockout("diag", "vaultwarden") is None


# ---------------------------------------------------------------------------
# Wrong sender / requester check
# ---------------------------------------------------------------------------


class TestRequesterCheck:
    def test_wrong_sender_ignored_session_match(self, store):
        store.submit(_build_proposal(requester="@alice:example"))
        store.arm_text_confirm(
            "appr-e2e",
            event_id="$evt",
            origin_server_ts_ms=int(time.time() * 1000),
        )

        outcome = process_text_confirm(
            "s-e2e", "appr-e2e",
            phrase="restart-tier2 vaultwarden 4821",
            confirmer="@bob:example",   # not the requester
            server_event_id="$evt",
            server_ts_ms=int(time.time() * 1000),
        )
        assert outcome["outcome"] == "ignored_sender_mismatch"
        assert outcome["expected"] == "@alice:example"
        assert outcome["got"] == "@bob:example"

        # State unchanged — no invalid_attempt, no consume
        loaded = store.get("appr-e2e")
        assert loaded.status == "pending_confirm"
        assert loaded.invalid_attempts == 0

    def test_wrong_session_ignored(self, store):
        store.submit(_build_proposal(session_key="s-A"))
        store.arm_text_confirm(
            "appr-e2e",
            event_id="$evt",
            origin_server_ts_ms=int(time.time() * 1000),
        )

        outcome = process_text_confirm(
            "s-B", "appr-e2e",   # wrong session
            phrase="restart-tier2 vaultwarden 4821",
            confirmer="@requester:example",
            server_event_id="$evt",
            server_ts_ms=int(time.time() * 1000),
        )
        assert outcome["outcome"] == "ignored_sender_mismatch"

        loaded = store.get("appr-e2e")
        assert loaded.status == "pending_confirm"


# ---------------------------------------------------------------------------
# Clock skew
# ---------------------------------------------------------------------------


class TestClockSkew:
    def test_skew_above_threshold_blocks(self, store):
        store.submit(_build_proposal())
        server_ts_ms = int(time.time() * 1000)
        store.arm_text_confirm(
            "appr-e2e",
            event_id="$evt-arm",
            origin_server_ts_ms=server_ts_ms,
        )

        # Confirm message comes with a server_ts that's WAY off from
        # local clock — simulating a server-time drift or replay.
        bad_server_ts = (
            int(time.time() * 1000) + TIER2_CLOCK_SKEW_THRESHOLD_MS + 5_000
        )
        outcome = process_text_confirm(
            "s-e2e", "appr-e2e",
            phrase="restart-tier2 vaultwarden 4821",
            confirmer="@requester:example",
            server_event_id="$evt",
            server_ts_ms=bad_server_ts,
        )
        assert outcome["outcome"] == "blocked_clock_skew"
        assert outcome["skew_ms"] > TIER2_CLOCK_SKEW_THRESHOLD_MS

        loaded = store.get("appr-e2e")
        assert loaded.status == "blocked"
        assert loaded.terminal_reason == "clock_skew"

    def test_skew_at_threshold_is_ok(self, store):
        store.submit(_build_proposal())
        now_ms = int(time.time() * 1000)
        store.arm_text_confirm(
            "appr-e2e",
            event_id="$evt-arm",
            origin_server_ts_ms=now_ms,
        )

        # exactly at threshold — accept
        confirm_ts = now_ms + TIER2_CLOCK_SKEW_THRESHOLD_MS
        outcome = process_text_confirm(
            "s-e2e", "appr-e2e",
            phrase="restart-tier2 vaultwarden 4821",
            confirmer="@requester:example",
            server_event_id="$evt",
            server_ts_ms=confirm_ts,
            now_ms=now_ms,
        )
        assert outcome["outcome"] == "consumed"


# ---------------------------------------------------------------------------
# Explicit deny
# ---------------------------------------------------------------------------


class TestExplicitDeny:
    def test_reaction_deny_consumes_immediately(self, store):
        store.submit(_build_proposal())
        store.arm_text_confirm(
            "appr-e2e",
            event_id="$evt-arm",
            origin_server_ts_ms=int(time.time() * 1000),
        )

        outcome = process_explicit_deny(
            "s-e2e", "appr-e2e",
            denier="@requester:example",
            source="reaction",
        )
        assert outcome["outcome"] == "denied_explicit"
        assert outcome["source"] == "reaction"

        loaded = store.get("appr-e2e")
        assert loaded.status == "denied"
        assert loaded.terminal_reason == "denied_explicit"

    def test_text_deny_phrase_consumes_immediately(self, store):
        store.submit(_build_proposal())
        store.arm_text_confirm(
            "appr-e2e",
            event_id="$evt-arm",
            origin_server_ts_ms=int(time.time() * 1000),
        )

        outcome = process_explicit_deny(
            "s-e2e", "appr-e2e",
            denier="@requester:example",
            source="text",
        )
        assert outcome["outcome"] == "denied_explicit"

        loaded = store.get("appr-e2e")
        assert loaded.status == "denied"

    def test_explicit_deny_from_wrong_sender_ignored(self, store):
        store.submit(_build_proposal(requester="@alice:example"))
        store.arm_text_confirm(
            "appr-e2e",
            event_id="$evt-arm",
            origin_server_ts_ms=int(time.time() * 1000),
        )

        outcome = process_explicit_deny(
            "s-e2e", "appr-e2e",
            denier="@bob:example",
            source="reaction",
        )
        assert outcome["outcome"] == "ignored_sender_mismatch"

        loaded = store.get("appr-e2e")
        assert loaded.status == "pending_confirm"

    def test_explicit_deny_before_arm_also_works(self, store):
        """A deny can come in on a still-pending tier-2 row (e.g.
        before the user types anything), and the row transitions
        cleanly to denied."""
        store.submit(_build_proposal())
        # NB: not armed yet

        outcome = process_explicit_deny(
            "s-e2e", "appr-e2e",
            denier="@requester:example",
            source="reaction",
        )
        assert outcome["outcome"] == "denied_explicit"

        loaded = store.get("appr-e2e")
        assert loaded.status == "denied"


# ---------------------------------------------------------------------------
# TTL: anchored to origin_server_ts, not gateway clock
# ---------------------------------------------------------------------------


class TestTtl:
    def test_expire_due_after_ttl_stamps_expired_no_confirm(self, store):
        created = time.time()
        proposal = _build_proposal(created=created)
        store.submit(proposal)
        server_ts_ms = int(created * 1000)
        store.arm_text_confirm(
            "appr-e2e",
            event_id="$evt-arm",
            origin_server_ts_ms=server_ts_ms,
            now=created,
        )

        # Advance time past TTL
        future_now = created + TIER2_TTL_SECONDS + 10
        marked = store.expire_due(now=future_now)
        assert marked >= 1

        loaded = store.get("appr-e2e")
        assert loaded.status == "expired"
        assert loaded.terminal_reason == "expired_no_confirm"

    def test_confirm_after_ttl_returns_expired_no_confirm(self, store):
        created = time.time()
        store.submit(_build_proposal(created=created))
        server_ts_ms = int(created * 1000)
        store.arm_text_confirm(
            "appr-e2e",
            event_id="$evt-arm",
            origin_server_ts_ms=server_ts_ms,
            now=created,
        )

        # Run the orchestrator with now_ms past the TTL — TTL is
        # anchored to approval_event_ts_ms (server_ts_ms above).
        late_now_ms = server_ts_ms + (TIER2_TTL_SECONDS + 10) * 1000
        outcome = process_text_confirm(
            "s-e2e", "appr-e2e",
            phrase="restart-tier2 vaultwarden 4821",
            confirmer="@requester:example",
            server_event_id="$evt",
            server_ts_ms=late_now_ms,
            now_ms=late_now_ms,
        )
        assert outcome["outcome"] == "expired_no_confirm"


# ---------------------------------------------------------------------------
# Nonce binding (defensive: orchestrator trusts row state, not message body)
# ---------------------------------------------------------------------------


class TestNonceBinding:
    def test_nonce_pinned_at_submit_not_inferred_from_phrase(self, store):
        """A confirmer cannot 'guess' the right nonce by typing
        anything — the pinned nonce is what the orchestrator compares
        against, regardless of what the message body says first."""
        proposal = _build_proposal(nonce="0042")
        store.submit(proposal)
        store.arm_text_confirm(
            "appr-e2e",
            event_id="$evt-arm",
            origin_server_ts_ms=int(time.time() * 1000),
        )

        # Try every wrong nonce — all become invalid_attempts until limit
        for guess in ("0001", "0002"):
            outcome = process_text_confirm(
                "s-e2e", "appr-e2e",
                phrase=f"restart-tier2 vaultwarden {guess}",
                confirmer="@requester:example",
                server_event_id="$evt",
                server_ts_ms=int(time.time() * 1000),
            )
            assert outcome["outcome"] == "invalid_attempt"

        # The correct pinned nonce works
        ok = process_text_confirm(
            "s-e2e", "appr-e2e",
            phrase="restart-tier2 vaultwarden 0042",
            confirmer="@requester:example",
            server_event_id="$evt",
            server_ts_ms=int(time.time() * 1000),
        )
        assert ok["outcome"] == "consumed"

    def test_pinned_nonce_survives_round_trip_through_store(self, store):
        """Even after consuming, the loaded proposal carries the
        pinned (verb, target, nonce) — payload_json is immutable
        post-submit and the audit row can identify what was approved."""
        store.submit(_build_proposal(nonce="7777"))
        store.arm_text_confirm(
            "appr-e2e",
            event_id="$evt-arm",
            origin_server_ts_ms=int(time.time() * 1000),
        )
        process_text_confirm(
            "s-e2e", "appr-e2e",
            phrase="restart-tier2 vaultwarden 7777",
            confirmer="@requester:example",
            server_event_id="$evt",
            server_ts_ms=int(time.time() * 1000),
        )

        loaded = store.get("appr-e2e")
        assert loaded.status == "consumed"
        assert loaded.verb == "restart-tier2"
        assert loaded.target == "vaultwarden"
        assert loaded.nonce == "7777"


# ---------------------------------------------------------------------------
# Classifier coverage (sanity)
# ---------------------------------------------------------------------------


class TestClassifierCoverage:
    def test_vaultwarden_pilot_target_matches(self):
        assert classify_tier2_critical(
            "sudo hermes-ctl restart-tier2 vaultwarden",
        ) == ("restart-tier2", "vaultwarden")

    def test_other_targets_not_yet_tier2(self):
        """Cloudflared, proxy, authelia, postgresql — none of these
        are in the pilot allowlist. New (verb, target) pairs require
        explicit code change (Hermes spec: narrow gate)."""
        for cmd in (
            "sudo hermes-ctl restart-tier2 cloudflared",
            "sudo hermes-ctl restart-tier2 traefik",
            "sudo hermes-ctl restart-tier2 authelia",
            "sudo hermes-ctl restart-tier2 postgresql",
        ):
            assert classify_tier2_critical(cmd) is None, (
                f"{cmd!r} should not be tier-2-classified until "
                "explicitly added to _TIER2_CRITICAL_TARGETS"
            )
