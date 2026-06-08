"""Unit tests for tier-2-critical helpers in :mod:`tools.approval`.

Targets commit 3 scope only: pure helpers + the /approve-rejection
branch of ``resolve_gateway_approval_by_id``. End-to-end scenarios
(text-confirm flow, lockout, clock-skew on real proposals, ❌-reaction
routing) live in commit 6's ``test_tier2_text_confirm.py`` which
exercises the orchestrator + store layers together.
"""

from __future__ import annotations

import time

import pytest

from tools.approval import (
    TIER2_CLOCK_SKEW_THRESHOLD_MS,
    TIER2_INVALID_ATTEMPT_LIMIT,
    TIER2_LOCKOUT_SECONDS,
    TIER2_NONCE_LENGTH,
    TIER2_TTL_SECONDS,
    _await_gateway_decision,
    _check_tier2_clock_skew,
    _compute_tier2_ttl_remaining_ms,
    _normalize_confirm_phrase,
    _parse_confirm_phrase,
    _verify_confirm_phrase,
    classify_tier2_critical,
    generate_tier2_nonce,
    register_gateway_notify,
    resolve_gateway_approval_by_id,
    set_default_approval_store,
    unregister_gateway_notify,
)
from tools.approval_store import ApprovalProposal
from tools.approval_store_memory import InMemoryApprovalStore


# ---------------------------------------------------------------------------
# Phrase normalisation + parsing
# ---------------------------------------------------------------------------


class TestNormalizeConfirmPhrase:
    def test_uppercases(self):
        assert _normalize_confirm_phrase("restart vaultwarden 4821") == \
            "RESTART VAULTWARDEN 4821"

    def test_collapses_whitespace(self):
        assert _normalize_confirm_phrase("  RESTART   VAULTWARDEN   4821  ") == \
            "RESTART VAULTWARDEN 4821"

    def test_empty_returns_empty(self):
        assert _normalize_confirm_phrase("") == ""
        assert _normalize_confirm_phrase("   ") == ""


class TestParseConfirmPhrase:
    def test_well_formed(self):
        assert _parse_confirm_phrase("RESTART VAULTWARDEN 4821") == \
            ("RESTART", "VAULTWARDEN", "4821")

    def test_too_many_parts(self):
        assert _parse_confirm_phrase("RESTART VAULTWARDEN EXTRA 4821") is None

    def test_too_few_parts(self):
        assert _parse_confirm_phrase("RESTART VAULTWARDEN") is None

    def test_short_nonce(self):
        assert _parse_confirm_phrase("RESTART VAULTWARDEN 482") is None

    def test_long_nonce(self):
        assert _parse_confirm_phrase("RESTART VAULTWARDEN 48211") is None

    def test_non_digit_nonce(self):
        assert _parse_confirm_phrase("RESTART VAULTWARDEN ABCD") is None

    def test_lowercase_normalised(self):
        assert _parse_confirm_phrase("restart vaultwarden 4821") == \
            ("RESTART", "VAULTWARDEN", "4821")


class TestVerifyConfirmPhrase:
    BASE = dict(
        expected_verb="restart-tier2",
        expected_target="vaultwarden",
        expected_nonce="4821",
    )

    def test_match_canonical(self):
        ok, reason = _verify_confirm_phrase(
            "RESTART-TIER2 VAULTWARDEN 4821", **self.BASE,
        )
        assert (ok, reason) == (True, None)

    def test_match_lowercase(self):
        ok, reason = _verify_confirm_phrase(
            "restart-tier2 vaultwarden 4821", **self.BASE,
        )
        assert (ok, reason) == (True, None)

    def test_match_extra_whitespace(self):
        ok, reason = _verify_confirm_phrase(
            "   restart-tier2   vaultwarden  4821  ", **self.BASE,
        )
        assert (ok, reason) == (True, None)

    def test_verb_mismatch(self):
        ok, reason = _verify_confirm_phrase(
            "DIAG VAULTWARDEN 4821", **self.BASE,
        )
        assert (ok, reason) == (False, "verb_mismatch")

    def test_target_mismatch(self):
        ok, reason = _verify_confirm_phrase(
            "RESTART-TIER2 CLOUDFLARED 4821", **self.BASE,
        )
        assert (ok, reason) == (False, "target_mismatch")

    def test_nonce_mismatch(self):
        ok, reason = _verify_confirm_phrase(
            "RESTART-TIER2 VAULTWARDEN 9999", **self.BASE,
        )
        assert (ok, reason) == (False, "nonce_mismatch")

    def test_unparseable(self):
        ok, reason = _verify_confirm_phrase(
            "just-typing-anything", **self.BASE,
        )
        assert (ok, reason) == (False, "unparseable")

    def test_empty(self):
        ok, reason = _verify_confirm_phrase("", **self.BASE)
        assert (ok, reason) == (False, "unparseable")


# ---------------------------------------------------------------------------
# Clock skew
# ---------------------------------------------------------------------------


class TestCheckTier2ClockSkew:
    def test_zero_skew(self):
        ts = 1_700_000_000_000
        ok, skew = _check_tier2_clock_skew(ts, now_ms=ts)
        assert (ok, skew) == (True, 0)

    def test_under_threshold(self):
        ts = 1_700_000_000_000
        ok, skew = _check_tier2_clock_skew(ts, now_ms=ts + 10_000)
        assert ok is True
        assert skew == 10_000

    def test_at_threshold_is_ok(self):
        """Boundary: exactly at threshold should be accepted; only
        strictly exceeding fails closed."""
        ts = 1_700_000_000_000
        ok, skew = _check_tier2_clock_skew(
            ts, now_ms=ts + TIER2_CLOCK_SKEW_THRESHOLD_MS,
        )
        assert ok is True
        assert skew == TIER2_CLOCK_SKEW_THRESHOLD_MS

    def test_just_over_threshold_fails(self):
        ts = 1_700_000_000_000
        ok, _ = _check_tier2_clock_skew(
            ts, now_ms=ts + TIER2_CLOCK_SKEW_THRESHOLD_MS + 1,
        )
        assert ok is False

    def test_negative_skew(self):
        """Future server_ts (impossible in practice) still measured by
        absolute value."""
        ts = 1_700_000_000_000
        ok, skew = _check_tier2_clock_skew(ts, now_ms=ts - 50_000)
        assert ok is False
        assert skew == 50_000

    def test_none_server_ts_fails(self):
        """Missing server_ts means we have no anchor → fail closed."""
        ok, _ = _check_tier2_clock_skew(None)
        assert ok is False


# ---------------------------------------------------------------------------
# TTL remaining
# ---------------------------------------------------------------------------


class TestComputeTier2TtlRemainingMs:
    def test_fresh_proposal_full_ttl(self):
        ts = 1_700_000_000_000
        rem = _compute_tier2_ttl_remaining_ms(ts, now_ms=ts)
        assert rem == TIER2_TTL_SECONDS * 1000

    def test_partial_elapsed(self):
        ts = 1_700_000_000_000
        rem = _compute_tier2_ttl_remaining_ms(ts, now_ms=ts + 60_000)
        assert rem == (TIER2_TTL_SECONDS - 60) * 1000

    def test_past_ttl_negative(self):
        ts = 1_700_000_000_000
        rem = _compute_tier2_ttl_remaining_ms(
            ts, now_ms=ts + (TIER2_TTL_SECONDS + 10) * 1000,
        )
        assert rem < 0

    def test_none_anchor_returns_none(self):
        assert _compute_tier2_ttl_remaining_ms(None) is None


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class TestClassifyTier2Critical:
    def test_canonical_sudo_form(self):
        assert classify_tier2_critical(
            "sudo hermes-ctl restart-tier2 vaultwarden",
        ) == ("restart-tier2", "vaultwarden")

    def test_without_sudo(self):
        assert classify_tier2_critical(
            "hermes-ctl restart-tier2 vaultwarden",
        ) == ("restart-tier2", "vaultwarden")

    def test_with_trailing_args(self):
        assert classify_tier2_critical(
            "sudo hermes-ctl restart-tier2 vaultwarden --force",
        ) == ("restart-tier2", "vaultwarden")

    def test_non_tier2_verb(self):
        assert classify_tier2_critical(
            "sudo hermes-ctl diag vaultwarden",
        ) is None

    def test_non_tier2_target(self):
        assert classify_tier2_critical(
            "sudo hermes-ctl restart-tier2 cloudflared",
        ) is None

    def test_not_hermes_ctl(self):
        assert classify_tier2_critical(
            "sudo systemctl restart vaultwarden",
        ) is None

    def test_empty(self):
        assert classify_tier2_critical("") is None

    def test_too_few_tokens(self):
        assert classify_tier2_critical("hermes-ctl") is None


# ---------------------------------------------------------------------------
# Nonce generation
# ---------------------------------------------------------------------------


class TestGenerateTier2Nonce:
    def test_length(self):
        n = generate_tier2_nonce()
        assert len(n) == TIER2_NONCE_LENGTH

    def test_all_digits(self):
        for _ in range(50):
            assert generate_tier2_nonce().isdigit()

    def test_zero_padded(self):
        """Generation of a sub-1000 number must zero-pad to full length.
        Sample enough to hit the leading-zero case at least once
        statistically."""
        seen_padded = False
        for _ in range(2000):
            n = generate_tier2_nonce()
            assert len(n) == TIER2_NONCE_LENGTH
            if n[0] == "0":
                seen_padded = True
        assert seen_padded, (
            "leading-zero case not exercised — sample too small or "
            "generator is biased away from low values"
        )


# ---------------------------------------------------------------------------
# resolve_gateway_approval_by_id: tier 2 /approve rejection
# ---------------------------------------------------------------------------


def _make_tier2_proposal(approval_id="appr-rej", session_key="s-tier2",
                        requester="@u:example", verb="restart-tier2",
                        target="vaultwarden", nonce="4821"):
    created = time.time()
    return ApprovalProposal(
        approval_id=approval_id,
        created_at=created,
        expires_at=created + TIER2_TTL_SECONDS,
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


class TestResolveByIdTier2Rejection:
    @pytest.fixture(autouse=True)
    def _restore_store(self):
        # Save and restore the module-level default store across tests
        # so this file does not leak state into other test modules.
        from tools import approval as _appr
        prev = _appr.get_default_approval_store()
        yield
        set_default_approval_store(prev)

    def test_approve_on_tier2_returns_minus_two(self):
        """/approve <id> on a tier2_critical proposal must be rejected
        (return -2) so the handler can surface the phrase hint without
        consuming the row."""
        store = InMemoryApprovalStore()
        set_default_approval_store(store)
        store.submit(_make_tier2_proposal("appr-rej-1"))

        rc = resolve_gateway_approval_by_id("s-tier2", "appr-rej-1", "approve")
        assert rc == -2

        # Row is still pending (not consumed by the rejected /approve)
        loaded = store.get("appr-rej-1")
        assert loaded.status == "pending"

    def test_approve_on_tier2_in_pending_confirm_also_rejected(self):
        """Even after arming, /approve <id> must be rejected for
        tier2_critical — only the phrase consumes."""
        store = InMemoryApprovalStore()
        set_default_approval_store(store)
        store.submit(_make_tier2_proposal("appr-rej-2"))
        store.arm_text_confirm(
            "appr-rej-2",
            event_id="$evt",
            origin_server_ts_ms=int(time.time() * 1000),
        )

        rc = resolve_gateway_approval_by_id("s-tier2", "appr-rej-2", "approve")
        assert rc == -2
        assert store.get("appr-rej-2").status == "pending_confirm"

    def test_deny_on_tier2_is_allowed_and_stamps_reason(self):
        """/deny <id> for tier2_critical IS allowed and records
        terminal_reason='denied_explicit'."""
        store = InMemoryApprovalStore()
        set_default_approval_store(store)
        store.submit(_make_tier2_proposal("appr-rej-3"))

        rc = resolve_gateway_approval_by_id("s-tier2", "appr-rej-3", "deny")
        # No waiter → orphan → -1, but the store transition committed.
        assert rc in (-1, 1)

        loaded = store.get("appr-rej-3")
        assert loaded.status == "denied"
        assert loaded.terminal_reason == "denied_explicit"

    def test_deny_on_tier2_works_when_armed(self):
        """/deny on a pending_confirm row also transitions to denied."""
        store = InMemoryApprovalStore()
        set_default_approval_store(store)
        store.submit(_make_tier2_proposal("appr-rej-4"))
        store.arm_text_confirm(
            "appr-rej-4",
            event_id="$evt",
            origin_server_ts_ms=int(time.time() * 1000),
        )

        rc = resolve_gateway_approval_by_id("s-tier2", "appr-rej-4", "deny")
        assert rc in (-1, 1)

        loaded = store.get("appr-rej-4")
        assert loaded.status == "denied"
        assert loaded.terminal_reason == "denied_explicit"


# ---------------------------------------------------------------------------
# _await_gateway_decision: tier 2 lockout + arm path (commit 4 scope)
# ---------------------------------------------------------------------------


class TestAwaitGatewayDecisionTier2:
    """End-to-end-ish tests of the tier-2 branches in _await_gateway_decision.

    The store is real (InMemoryApprovalStore); the notify_cb is a lambda
    that either returns no binding (failure case) or returns a valid
    ArmBinding-shaped dict (success case). We do NOT exercise the
    blocking wait — these tests assert the synchronous return value
    that signals fail-closed paths before the wait loop ever runs.
    """

    TIER2_COMMAND = "sudo hermes-ctl restart-tier2 vaultwarden"

    @pytest.fixture(autouse=True)
    def _isolate_store_and_callbacks(self):
        from tools import approval as _appr
        prev = _appr.get_default_approval_store()
        store = InMemoryApprovalStore()
        set_default_approval_store(store)
        self.store = store
        self.session_key = f"tier2-await-{id(self)}"
        register_gateway_notify(
            self.session_key,
            lambda data: None,  # placeholder; overridden per test
        )
        yield
        unregister_gateway_notify(self.session_key)
        set_default_approval_store(prev)

    def _approval_data(self):
        return {
            "command": self.TIER2_COMMAND,
            "description": "tier2-critical pilot",
            "pattern_key": "hermes-ctl-restart-tier2",
            "pattern_keys": ["hermes-ctl-restart-tier2"],
            "requester": "@u:example",
            "cwd": "/",
            "backend": "bash",
        }

    def test_lockout_pre_check_fails_closed(self):
        """An active lockout on (verb, target) blocks new approvals
        before submit even runs. Result signals tier2_lockout=True and
        no proposal row is created."""
        self.store.set_lockout(
            "restart-tier2", "vaultwarden",
            expires_at=time.time() + TIER2_LOCKOUT_SECONDS,
            reason="invalid_confirm_limit",
        )
        result = _await_gateway_decision(
            session_key=self.session_key,
            notify_cb=lambda data: {
                "event_id": "$x", "origin_server_ts_ms": 1,
            },
            approval_data=self._approval_data(),
            surface="test",
        )
        assert result["resolved"] is False
        assert result.get("tier2_lockout") is True
        assert "lockout_expires_at" in result
        # No proposal should have been created — list all stored
        # approval_ids; we don't expose a listing API on store, but
        # we know the only approval_id would be generated by the
        # _await_gateway_decision call, and since it returned before
        # submit, the internal _proposals dict stays empty for this
        # test (set_lockout doesn't add to _proposals).
        assert self.store._proposals == {}

    def test_notify_returns_none_fails_closed_and_denies(self):
        """If notify_cb returns None for a tier-2 proposal the row is
        already submitted but cannot be confirmed (no Matrix binding).
        Must FAIL CLOSED and stamp the row denied."""
        result = _await_gateway_decision(
            session_key=self.session_key,
            notify_cb=lambda data: None,
            approval_data=self._approval_data(),
            surface="test",
        )
        assert result["resolved"] is False
        assert result.get("tier2_no_binding") is True

        # Find the submitted proposal — there should be exactly one
        # in the store, in status='denied'.
        assert len(self.store._proposals) == 1
        proposal = next(iter(self.store._proposals.values()))
        assert proposal.status == "denied"
        assert proposal.terminal_reason == "store_failed"

    def test_notify_returns_malformed_binding_fails_closed(self):
        """A dict that's missing required keys or has wrong types is
        treated as 'no binding'."""
        result = _await_gateway_decision(
            session_key=self.session_key,
            notify_cb=lambda data: {"event_id": "$x"},  # no ts
            approval_data=self._approval_data(),
            surface="test",
        )
        assert result.get("tier2_no_binding") is True

        proposal = next(iter(self.store._proposals.values()))
        assert proposal.status == "denied"

    def test_notify_returns_empty_event_id_fails_closed(self):
        """An empty event_id is invalid — Matrix can't be queried
        with it later for original-event verification."""
        result = _await_gateway_decision(
            session_key=self.session_key,
            notify_cb=lambda data: {
                "event_id": "", "origin_server_ts_ms": 1,
            },
            approval_data=self._approval_data(),
            surface="test",
        )
        assert result.get("tier2_no_binding") is True

    def test_notify_returns_non_int_ts_fails_closed(self):
        """origin_server_ts_ms must be int (epoch milliseconds)."""
        result = _await_gateway_decision(
            session_key=self.session_key,
            notify_cb=lambda data: {
                "event_id": "$x", "origin_server_ts_ms": "1700000000000",
            },
            approval_data=self._approval_data(),
            surface="test",
        )
        assert result.get("tier2_no_binding") is True

    def test_notify_raises_denies_pending_proposal(self):
        """If notify_cb raises during a tier-2 flow, the submitted
        proposal must be cleaned up — otherwise we leave a row with
        pinned phrase state that nobody can ever confirm."""
        def raising_cb(data):
            raise RuntimeError("simulated send failure")
        result = _await_gateway_decision(
            session_key=self.session_key,
            notify_cb=raising_cb,
            approval_data=self._approval_data(),
            surface="test",
        )
        assert result["resolved"] is False
        assert result["notify_failed"] is True

        # Submitted row stamped denied (store_failed) — no dangling
        # pending tier-2 row.
        if self.store._proposals:
            proposal = next(iter(self.store._proposals.values()))
            assert proposal.status == "denied"

    def test_valid_binding_arms_proposal(self):
        """Happy path through the synchronous portion: valid binding
        leads to arm_text_confirm being called. The blocking wait
        happens after, which we don't exercise here (no /confirm
        comes in, so we'd hit the timeout)."""
        # Spawn _await_gateway_decision in a thread; check the
        # proposal lands in pending_confirm with the binding attached.
        import threading

        arm_observed = threading.Event()

        def notify_cb(data):
            return {
                "event_id": "$arm-evt-abc",
                "origin_server_ts_ms": int(time.time() * 1000),
            }

        result_holder = {}

        def runner():
            result_holder["result"] = _await_gateway_decision(
                session_key=self.session_key,
                notify_cb=notify_cb,
                approval_data=self._approval_data(),
                surface="test",
            )
            arm_observed.set()

        t = threading.Thread(target=runner)
        t.start()
        # Poll briefly for the proposal to be created + armed. Give
        # up to 5s; on healthy systems this completes in well under
        # a millisecond. The runner will block in the wait loop after.
        deadline = time.time() + 5
        proposal = None
        while time.time() < deadline:
            if self.store._proposals:
                aid = next(iter(self.store._proposals))
                proposal = self.store._proposals[aid]
                if proposal.status == "pending_confirm":
                    break
            time.sleep(0.01)
        assert proposal is not None
        assert proposal.status == "pending_confirm", (
            f"expected pending_confirm; got {proposal.status}"
        )
        assert proposal.approval_event_id == "$arm-evt-abc"
        assert proposal.approval_event_ts_ms is not None
        # Resolve the proposal to let the wait loop exit cleanly so the
        # thread doesn't dangle past the test.
        resolve_gateway_approval_by_id(
            self.session_key, proposal.approval_id, "deny",
        )
        t.join(timeout=10)
        assert not t.is_alive(), "wait loop did not exit after deny"

    def test_non_tier2_command_ignores_notify_return(self):
        """Backward compat: non-tier-2 commands keep the legacy contract
        — notify_cb return value is ignored, no arm_text_confirm is
        called, no tier2_no_binding error path."""
        import threading

        def notify_cb(data):
            # Returning something weird must not crash anything for
            # non-tier2 commands.
            return "legacy callback returns garbage"

        result_holder = {}

        def runner():
            result_holder["result"] = _await_gateway_decision(
                session_key=self.session_key,
                notify_cb=notify_cb,
                approval_data={
                    "command": "rm -rf /tmp/x",
                    "description": "recursive delete",
                    "pattern_key": "rm-recursive",
                    "pattern_keys": ["rm-recursive"],
                    "cwd": "/",
                    "backend": "bash",
                },
                surface="test",
            )

        t = threading.Thread(target=runner)
        t.start()

        # Resolve via /deny once the proposal is queued.
        deadline = time.time() + 5
        while time.time() < deadline:
            if self.store._proposals:
                proposal = next(iter(self.store._proposals.values()))
                if proposal.risk_level != "tier2_critical":
                    resolve_gateway_approval_by_id(
                        self.session_key, proposal.approval_id, "deny",
                    )
                    break
            time.sleep(0.01)
        t.join(timeout=10)
        assert not t.is_alive()
        assert result_holder["result"]["resolved"] is True
        assert result_holder["result"]["choice"] == "deny"
