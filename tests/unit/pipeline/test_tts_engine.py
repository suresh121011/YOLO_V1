"""
Unit tests for src.pipeline.tts_engine queue admission.

The live speech path bounded its PriorityQueue at 5 and, on overflow, dropped
the **incoming** message. With routine prompts backed up behind a multi-second
synthesis, an arriving CRITICAL was discarded while the chatter still played —
inverting the priority contract that `AlertQueue.put` was written to uphold and
never wired in. See docs/08_scenario_engineering/architecture_review.md §6.

`speak()` depends only on `self._queue` (`_sanitize` is a staticmethod), so
these construct the object without starting the worker thread or requiring a
Piper binary.
"""

from __future__ import annotations

import queue

import pytest

from src.pipeline.tts_engine import TTS_QUEUE_MAXSIZE, PiperTTS

NORMAL = 1
PRIORITY = 0


def _engine() -> PiperTTS:
    """A PiperTTS with only the queue wired — no thread, no subprocess."""
    engine = object.__new__(PiperTTS)
    engine._queue = queue.PriorityQueue(maxsize=TTS_QUEUE_MAXSIZE)
    return engine


def _fill_with_normal(engine: PiperTTS) -> None:
    for i in range(TTS_QUEUE_MAXSIZE):
        engine.speak(f"routine prompt {i}", priority=False)
    assert engine._queue.full()


def _drain(engine: PiperTTS) -> list[tuple[int, str]]:
    items = []
    while not engine._queue.empty():
        items.append(engine._queue.get_nowait())
    return items


class TestNormalAdmission:
    @pytest.mark.unit
    def test_enqueues_until_full(self) -> None:
        engine = _engine()
        _fill_with_normal(engine)
        assert engine._queue.qsize() == TTS_QUEUE_MAXSIZE

    @pytest.mark.unit
    def test_priority_is_spoken_first(self) -> None:
        engine = _engine()
        engine.speak("routine", priority=False)
        engine.speak("urgent", priority=True)
        assert _drain(engine)[0] == (PRIORITY, "urgent")

    @pytest.mark.unit
    def test_blank_text_is_ignored(self) -> None:
        engine = _engine()
        engine.speak("   ")
        engine.speak("\x00\x01")
        assert engine._queue.empty()


class TestOverflowEviction:
    @pytest.mark.unit
    def test_critical_evicts_a_routine_message(self) -> None:
        """The core inversion: an arriving CRITICAL used to be the one dropped."""
        engine = _engine()
        _fill_with_normal(engine)

        engine.speak("Please check the kitchen.", priority=True)

        items = _drain(engine)
        assert (PRIORITY, "Please check the kitchen.") in items
        assert len(items) == TTS_QUEUE_MAXSIZE
        assert sum(1 for prio, _ in items if prio == NORMAL) == TTS_QUEUE_MAXSIZE - 1

    @pytest.mark.unit
    def test_critical_is_at_the_front_after_eviction(self) -> None:
        engine = _engine()
        _fill_with_normal(engine)
        engine.speak("Please check the kitchen.", priority=True)
        assert _drain(engine)[0] == (PRIORITY, "Please check the kitchen.")

    @pytest.mark.unit
    def test_routine_message_is_dropped_when_nothing_outranks_it(self) -> None:
        """Eviction must not degenerate into churning equal-priority messages."""
        engine = _engine()
        _fill_with_normal(engine)

        engine.speak("another routine prompt", priority=False)

        texts = [text for _, text in _drain(engine)]
        assert "another routine prompt" not in texts
        assert len(texts) == TTS_QUEUE_MAXSIZE

    @pytest.mark.unit
    def test_critical_does_not_evict_another_critical(self) -> None:
        engine = _engine()
        for i in range(TTS_QUEUE_MAXSIZE):
            engine.speak(f"urgent {i}", priority=True)

        engine.speak("later urgent", priority=False)
        engine.speak("later critical", priority=True)

        texts = [text for _, text in _drain(engine)]
        assert "later critical" not in texts
        assert len(texts) == TTS_QUEUE_MAXSIZE

    @pytest.mark.unit
    def test_queue_never_exceeds_its_bound(self) -> None:
        engine = _engine()
        for i in range(50):
            engine.speak(f"msg {i}", priority=(i % 7 == 0))
        assert engine._queue.qsize() <= TTS_QUEUE_MAXSIZE

    @pytest.mark.unit
    def test_recovers_if_worker_drains_between_attempts(self) -> None:
        """Guards the race between the failed put and taking the lock."""
        engine = _engine()
        _fill_with_normal(engine)
        while not engine._queue.empty():
            engine._queue.get_nowait()

        engine.speak("after drain", priority=True)
        assert _drain(engine) == [(PRIORITY, "after drain")]
