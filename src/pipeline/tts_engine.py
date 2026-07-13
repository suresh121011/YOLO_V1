"""
Piper TTS Engine — Non-blocking Neural Text-to-Speech
=======================================================
Wraps the Piper TTS CLI (https://github.com/rhasspy/piper) in a background
thread with a priority queue, so the main inference loop is never blocked
by speech synthesis.

Design:
  - Background daemon thread consumes a PriorityQueue[Alert]
  - Main thread calls speak() → puts into queue (non-blocking)
  - On TTS failure: falls back to an audible beep
  - On Piper not found: logs warning, silently drops all speech

Usage:
    tts = PiperTTS(model_path="models/tts/en_IN-medium.onnx",
                   config_path="models/tts/en_IN-medium.onnx.json")
    tts.speak("Please be careful, there is a knife nearby.")
    # ... main loop continues immediately; speech plays in background
    tts.shutdown()  # on exit
"""

from __future__ import annotations

import logging
import queue
import subprocess
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

TTS_QUEUE_MAXSIZE = 5  # Drop oldest low-priority items when queue is full


class PiperTTS:
    """Non-blocking Piper neural TTS engine.

    All speech synthesis runs on a background daemon thread.
    The main inference loop never waits for TTS completion.

    Priority:
        priority=True  → int 0 (spoken before normal)
        priority=False → int 1 (normal order)
    """

    def __init__(
        self,
        model_path: str,
        config_path: str,
        speech_rate: float = 0.9,
    ) -> None:
        """
        Args:
            model_path:  Path to .onnx Piper voice model.
            config_path: Path to corresponding .json config file.
            speech_rate: Speaking rate multiplier. <1.0 = slower, >1.0 = faster.
        """
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        self.speech_rate = speech_rate

        self._queue: queue.PriorityQueue[tuple[int, str]] = queue.PriorityQueue(
            maxsize=TTS_QUEUE_MAXSIZE
        )
        self._shutdown = threading.Event()
        self._speaking = threading.Event()
        self._healthy = self._health_check()

        self._thread = threading.Thread(
            target=self._worker,
            daemon=True,
            name="tts-worker",
        )
        self._thread.start()
        logger.info(f"PiperTTS initialized (healthy={self._healthy})")

    # ─────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────

    def speak(self, text: str, priority: bool = False) -> None:
        """Enqueue text for speech synthesis (non-blocking).

        Args:
            text:     Text to speak. Sanitized before passing to Piper.
            priority: If True, placed ahead of normal-priority messages.
        """
        text = self._sanitize(text)
        if not text:
            return
        prio = 0 if priority else 1
        try:
            self._queue.put_nowait((prio, text))
        except queue.Full:
            logger.warning(f"TTS queue full, dropping: '{text[:40]}...'")

    def is_speaking(self) -> bool:
        """Return True if TTS is currently synthesizing or playing audio."""
        return self._speaking.is_set()

    def health_check(self) -> bool:
        """Return True if Piper binary is available and worker thread is alive."""
        return self._healthy and self._thread.is_alive()

    def shutdown(self) -> None:
        """Signal worker thread to stop and wait for it to finish."""
        self._shutdown.set()
        self._thread.join(timeout=5.0)
        logger.info("PiperTTS shutdown complete")

    # ─────────────────────────────────────────
    # Internal
    # ─────────────────────────────────────────

    def _health_check(self) -> bool:
        """Verify Piper binary is on PATH and executable."""
        try:
            result = subprocess.run(
                ["piper", "--help"],
                capture_output=True,
                timeout=3,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            logger.warning("Piper TTS binary not found — speech output disabled")
            return False

    def _worker(self) -> None:
        """Background daemon: consume queue and synthesize speech."""
        while not self._shutdown.is_set():
            try:
                _, text = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            self._speaking.set()
            try:
                if self._healthy:
                    self._synthesize_and_play(text)
                else:
                    self._beep()
            except Exception as e:
                logger.error(f"TTS synthesis error: {e}")
                self._beep()
            finally:
                self._speaking.clear()

    def _synthesize_and_play(self, text: str) -> None:
        """Run Piper CLI and pipe WAV output to audio playback."""
        length_scale = 1.0 / max(self.speech_rate, 0.1)
        cmd = [
            "piper",
            "--model",
            str(self.model_path),
            "--config",
            str(self.config_path),
            "--output_file",
            "-",  # Write WAV to stdout
            "--length_scale",
            str(length_scale),
        ]
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        audio_data, stderr = process.communicate(
            input=text.encode("utf-8"),
            timeout=10,
        )
        if process.returncode != 0:
            raise RuntimeError(f"Piper returned non-zero: {stderr.decode()[:200]}")
        self._play_wav(audio_data)

    def _play_wav(self, wav_bytes: bytes) -> None:
        """Play WAV bytes via sounddevice (primary) or system command (fallback)."""
        try:
            import io
            import wave

            import numpy as np
            import sounddevice as sd

            with wave.open(io.BytesIO(wav_bytes)) as wf:
                frames = wf.readframes(wf.getnframes())
                audio = np.frombuffer(frames, dtype=np.int16)
                sd.play(audio, wf.getframerate(), blocking=True)
        except ImportError:
            # sounddevice not available — use OS audio command
            import os
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(wav_bytes)
                tmp_path = f.name
            # tmp_path comes from NamedTemporaryFile — system-controlled, not user input.
            if os.name == "nt":
                os.system(  # noqa: S605
                    f"powershell -c \"(New-Object Media.SoundPlayer '{tmp_path}').PlaySync()\""
                )
            else:
                os.system(f'aplay "{tmp_path}" 2>/dev/null')  # noqa: S605
            os.unlink(tmp_path)

    def _beep(self) -> None:
        """Emit an audible alert beep when TTS is unavailable."""
        try:
            import numpy as np
            import sounddevice as sd

            duration, sample_rate, freq = 0.4, 22050, 880
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            tone = (np.sin(2 * np.pi * freq * t) * 32767 * 0.5).astype(np.int16)
            sd.play(tone, sample_rate, blocking=True)
        except Exception:
            print("\a", end="", flush=True)  # ASCII bell as last resort

    @staticmethod
    def _sanitize(text: str) -> str:
        """Remove control characters and excessive whitespace from TTS input."""
        import re

        # Strip control characters (keep printable ASCII + basic Unicode)
        text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", "", text)
        return " ".join(text.split()).strip()
