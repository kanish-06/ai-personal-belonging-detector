"""
voice_output.py — Text-to-Speech Voice Guidance Module

Converts fuzzy system outputs into short spoken phrases and manages
announcement timing/rate-limiting so the user isn't overwhelmed.

Speech is produced by pyttsx3 (fully offline TTS) running in a short-lived
child process, one per utterance. The child does exactly what a standalone
pyttsx3 test does (init -> say -> runAndWait on its own main thread), so it
does not depend on the COM/audio state of the detection process (torch,
OpenCV camera) and needs no worker thread or message pump in this process.
If the child fails, the error is printed instead of being swallowed.
"""

import os
import sys
import time
import subprocess
import importlib.util

# Only checks that pyttsx3 is installed. It is NOT imported here on purpose:
# the engine must only ever be created inside the child process.
HAS_PYTTSX3 = importlib.util.find_spec("pyttsx3") is not None
if not HAS_PYTTSX3:
    print("[WARNING] pyttsx3 not installed. Voice output will print to console only.")
    print("         Install with: pip install pyttsx3")

_THIS_FILE = os.path.abspath(__file__)
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)  # Windows only, 0 elsewhere


class VoiceOutput:
    """
    Manages spoken voice guidance for the belonging detector.

    Features:
    - Rate-limiting: respects fuzzy frequency output to avoid over-announcing
    - Non-blocking: each phrase is spoken by a child process, so the detection
      loop is never paused
    - Phrase construction: builds natural-sounding guidance phrases
    - No overlap: while a phrase is still being spoken, new announcements are
      skipped (without using up the rate-limit slot), so the next frame
      speaks fresh information as soon as the speaker is free
    - Loud failures: TTS problems are printed, never silently dropped
    """

    # Minimum seconds between announcements at different frequency levels
    # frequency=1.0 -> MIN_INTERVAL, frequency=0.0 -> MAX_INTERVAL
    MIN_INTERVAL = 1.5   # seconds (fastest announcements)
    MAX_INTERVAL = 8.0   # seconds (slowest announcements)

    def __init__(self, rate=160, volume=0.9, voice_index=0):
        """
        Args:
            rate (int): Speech rate in words per minute.
            volume (float): Volume level (0.0 to 1.0).
            voice_index (int): Index of the TTS voice to use (0 = default).
        """
        self._rate = int(rate)
        self._volume = float(volume)
        self._voice_index = int(voice_index)

        self._last_announcement_time = 0.0
        self._last_phrase = ""
        self._proc = None            # currently speaking child process, if any
        self.last_exit_code = None   # exit code of the most recent finished utterance

        self._has_engine = False
        if HAS_PYTTSX3:
            self._has_engine = self._probe_engine()
            if self._has_engine:
                print(f"[VoiceOutput] TTS engine initialized "
                      f"(rate={self._rate}, volume={self._volume})")
        if not self._has_engine:
            print("[VoiceOutput] Running in console-only mode (no TTS engine).")

    @property
    def has_engine(self):
        """True if spoken output is available (self-check passed)."""
        return self._has_engine

    # ── Public announce API (unchanged signatures) ────────────────────────

    def announce(self, class_name, direction, urgency, frequency):
        """
        Generate and speak a guidance phrase, respecting rate-limiting.

        Args:
            class_name (str): Detected object name (e.g., "phone").
            direction (str): Direction label ('left', 'center', 'right').
            urgency (float): Urgency value from fuzzy system (0-1).
            frequency (float): Announcement frequency from fuzzy system (0-1).

        Returns:
            str or None: The spoken phrase, or None if rate-limited.
        """
        if not self._should_announce(frequency):
            return None

        phrase = self._build_phrase(class_name, direction, urgency)
        return self._emit(phrase)

    def announce_with_angle(self, class_name, angle_offset, urgency, frequency, distance=None):
        """
        Like announce(), but accepts raw angle_offset for a more precise direction phrase.

        Args:
            class_name (str): Detected object name.
            angle_offset (float): Raw angle offset (-1 to +1).
            urgency (float): Urgency from fuzzy system (0-1).
            frequency (float): Frequency from fuzzy system (0-1).
            distance (float, optional): Estimated distance in meters.

        Returns:
            str or None: The spoken phrase, or None if rate-limited.
        """
        if not self._should_announce(frequency):
            return None

        # Use CWD-independent import so this works regardless of launch directory
        if os.path.dirname(_THIS_FILE) not in sys.path:
            sys.path.insert(0, os.path.dirname(_THIS_FILE))
        from fuzzy_guidance import generate_direction_phrase
        dir_phrase = generate_direction_phrase(angle_offset)

        phrase = self._compose_phrase(class_name, dir_phrase, urgency, distance)
        return self._emit(phrase)

    def speak_raw(self, text):
        """Speak arbitrary text immediately (no rate limiting, interrupts current speech)."""
        self._speak(text, interrupt=True)

    def announce_startup(self):
        """Speak a startup message."""
        self._speak("Belonging detector is ready. Point the camera to search.")

    def announce_no_objects(self):
        """Speak a 'nothing found' message (rate-limited)."""
        if self._should_announce(0.1):  # very slow rate
            if self._speak("No objects detected. Keep scanning."):
                self._last_announcement_time = time.time()

    def wait_until_idle(self, timeout=None):
        """
        Block until the current utterance (if any) has finished.

        Returns:
            bool: True if idle, False if the timeout expired first.
        """
        if self._proc is not None:
            try:
                self._proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                return False
        self._reap()
        return True

    def shutdown(self):
        """Let the current utterance finish (briefly), then clean up."""
        if self._proc is not None:
            try:
                self._proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
            self._reap()

    # ── Rate limiting and phrase construction (logic unchanged) ───────────

    def _should_announce(self, frequency):
        """
        Check if enough time has passed since the last announcement,
        based on the fuzzy frequency output.

        Args:
            frequency (float): 0 (rarely) to 1 (frequently).

        Returns:
            bool: True if we should announce now.
        """
        # Map frequency (0-1) to interval (MAX_INTERVAL - MIN_INTERVAL)
        interval = self.MAX_INTERVAL - frequency * (self.MAX_INTERVAL - self.MIN_INTERVAL)
        elapsed = time.time() - self._last_announcement_time
        return elapsed >= interval

    def _build_phrase(self, class_name, direction, urgency, distance=None):
        """
        Construct a guidance phrase from a direction label.

        Examples:
            - "Phone found, directly ahead!"
            - "Handbag detected, to your left."
            - "Scanning... Backpack spotted to your right."
        """
        dir_phrases = {
            'left': 'to your left',
            'center': 'directly ahead',
            'right': 'to your right',
        }
        dir_phrase = dir_phrases.get(direction, 'ahead')
        return self._compose_phrase(class_name, dir_phrase, urgency, distance)

    @staticmethod
    def _compose_phrase(class_name, dir_phrase, urgency, distance=None):
        """Build the phrase text; urgency controls the style."""
        obj = class_name.capitalize()

        dist_str = f" at {distance:.1f} meters" if distance is not None and distance > 0 else ""

        if urgency > 0.7:
            return f"{obj} found{dist_str}, {dir_phrase}!"
        elif urgency > 0.4:
            return f"{obj} detected{dist_str}, {dir_phrase}."
        return f"Scanning... {obj} spotted {dir_phrase}{dist_str}."

    def _emit(self, phrase):
        """
        Repeat-suppression + speak + bookkeeping shared by announce*().

        Returns:
            str or None: The phrase if it was spoken, else None.
        """
        # Don't repeat the exact same phrase consecutively
        if phrase == self._last_phrase and (time.time() - self._last_announcement_time) < 3.0:
            return None

        # Still speaking the previous phrase -> skip; do NOT consume the
        # rate-limit slot, so the next call speaks as soon as we're free.
        if not self._speak(phrase):
            return None

        self._last_phrase = phrase
        self._last_announcement_time = time.time()
        return phrase

    # ── Speech backend (child process) ────────────────────────────────────

    def _speak(self, text, interrupt=False):
        """
        Speak text without blocking. Always prints to console.

        Returns:
            bool: True if the phrase was delivered, False if skipped because
                  the previous phrase is still being spoken.
        """
        if self._is_busy():
            if not interrupt:
                return False
            self._proc.kill()
            self._proc.wait()
            self._reap()

        print(f"[VOICE] {text}")

        if self._has_engine:
            self._start_child(text)
        return True

    def _is_busy(self):
        """True while a child process is still speaking."""
        self._reap()
        return self._proc is not None

    def _start_child(self, text):
        cmd = [sys.executable, _THIS_FILE, "--speak", text,
               str(self._rate), str(self._volume), str(self._voice_index)]
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                errors="replace",
                creationflags=_NO_WINDOW,
            )
        except OSError as e:
            self._proc = None
            print(f"[VoiceOutput] Could not start TTS process: {e}")

    def _reap(self):
        """Collect a finished child and report a failure loudly."""
        if self._proc is None:
            return
        code = self._proc.poll()
        if code is None:
            return

        err = ""
        if self._proc.stderr is not None:
            err = self._proc.stderr.read()
            self._proc.stderr.close()
        self.last_exit_code = code
        self._proc = None
        if code != 0:
            print(f"[VoiceOutput] TTS process failed (exit {code}): {err.strip()[-400:]}")

    def _probe_engine(self):
        """Run the child once in --check mode so a broken TTS is reported at startup."""
        try:
            result = subprocess.run(
                [sys.executable, _THIS_FILE, "--check"],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=30,
                creationflags=_NO_WINDOW,
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            print(f"[VoiceOutput] TTS self-check could not run: {e}")
            return False

        if result.returncode != 0:
            print(f"[VoiceOutput] TTS self-check failed (exit {result.returncode}): "
                  f"{result.stderr.strip()[-400:]}")
            return False
        return True


# ─── Child-process entry point ────────────────────────────────────────────────

def _child_main(argv):
    """
    Runs in the child process:
        python voice_output.py --check
        python voice_output.py --speak <text> <rate> <volume> <voice_index>

    This is the plain main-thread pyttsx3 sequence. Any exception propagates,
    producing a traceback on stderr and a non-zero exit code for the parent.
    """
    import pyttsx3

    engine = pyttsx3.init()
    if argv[0] == "--check":
        engine.getProperty('voices')
        return 0

    text, rate, volume, voice_index = argv[1], int(argv[2]), float(argv[3]), int(argv[4])
    engine.setProperty('rate', rate)
    engine.setProperty('volume', volume)
    voices = engine.getProperty('voices')
    if voices and 0 <= voice_index < len(voices):
        engine.setProperty('voice', voices[voice_index].id)
    engine.say(text)
    engine.runAndWait()
    return 0


# ─── Main (test) ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("--speak", "--check"):
        sys.exit(_child_main(sys.argv[1:]))

    print("=== Voice Output — Test ===\n")

    voice = VoiceOutput(rate=160, volume=0.8)

    test_cases = [
        ("mobile phone", "center", 0.9, 0.8),
        ("handbag", "left", 0.5, 0.4),
        ("backpack", "right", 0.2, 0.2),
        ("umbrella", "center", 0.85, 0.9),
        ("luggage", "left", 0.6, 0.5),
    ]

    voice.announce_startup()
    voice.wait_until_idle(30)
    time.sleep(2)

    for cls, direction, urg, freq in test_cases:
        phrase = voice.announce(cls, direction, urg, freq)
        if phrase:
            print(f"  -> Announced: {phrase}")
        else:
            print(f"  -> Rate-limited (skipped)")
        voice.wait_until_idle(30)
        time.sleep(2)

    voice.shutdown()
    print("\nDone.")