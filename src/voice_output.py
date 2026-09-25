"""
voice_output.py — Text-to-Speech Voice Guidance Module

Converts fuzzy system outputs into short spoken phrases and manages
announcement timing/rate-limiting so the user isn't overwhelmed.

Uses pyttsx3 (fully offline TTS) as the primary engine.
"""

import os
import sys
import time
import threading
import queue

try:
    import pyttsx3
    HAS_PYTTSX3 = True
except ImportError:
    HAS_PYTTSX3 = False
    print("[WARNING] pyttsx3 not installed. Voice output will print to console only.")
    print("         Install with: pip install pyttsx3")


class VoiceOutput:
    """
    Manages spoken voice guidance for the belonging detector.

    Features:
    - Rate-limiting: respects fuzzy frequency output to avoid over-announcing
    - Non-blocking: speaks in a background thread so detection loop isn't paused
    - Phrase construction: builds natural-sounding guidance phrases
    - Thread-safe: pyttsx3 engine is created and used entirely within a
      dedicated worker thread to avoid cross-thread COM/driver issues.
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
        self._last_announcement_time = 0.0
        self._last_phrase = ""
        self._speaking = False
        self._has_engine = False

        # Queue-based architecture: main thread enqueues text,
        # worker thread dequeues and speaks using its own pyttsx3 engine.
        self._speech_queue = queue.Queue()
        self._shutdown_event = threading.Event()

        if HAS_PYTTSX3:
            self._tts_thread = threading.Thread(
                target=self._tts_worker,
                args=(rate, volume, voice_index),
                daemon=True,
            )
            self._tts_thread.start()
            # Give the worker a moment to initialize
            time.sleep(0.3)
            self._has_engine = True
            print(f"[VoiceOutput] TTS engine initialized (rate={rate}, volume={volume})")
        else:
            self._tts_thread = None
            print("[VoiceOutput] Running in console-only mode (no TTS engine).")

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
        # Check rate limiting
        if not self._should_announce(frequency):
            return None

        # Build the phrase
        phrase = self._build_phrase(class_name, direction, urgency)

        # Don't repeat the exact same phrase consecutively
        if phrase == self._last_phrase and (time.time() - self._last_announcement_time) < 3.0:
            return None

        # Speak it
        self._speak(phrase)
        self._last_phrase = phrase
        self._last_announcement_time = time.time()

        return phrase

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

    def _build_phrase(self, class_name, direction, urgency):
        """
        Construct a natural-sounding guidance phrase.

        Examples:
            - "Phone found, directly ahead!"
            - "Handbag detected, to your left."
            - "Backpack spotted, to your right."

        Args:
            class_name (str): Object name.
            direction (str): 'left', 'center', or 'right'.
            urgency (float): Controls phrase style (0-1).

        Returns:
            str: The guidance phrase.
        """
        obj = class_name.capitalize()

        dir_phrases = {
            'left': 'to your left',
            'center': 'directly ahead',
            'right': 'to your right',
        }
        dir_phrase = dir_phrases.get(direction, 'ahead')

        # Construct phrase based on urgency level
        if urgency > 0.7:
            phrase = f"{obj} found, {dir_phrase}!"
        elif urgency > 0.4:
            phrase = f"{obj} detected, {dir_phrase}."
        else:
            phrase = f"Scanning... {obj} spotted {dir_phrase}."

        return phrase

    def announce_with_angle(self, class_name, angle_offset, urgency, frequency):
        """
        Like announce(), but accepts raw angle_offset for a more precise direction phrase.

        Args:
            class_name (str): Detected object name.
            angle_offset (float): Raw angle offset (-1 to +1).
            urgency (float): Urgency from fuzzy system (0-1).
            frequency (float): Frequency from fuzzy system (0-1).

        Returns:
            str or None: The spoken phrase, or None if rate-limited.
        """
        if not self._should_announce(frequency):
            return None

        # Use CWD-independent import so this works regardless of launch directory
        _src_dir = os.path.dirname(os.path.abspath(__file__))
        if _src_dir not in sys.path:
            sys.path.insert(0, _src_dir)
        from fuzzy_guidance import generate_direction_phrase
        dir_phrase = generate_direction_phrase(angle_offset)

        obj = class_name.capitalize()

        if urgency > 0.7:
            phrase = f"{obj} found, {dir_phrase}!"
        elif urgency > 0.4:
            phrase = f"{obj} detected, {dir_phrase}."
        else:
            phrase = f"Scanning... {obj} spotted {dir_phrase}."

        if phrase == self._last_phrase and (time.time() - self._last_announcement_time) < 3.0:
            return None

        self._speak(phrase)
        self._last_phrase = phrase
        self._last_announcement_time = time.time()

        return phrase

    def speak_raw(self, text):
        """Speak arbitrary text immediately (no rate limiting)."""
        self._speak(text)

    def _speak(self, text):
        """
        Speak text using TTS engine (non-blocking).
        Falls back to console print if engine is unavailable.
        """
        print(f"[VOICE] {text}")

        if self._has_engine and not self._speaking:
            # Enqueue for the dedicated TTS worker thread
            self._speech_queue.put(text)

    def _tts_worker(self, rate, volume, voice_index):
        """
        Dedicated worker thread that owns the pyttsx3 engine.
        pyttsx3 is not thread-safe — the engine must be created and used
        entirely within a single thread.
        """
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', rate)
            engine.setProperty('volume', volume)
            voices = engine.getProperty('voices')
            if voices and voice_index < len(voices):
                engine.setProperty('voice', voices[voice_index].id)
        except Exception as e:
            print(f"[VoiceOutput] Failed to initialize TTS engine: {e}")
            self._has_engine = False
            return

        while not self._shutdown_event.is_set():
            try:
                text = self._speech_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            self._speaking = True
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"[VoiceOutput] TTS error: {e}")
            finally:
                self._speaking = False

        # Clean shutdown of the engine
        try:
            engine.stop()
        except Exception:
            pass

    def announce_startup(self):
        """Speak a startup message."""
        self._speak("Belonging detector is ready. Point the camera to search.")

    def announce_no_objects(self):
        """Speak a 'nothing found' message (rate-limited)."""
        if self._should_announce(0.1):  # very slow rate
            self._speak("No objects detected. Keep scanning.")
            self._last_announcement_time = time.time()

    def shutdown(self):
        """Clean up the TTS engine and worker thread."""
        self._shutdown_event.set()
        if self._tts_thread is not None:
            self._tts_thread.join(timeout=3.0)


# ─── Main (test) ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
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
    time.sleep(2)

    for cls, direction, urg, freq in test_cases:
        phrase = voice.announce(cls, direction, urg, freq)
        if phrase:
            print(f"  -> Announced: {phrase}")
        else:
            print(f"  -> Rate-limited (skipped)")
        time.sleep(2)

    voice.shutdown()
    print("\nDone.")
