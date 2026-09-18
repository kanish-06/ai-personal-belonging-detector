"""
voice_output.py — Text-to-Speech Voice Guidance Module

Converts fuzzy system outputs into short spoken phrases and manages
announcement timing/rate-limiting so the user isn't overwhelmed.

Uses pyttsx3 (fully offline TTS) as the primary engine.
"""

import time
import threading

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
    """

    # Distance label → spoken phrase fragment
    DISTANCE_PHRASES = {
        'near': 'close by',
        'medium': 'a few steps away',
        'far': 'far away',
    }

    # Minimum seconds between announcements at different frequency levels
    # frequency=1.0 → MIN_INTERVAL, frequency=0.0 → MAX_INTERVAL
    MIN_INTERVAL = 1.5   # seconds (fastest announcements)
    MAX_INTERVAL = 8.0   # seconds (slowest announcements)

    def __init__(self, rate=160, volume=0.9, voice_index=0):
        """
        Args:
            rate (int): Speech rate in words per minute.
            volume (float): Volume level (0.0 to 1.0).
            voice_index (int): Index of the TTS voice to use (0 = default).
        """
        self._engine = None
        self._lock = threading.Lock()
        self._last_announcement_time = 0.0
        self._last_phrase = ""
        self._speaking = False

        if HAS_PYTTSX3:
            try:
                self._engine = pyttsx3.init()
                self._engine.setProperty('rate', rate)
                self._engine.setProperty('volume', volume)

                # Try to set a specific voice
                voices = self._engine.getProperty('voices')
                if voices and voice_index < len(voices):
                    self._engine.setProperty('voice', voices[voice_index].id)

                print(f"[VoiceOutput] TTS engine initialized (rate={rate}, volume={volume})")
            except Exception as e:
                print(f"[VoiceOutput] Failed to initialize TTS engine: {e}")
                self._engine = None
        else:
            print("[VoiceOutput] Running in console-only mode (no TTS engine).")

    def announce(self, class_name, direction, distance_label, urgency, frequency):
        """
        Generate and speak a guidance phrase, respecting rate-limiting.

        Args:
            class_name (str): Detected object name (e.g., "wallet").
            direction (str): Direction label ('left', 'center', 'right').
            distance_label (str): Distance label ('near', 'medium', 'far').
            urgency (float): Urgency value from fuzzy system (0-1).
            frequency (float): Announcement frequency from fuzzy system (0-1).

        Returns:
            str or None: The spoken phrase, or None if rate-limited.
        """
        # Check rate limiting
        if not self._should_announce(frequency):
            return None

        # Build the phrase
        phrase = self._build_phrase(class_name, direction, distance_label, urgency)

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

    def _build_phrase(self, class_name, direction, distance_label, urgency):
        """
        Construct a natural-sounding guidance phrase.

        Examples:
            - "Phone found, directly ahead, close by."
            - "Wallet detected, to your left, a few steps away."
            - "Keys nearby, slightly to your right."

        Args:
            class_name (str): Object name.
            direction (str): 'left', 'center', or 'right'.
            distance_label (str): 'near', 'medium', or 'far'.
            urgency (float): Controls phrase style (0-1).

        Returns:
            str: The guidance phrase.
        """
        # Object name (capitalize first letter)
        obj = class_name.capitalize()

        # Direction phrase
        from fuzzy_guidance import generate_direction_phrase
        # We need the raw angle for the detailed phrase, but we only have
        # the direction label here. Use a simplified mapping.
        dir_phrases = {
            'left': 'to your left',
            'center': 'directly ahead',
            'right': 'to your right',
        }
        dir_phrase = dir_phrases.get(direction, 'ahead')

        # Distance phrase
        dist_phrase = self.DISTANCE_PHRASES.get(distance_label, '')

        # Construct phrase based on urgency level
        if urgency > 0.7:
            # High urgency — short, direct
            if distance_label == 'near':
                phrase = f"{obj} found, {dir_phrase}, reach forward!"
            else:
                phrase = f"{obj} found, {dir_phrase}, {dist_phrase}."
        elif urgency > 0.4:
            # Medium urgency — informational
            phrase = f"{obj} detected, {dir_phrase}, {dist_phrase}."
        else:
            # Low urgency — casual
            phrase = f"Scanning... {obj} spotted {dist_phrase}."

        return phrase

    def announce_with_angle(self, class_name, angle_offset, distance_label, urgency, frequency):
        """
        Like announce(), but accepts raw angle_offset for a more precise direction phrase.

        Args:
            class_name (str): Detected object name.
            angle_offset (float): Raw angle offset (-1 to +1).
            distance_label (str): 'near', 'medium', 'far'.
            urgency (float): Urgency from fuzzy system (0-1).
            frequency (float): Frequency from fuzzy system (0-1).

        Returns:
            str or None: The spoken phrase, or None if rate-limited.
        """
        if not self._should_announce(frequency):
            return None

        from fuzzy_guidance import generate_direction_phrase
        dir_phrase = generate_direction_phrase(angle_offset)

        obj = class_name.capitalize()
        dist_phrase = self.DISTANCE_PHRASES.get(distance_label, '')

        if urgency > 0.7:
            if distance_label == 'near':
                phrase = f"{obj} found, {dir_phrase}, reach forward!"
            else:
                phrase = f"{obj} found, {dir_phrase}, {dist_phrase}."
        elif urgency > 0.4:
            phrase = f"{obj} detected, {dir_phrase}, {dist_phrase}."
        else:
            phrase = f"Scanning... {obj} spotted {dist_phrase}."

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

        if self._engine is not None and not self._speaking:
            thread = threading.Thread(target=self._speak_thread, args=(text,), daemon=True)
            thread.start()

    def _speak_thread(self, text):
        """Run TTS in a background thread to avoid blocking the main loop."""
        with self._lock:
            self._speaking = True
            try:
                self._engine.say(text)
                self._engine.runAndWait()
            except Exception as e:
                print(f"[VoiceOutput] TTS error: {e}")
            finally:
                self._speaking = False

    def announce_startup(self):
        """Speak a startup message."""
        self._speak("Belonging detector is ready. Point the camera to search.")

    def announce_no_objects(self):
        """Speak a 'nothing found' message (rate-limited)."""
        if self._should_announce(0.1):  # very slow rate
            self._speak("No objects detected. Keep scanning.")
            self._last_announcement_time = time.time()

    def shutdown(self):
        """Clean up the TTS engine."""
        if self._engine is not None:
            try:
                self._engine.stop()
            except Exception:
                pass


# ─── Main (test) ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Voice Output — Test ===\n")

    voice = VoiceOutput(rate=160, volume=0.8)

    # Test phrase generation
    test_cases = [
        ("phone", "center", "near", 0.9, 0.8),
        ("wallet", "left", "medium", 0.5, 0.4),
        ("keys", "right", "far", 0.2, 0.2),
        ("watch", "center", "near", 0.85, 0.9),
        ("glasses", "left", "medium", 0.6, 0.5),
    ]

    voice.announce_startup()
    time.sleep(2)

    for cls, direction, dist, urg, freq in test_cases:
        phrase = voice.announce(cls, direction, dist, urg, freq)
        if phrase:
            print(f"  → Announced: {phrase}")
        else:
            print(f"  → Rate-limited (skipped)")
        time.sleep(2)  # wait between announcements for demo

    voice.shutdown()
    print("\nDone.")
