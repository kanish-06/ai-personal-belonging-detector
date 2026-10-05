"""
test_voice_pipeline_path.py — proves speech is actually produced through the
same VoiceOutput calls that pipeline.py makes.

Run from src/:
    python test_voice_pipeline_path.py            # voice path only
    python test_voice_pipeline_path.py --camera   # also open the webcam first,
                                                  # so the process is in the same
                                                  # state as the real pipeline

What is checked automatically:
  * the TTS self-check passes (has_engine)
  * every utterance's child process exits 0
  * every utterance takes > 1 s of wall time (a silent runAndWait that returns
    instantly would fail here)
  * rate limiting still works (frequency=0.0 right after a phrase -> None)
Then you confirm by ear.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    if "--camera" in sys.argv:
        import cv2
        cap = cv2.VideoCapture(0)
        ok, _ = cap.read()
        print(f"[test] camera opened={cap.isOpened()} frame_ok={ok}")

    from voice_output import VoiceOutput

    voice = VoiceOutput(rate=160, volume=0.9)   # same args as pipeline.py
    assert voice.has_engine, "TTS self-check failed - see the message printed above"

    # 1) pipeline.run() -> announce_startup()
    t = time.time()
    voice.announce_startup()
    assert voice.wait_until_idle(30), "startup phrase never finished"
    dt = time.time() - t
    print(f"[test] startup phrase: {dt:.1f}s, exit={voice.last_exit_code}")
    assert voice.last_exit_code == 0, "TTS child process failed"
    assert dt > 1.0, "finished too fast - nothing was actually spoken"

    # 2) pipeline loop -> announce_with_angle(...) with the same keyword args
    t = time.time()
    phrase = voice.announce_with_angle(
        class_name="phone", angle_offset=-0.4, urgency=0.9, frequency=1.0,
    )
    assert phrase is not None, "announce_with_angle was skipped unexpectedly"
    assert voice.wait_until_idle(30), "detection phrase never finished"
    dt = time.time() - t
    print(f"[test] detection phrase {phrase!r}: {dt:.1f}s, exit={voice.last_exit_code}")
    assert voice.last_exit_code == 0, "TTS child process failed"
    assert dt > 1.0, "finished too fast - nothing was actually spoken"

    # 3) rate limiting preserved: frequency=0.0 means 8 s interval, so this must be skipped
    again = voice.announce_with_angle(
        class_name="phone", angle_offset=-0.4, urgency=0.9, frequency=0.0,
    )
    assert again is None, "rate limiting broke"
    print("[test] rate limiting OK")

    voice.shutdown()

    answer = input("\nDid you HEAR both phrases (startup + 'Phone found...')? [y/n] ")
    assert answer.strip().lower().startswith("y"), "no audible speech"
    print("\nPASS")


if __name__ == "__main__":
    main()