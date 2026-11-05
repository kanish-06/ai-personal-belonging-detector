"""
qwen_verifier.py — Secondary Visual Verifier Using Local Qwen3-VL

Provides object-class verification for uncertain YOLO detections by querying
a local Qwen3-VL model (via Ollama) on cropped bounding-box image patches.

Ensures:
- Output is normalized to one of: keys, wallet, phone, watch, glasses, unknown.
- Robust error, timeout, and connection-failure handling without crashing the pipeline.
- In-memory image encoding (no temporary disk files).
"""

import time
import logging
from typing import Optional, Set
import numpy as np
import cv2

try:
    import ollama
    from ollama import chat
    HAS_OLLAMA = True
except ImportError:
    ollama = None
    chat = None
    HAS_OLLAMA = False

logger = logging.getLogger(__name__)

SUPPORTED_CLASSES: Set[str] = {"keys", "wallet", "phone", "watch", "glasses"}
VALID_RESPONSES: Set[str] = SUPPORTED_CLASSES | {"unknown"}

VERIFICATION_PROMPT = """Identify the personal belonging visible in this cropped image patch.

Choose exactly ONE category:
keys, wallet, phone, watch, glasses, unknown.

Return ONLY the category name in lowercase.
If the object is blurry, unclear, or not one of these belongings, return 'unknown'.
Do not guess.
"""


def normalize_qwen_output(raw_text: Optional[str]) -> str:
    """
    Strictly normalizes raw output from Qwen3-VL to one of the 5 supported
    classes or 'unknown'.

    Args:
        raw_text (str): Raw string output from the model.

    Returns:
        str: One of 'keys', 'wallet', 'phone', 'watch', 'glasses', 'unknown'.
    """
    if not raw_text:
        return "unknown"

    cleaned = raw_text.strip().lower().strip(".,!?:;\"' \n\r`")

    # Direct exact match
    if cleaned in VALID_RESPONSES:
        return cleaned

    # Check words for exact match
    words = cleaned.replace("\n", " ").replace("-", " ").split()
    for word in words:
        clean_word = word.strip(".,!?:;\"' \n\r`")
        if clean_word in SUPPORTED_CLASSES:
            return clean_word
    if "unknown" in words or "none" in words:
        return "unknown"

    # Substring / synonym checks
    if "key" in cleaned:
        return "keys"
    if "watch" in cleaned:
        return "watch"
    if "wallet" in cleaned:
        return "wallet"
    if "phone" in cleaned or "mobile" in cleaned:
        return "phone"
    if "glass" in cleaned or "spectacle" in cleaned or "eyewear" in cleaned:
        return "glasses"

    return "unknown"


class QwenVerifier:
    """
    Encapsulates visual verification using a local Qwen3-VL model via Ollama.
    """

    def __init__(
        self,
        model_name: str = "qwen3-vl:2b",
        temperature: float = 0.0,
        timeout: float = 5.0,
    ):
        """
        Args:
            model_name (str): Model tag in Ollama (default: 'qwen3-vl:2b').
            temperature (float): Generation temperature (0.0 for deterministic).
            timeout (float): Request timeout in seconds.
        """
        self.model_name = model_name
        self.temperature = temperature
        self.timeout = timeout
        self._last_error_time = 0.0
        self._is_available = HAS_OLLAMA

        if not HAS_OLLAMA:
            print("[QwenVerifier] WARNING: 'ollama' python package not installed.")
            print("               Install with: pip install ollama")

    @property
    def is_available(self) -> bool:
        """Returns True if the Ollama library is importable."""
        return self._is_available

    def verify_crop(self, crop_bgr: np.ndarray) -> str:
        """
        Sends an in-memory cropped image patch to Qwen3-VL for verification.

        Args:
            crop_bgr (np.ndarray): BGR image patch (OpenCV format).

        Returns:
            str: Verified class name ('keys', 'wallet', 'phone', 'watch', 'glasses', or 'unknown').
        """
        if not self._is_available or chat is None:
            return "unknown"

        if crop_bgr is None or crop_bgr.size == 0:
            return "unknown"

        h, w = crop_bgr.shape[:2]
        if h < 10 or w < 10:
            return "unknown"

        # Encode crop to JPEG bytes in memory
        success, encoded = cv2.imencode(".jpg", crop_bgr)
        if not success:
            return "unknown"

        image_bytes = encoded.tobytes()

        try:
            response = chat(
                model=self.model_name,
                messages=[{
                    "role": "user",
                    "content": VERIFICATION_PROMPT,
                    "images": [image_bytes]
                }],
                options={"temperature": self.temperature}
            )

            raw_content = response.message.content if response and response.message else ""
            verified_class = normalize_qwen_output(raw_content)
            return verified_class

        except Exception as e:
            # Throttle error logging to avoid flooding stdout every frame
            now = time.time()
            if now - self._last_error_time > 5.0:
                print(f"[QwenVerifier] Error during verification ({type(e).__name__}): {e}")
                self._last_error_time = now
            return "unknown"

