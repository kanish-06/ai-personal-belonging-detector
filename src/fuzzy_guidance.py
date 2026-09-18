"""
fuzzy_guidance.py — Mamdani Fuzzy Inference System for Voice Guidance

Takes perception features (angle, confidence, stability) and outputs
guidance parameters (urgency, announcement frequency) that control how the
system speaks to the user.

Uses scikit-fuzzy (skfuzzy) for the Mamdani-style FIS implementation.
"""

import numpy as np

try:
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
    HAS_SKFUZZY = True
except ImportError:
    HAS_SKFUZZY = False
    print("[WARNING] scikit-fuzzy not installed. Fuzzy guidance will use fallback logic.")
    print("         Install with: pip install scikit-fuzzy")


class FuzzyGuidanceSystem:
    """
    Mamdani-style Fuzzy Inference System for guidance decision-making.

    Inputs:
        - angle (-1 to +1):       Left / Center / Right
        - confidence (0 to 1):    Low / Medium / High
        - stability (0 to 1):     Unstable / Stable

    Outputs:
        - urgency (0 to 1):       Low / Medium / High
        - frequency (0 to 1):     Slow / Fast
    """

    def __init__(self):
        """Initialize the fuzzy inference system with membership functions and rules."""
        if not HAS_SKFUZZY:
            print("[FuzzyGuidance] Running in fallback mode (no scikit-fuzzy).")
            self._use_fallback = True
            return

        self._use_fallback = False
        self._build_system()

    def _build_system(self):
        """Construct the complete fuzzy system: variables, MFs, rules, and control system."""

        # ─── Input Variables (Antecedents) ──────────────────────────────────

        # Angle offset (-1 to +1): Left / Center / Right
        self.angle = ctrl.Antecedent(np.arange(-1.0, 1.01, 0.01), 'angle')
        self.angle['left'] = fuzz.trapmf(self.angle.universe, [-1.0, -1.0, -0.5, -0.1])
        self.angle['center'] = fuzz.trimf(self.angle.universe, [-0.3, 0.0, 0.3])
        self.angle['right'] = fuzz.trapmf(self.angle.universe, [0.1, 0.5, 1.0, 1.0])

        # Confidence (0 to 1): Low / Medium / High
        self.confidence = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'confidence')
        self.confidence['low'] = fuzz.trapmf(self.confidence.universe, [0, 0, 0.3, 0.5])
        self.confidence['medium'] = fuzz.trimf(self.confidence.universe, [0.35, 0.55, 0.75])
        self.confidence['high'] = fuzz.trapmf(self.confidence.universe, [0.6, 0.8, 1.0, 1.0])

        # Stability (0 to 1): Unstable / Stable
        self.stability = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'stability')
        self.stability['unstable'] = fuzz.trapmf(self.stability.universe, [0, 0, 0.25, 0.5])
        self.stability['stable'] = fuzz.trapmf(self.stability.universe, [0.4, 0.6, 1.0, 1.0])

        # ─── Output Variables (Consequents) ─────────────────────────────────

        # Urgency (0 to 1): how urgently to speak
        self.urgency = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'urgency')
        self.urgency['low'] = fuzz.trimf(self.urgency.universe, [0, 0.15, 0.35])
        self.urgency['medium'] = fuzz.trimf(self.urgency.universe, [0.25, 0.5, 0.75])
        self.urgency['high'] = fuzz.trimf(self.urgency.universe, [0.65, 0.85, 1.0])

        # Frequency (0 to 1): how often to announce (0=rarely, 1=frequently)
        self.frequency = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'frequency')
        self.frequency['slow'] = fuzz.trimf(self.frequency.universe, [0, 0.2, 0.45])
        self.frequency['fast'] = fuzz.trimf(self.frequency.universe, [0.55, 0.8, 1.0])

        # ─── Rules ─────────────────────────────────────────────────────────

        # Rule 1: High confidence + Center + Stable -> High urgency, Fast frequency
        rule1 = ctrl.Rule(
            self.confidence['high'] & self.angle['center'] & self.stability['stable'],
            (self.urgency['high'], self.frequency['fast']),
        )

        # Rule 2: High confidence + Left/Right + Stable -> High urgency, Fast frequency
        rule2 = ctrl.Rule(
            self.confidence['high'] & (self.angle['left'] | self.angle['right']) & self.stability['stable'],
            (self.urgency['high'], self.frequency['fast']),
        )

        # Rule 3: Medium confidence + Stable -> Medium urgency, Slow frequency
        rule3 = ctrl.Rule(
            self.confidence['medium'] & self.stability['stable'],
            (self.urgency['medium'], self.frequency['slow']),
        )

        # Rule 4: High confidence + Unstable -> Medium urgency, Slow frequency
        rule4 = ctrl.Rule(
            self.confidence['high'] & self.stability['unstable'],
            (self.urgency['medium'], self.frequency['slow']),
        )

        # Rule 5: Low confidence -> Low urgency, Slow frequency
        rule5 = ctrl.Rule(
            self.confidence['low'],
            (self.urgency['low'], self.frequency['slow']),
        )

        # Rule 6: Unstable + Low/Medium confidence -> Low urgency, Slow frequency
        rule6 = ctrl.Rule(
            self.stability['unstable'] & (self.confidence['low'] | self.confidence['medium']),
            (self.urgency['low'], self.frequency['slow']),
        )

        # ─── Control System ────────────────────────────────────────────────

        self._ctrl = ctrl.ControlSystem([
            rule1, rule2, rule3, rule4, rule5, rule6,
        ])
        self._sim = ctrl.ControlSystemSimulation(self._ctrl)

        print("[FuzzyGuidance] Fuzzy Inference System initialized with 6 Mamdani rules.")

    def compute(self, angle_offset, confidence, stability):
        """
        Run the fuzzy inference system.

        Args:
            angle_offset (float): Horizontal offset (-1 to +1).
            confidence (float): Detection confidence (0-1).
            stability (float): Temporal stability (0-1).

        Returns:
            dict: {
                'urgency': float (0-1),
                'frequency': float (0-1),
                'direction': str ('left', 'center', 'right'),
                'should_announce': bool
            }
        """
        if self._use_fallback:
            return self._fallback_compute(angle_offset, confidence, stability)

        # Clamp inputs to valid ranges
        angle_offset = np.clip(angle_offset, -1.0, 1.0)
        confidence = np.clip(confidence, 0.0, 1.0)
        stability = np.clip(stability, 0.0, 1.0)

        try:
            self._sim.input['angle'] = angle_offset
            self._sim.input['confidence'] = confidence
            self._sim.input['stability'] = stability

            self._sim.compute()

            urgency = float(self._sim.output['urgency'])
            frequency = float(self._sim.output['frequency'])
        except Exception as e:
            # Fallback if FIS computation fails (e.g., no rules fire)
            print(f"[FuzzyGuidance] FIS computation error: {e}. Using fallback.")
            return self._fallback_compute(angle_offset, confidence, stability)

        # Derive direction label from angle_offset
        direction = self._get_direction(angle_offset)

        # Decide whether to announce (hard threshold on confidence and urgency)
        should_announce = confidence >= 0.30 and urgency > 0.15

        return {
            'urgency': urgency,
            'frequency': frequency,
            'direction': direction,
            'should_announce': should_announce,
        }

    def _get_direction(self, angle_offset):
        """Map angle offset to a human-readable direction."""
        if angle_offset < -0.25:
            return 'left'
        elif angle_offset > 0.25:
            return 'right'
        else:
            return 'center'

    def _fallback_compute(self, angle_offset, confidence, stability):
        """
        Simple rule-based fallback when scikit-fuzzy is not available.
        Approximates the fuzzy system's behavior with hard thresholds.
        """
        direction = self._get_direction(angle_offset)

        if confidence < 0.3 or stability < 0.3:
            urgency = 0.1
            frequency = 0.2
        elif confidence > 0.7 and stability > 0.6:
            urgency = 0.85
            frequency = 0.8
        else:
            urgency = 0.5
            frequency = 0.4

        should_announce = confidence >= 0.30 and urgency > 0.15

        return {
            'urgency': urgency,
            'frequency': frequency,
            'direction': direction,
            'should_announce': should_announce,
        }


# ─── Direction Phrase Generator ─────────────────────────────────────────────

def generate_direction_phrase(angle_offset):
    """
    Convert an angle offset value into a relative direction phrase.

    Args:
        angle_offset (float): -1 (far left) to +1 (far right).

    Returns:
        str: e.g., "directly ahead", "to your left", "far to your right"
    """
    if angle_offset < -0.6:
        return "far to your left"
    elif angle_offset < -0.25:
        return "to your left"
    elif angle_offset < -0.08:
        return "slightly to your left"
    elif angle_offset <= 0.08:
        return "directly ahead"
    elif angle_offset <= 0.25:
        return "slightly to your right"
    elif angle_offset <= 0.6:
        return "to your right"
    else:
        return "far to your right"


# ─── Main (test) ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Fuzzy Guidance System — Test Cases ===\n")

    fis = FuzzyGuidanceSystem()

    test_cases = [
        # (angle_offset, confidence, stability, description)
        (0.0, 0.92, 0.9, "Center, high-conf, stable -> HIGH urgency"),
        (-0.5, 0.88, 0.8, "Left, high-conf, stable -> HIGH urgency, left"),
        (0.6, 0.90, 0.85, "Right, high-conf, stable -> HIGH urgency, right"),
        (0.1, 0.55, 0.7, "Center, medium-conf, stable -> MEDIUM urgency"),
        (0.0, 0.85, 0.2, "High-conf but UNSTABLE -> MEDIUM/LOW urgency"),
        (0.0, 0.20, 0.8, "Stable but LOW confidence -> suppress"),
    ]

    for angle, conf, stab, desc in test_cases:
        result = fis.compute(angle, conf, stab)
        dir_phrase = generate_direction_phrase(angle)
        print(f"  {desc}")
        print(f"    Inputs:  angle={angle:.1f}, conf={conf:.2f}, stab={stab:.2f}")
        print(f"    Outputs: urgency={result['urgency']:.2f}, freq={result['frequency']:.2f}, "
              f"dir={result['direction']}, announce={result['should_announce']}")
        print(f"    Phrase:  \"{dir_phrase}\"")
        print()
