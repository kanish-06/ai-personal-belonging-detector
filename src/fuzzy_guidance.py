"""
fuzzy_guidance.py — Mamdani Fuzzy Inference System for Voice Guidance

Takes perception features (distance, angle, confidence, stability) and outputs
guidance parameters (urgency, announcement frequency) that control how the
system speaks to the user.

This is the "fuzzy half" of the neuro-fuzzy architecture: it converts noisy,
uncertain detector outputs into smooth, human-friendly guidance decisions.

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
        - distance (cm):          Near / Medium / Far
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

        # Distance (from ANFIS, in cm): 0 to 350
        self.distance = ctrl.Antecedent(np.arange(0, 351, 1), 'distance')
        self.distance['near'] = fuzz.trapmf(self.distance.universe, [0, 0, 30, 70])
        self.distance['medium'] = fuzz.trimf(self.distance.universe, [40, 90, 160])
        self.distance['far'] = fuzz.trapmf(self.distance.universe, [120, 200, 350, 350])

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

        # Rule 1: Near + Center + High confidence → High urgency, Fast frequency
        rule1 = ctrl.Rule(
            self.distance['near'] & self.angle['center'] & self.confidence['high'],
            (self.urgency['high'], self.frequency['fast']),
        )

        # Rule 2: Near + Left + High confidence → High urgency, Fast frequency
        rule2 = ctrl.Rule(
            self.distance['near'] & self.angle['left'] & self.confidence['high'],
            (self.urgency['high'], self.frequency['fast']),
        )

        # Rule 3: Near + Right + High confidence → High urgency, Fast frequency
        rule3 = ctrl.Rule(
            self.distance['near'] & self.angle['right'] & self.confidence['high'],
            (self.urgency['high'], self.frequency['fast']),
        )

        # Rule 4: Medium distance + Medium/High confidence → Medium urgency, Slow frequency
        rule4a = ctrl.Rule(
            self.distance['medium'] & self.confidence['medium'],
            (self.urgency['medium'], self.frequency['slow']),
        )
        rule4b = ctrl.Rule(
            self.distance['medium'] & self.confidence['high'],
            (self.urgency['medium'], self.frequency['slow']),
        )

        # Rule 5: Far distance → Low urgency, Slow frequency
        rule5 = ctrl.Rule(
            self.distance['far'],
            (self.urgency['low'], self.frequency['slow']),
        )

        # Rule 6: Unstable → Low urgency (suppress flickering detections)
        rule6 = ctrl.Rule(
            self.stability['unstable'],
            (self.urgency['low'], self.frequency['slow']),
        )

        # Rule 7: Low confidence → Low urgency, Slow frequency
        rule7 = ctrl.Rule(
            self.confidence['low'],
            (self.urgency['low'], self.frequency['slow']),
        )

        # Rule 8: Near + Medium confidence + Stable → Medium urgency, Fast frequency
        rule8 = ctrl.Rule(
            self.distance['near'] & self.confidence['medium'] & self.stability['stable'],
            (self.urgency['medium'], self.frequency['fast']),
        )

        # Rule 9: Medium distance + High confidence + Stable → Medium urgency, Slow frequency
        rule9 = ctrl.Rule(
            self.distance['medium'] & self.confidence['high'] & self.stability['stable'],
            (self.urgency['medium'], self.frequency['slow']),
        )

        # ─── Control System ────────────────────────────────────────────────

        self._ctrl = ctrl.ControlSystem([
            rule1, rule2, rule3, rule4a, rule4b,
            rule5, rule6, rule7, rule8, rule9,
        ])
        self._sim = ctrl.ControlSystemSimulation(self._ctrl)

        print("[FuzzyGuidance] Fuzzy Inference System initialized with 10 rules.")

    def compute(self, distance_cm, angle_offset, confidence, stability):
        """
        Run the fuzzy inference system.

        Args:
            distance_cm (float): Estimated distance in cm (0-350+).
            angle_offset (float): Horizontal offset (-1 to +1).
            confidence (float): Detection confidence (0-1).
            stability (float): Temporal stability (0-1).

        Returns:
            dict: {
                'urgency': float (0-1),
                'frequency': float (0-1),
                'direction': str ('left', 'center', 'right'),
                'distance_label': str ('near', 'medium', 'far'),
                'should_announce': bool
            }
        """
        if self._use_fallback:
            return self._fallback_compute(distance_cm, angle_offset, confidence, stability)

        # Clamp inputs to valid ranges
        distance_cm = np.clip(distance_cm, 0, 350)
        angle_offset = np.clip(angle_offset, -1.0, 1.0)
        confidence = np.clip(confidence, 0.0, 1.0)
        stability = np.clip(stability, 0.0, 1.0)

        try:
            self._sim.input['distance'] = distance_cm
            self._sim.input['angle'] = angle_offset
            self._sim.input['confidence'] = confidence
            self._sim.input['stability'] = stability

            self._sim.compute()

            urgency = float(self._sim.output['urgency'])
            frequency = float(self._sim.output['frequency'])
        except Exception as e:
            # Fallback if FIS computation fails (e.g., no rules fire)
            print(f"[FuzzyGuidance] FIS computation error: {e}. Using fallback.")
            return self._fallback_compute(distance_cm, angle_offset, confidence, stability)

        # Derive direction label from angle_offset
        direction = self._get_direction(angle_offset)

        # Derive distance label
        distance_label = self._get_distance_label(distance_cm)

        # Decide whether to announce (hard threshold on confidence)
        should_announce = confidence >= 0.30 and urgency > 0.15

        return {
            'urgency': urgency,
            'frequency': frequency,
            'direction': direction,
            'distance_label': distance_label,
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

    def _get_distance_label(self, distance_cm):
        """Map distance to a human-readable label."""
        if distance_cm < 60:
            return 'near'
        elif distance_cm < 150:
            return 'medium'
        else:
            return 'far'

    def _fallback_compute(self, distance_cm, angle_offset, confidence, stability):
        """
        Simple rule-based fallback when scikit-fuzzy is not available.
        Approximates the fuzzy system's behavior with hard thresholds.
        """
        # Direction
        direction = self._get_direction(angle_offset)
        distance_label = self._get_distance_label(distance_cm)

        # Urgency
        if confidence < 0.3 or stability < 0.3:
            urgency = 0.1
        elif distance_label == 'near' and confidence > 0.6:
            urgency = 0.85
        elif distance_label == 'medium':
            urgency = 0.5
        else:
            urgency = 0.2

        # Frequency
        if distance_label == 'near' and confidence > 0.5:
            frequency = 0.8
        else:
            frequency = 0.3

        should_announce = confidence >= 0.30 and urgency > 0.15

        return {
            'urgency': urgency,
            'frequency': frequency,
            'direction': direction,
            'distance_label': distance_label,
            'should_announce': should_announce,
        }


# ─── Direction Phrase Generator ─────────────────────────────────────────────

def generate_direction_phrase(angle_offset):
    """
    Convert an angle offset value into a clock-position or relative phrase.

    Args:
        angle_offset (float): -1 (far left) to +1 (far right).

    Returns:
        str: e.g., "directly ahead", "to your left", "at two o'clock"
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
        # (distance_cm, angle_offset, confidence, stability, description)
        (30, 0.0, 0.92, 0.9, "Near, center, high-conf, stable — should be HIGH urgency"),
        (30, -0.5, 0.88, 0.8, "Near, left, high-conf — should be HIGH urgency, left"),
        (30, 0.6, 0.90, 0.85, "Near, right, high-conf — should be HIGH urgency, right"),
        (90, 0.1, 0.70, 0.7, "Medium dist, center, medium-conf — MEDIUM urgency"),
        (200, 0.0, 0.60, 0.6, "Far, center — LOW urgency"),
        (50, 0.0, 0.85, 0.2, "Near, high-conf but UNSTABLE — should suppress"),
        (40, 0.0, 0.20, 0.8, "Near, stable but LOW confidence — should suppress"),
        (100, 0.3, 0.75, 0.9, "Medium, slightly right, good conf+stability"),
    ]

    for dist, angle, conf, stab, desc in test_cases:
        result = fis.compute(dist, angle, conf, stab)
        dir_phrase = generate_direction_phrase(angle)
        print(f"  {desc}")
        print(f"    Inputs:  dist={dist}cm, angle={angle:.1f}, conf={conf:.2f}, stab={stab:.2f}")
        print(f"    Outputs: urgency={result['urgency']:.2f}, freq={result['frequency']:.2f}, "
              f"dir={result['direction']}, dist_label={result['distance_label']}, "
              f"announce={result['should_announce']}")
        print(f"    Phrase:  \"{dir_phrase}\"")
        print()
