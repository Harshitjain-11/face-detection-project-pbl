import unittest
import numpy as np

from app import (
    classify_arcface_embedding,
    _normalize_vector,
)


class MatchingLogicTests(unittest.TestCase):
    def test_unknown_when_no_store(self):
        probe = _normalize_vector(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        label, score = classify_arcface_embedding(probe, {})
        self.assertEqual(label, "Unknown")
        self.assertAlmostEqual(score, 1.0)

    def test_known_when_clear_margin(self):
        alice = [
            _normalize_vector(np.array([1.0, 0.0, 0.0], dtype=np.float32)),
            _normalize_vector(np.array([0.98, 0.02, 0.0], dtype=np.float32)),
        ]
        bob = [
            _normalize_vector(np.array([0.0, 1.0, 0.0], dtype=np.float32)),
            _normalize_vector(np.array([0.0, 0.98, 0.02], dtype=np.float32)),
        ]
        probe = _normalize_vector(np.array([0.99, 0.01, 0.0], dtype=np.float32))

        label, score = classify_arcface_embedding(probe, {"Alice": alice, "Bob": bob})
        self.assertEqual(label, "Alice")
        self.assertLess(score, 0.1)

    def test_unknown_when_margin_too_small(self):
        # Probe is almost equally close to both identities.
        left = [_normalize_vector(np.array([1.0, 0.0, 0.0], dtype=np.float32))]
        right = [_normalize_vector(np.array([0.98, 0.02, 0.0], dtype=np.float32))]
        probe = _normalize_vector(np.array([0.99, 0.01, 0.0], dtype=np.float32))

        label, _ = classify_arcface_embedding(probe, {"Left": left, "Right": right})
        self.assertEqual(label, "Unknown")


if __name__ == "__main__":
    unittest.main()
