import unittest

from method_tvr.semantic_perturb.rewrite_sampler import sanitize_and_sample_rewrites


def _mk_item(text, relation_label, perturbation_type, semantic_relation="equivalent", confidence=0.95, severity=1):
    return {
        "text": text,
        "relation_label": relation_label,
        "perturbation_type": perturbation_type,
        "severity": severity,
        "short_rationale": "synthetic test sample",
        "verifier": {
            "verdict": "pass",
            "semantic_relation": semantic_relation,
            "confidence": confidence,
            "reason": "test",
        },
    }


class TestRewriteSampler(unittest.TestCase):
    def test_collision_sanitization_drops_negative_side(self):
        anchor = "A person closes a door in a room."
        shared = "In a room, a person closes a door."
        positives = [
            _mk_item(shared, "hard_positive", "syntax_reorder", semantic_relation="equivalent", confidence=1.0),
            _mk_item("A person shuts a door in a room.", "hard_positive", "lexical_variation", confidence=0.95),
        ]
        negatives = [
            _mk_item(shared, "hard_negative", "temporal_order_flip", semantic_relation="changed_core_semantics", confidence=0.95, severity=2),
            _mk_item("A person opens a door in a room.", "hard_negative", "action_swap", semantic_relation="contradiction", confidence=0.95, severity=3),
        ]
        pos, neg, neg_w, deb_w, stats = sanitize_and_sample_rewrites(
            anchor_text=anchor,
            positive_rewrites=positives,
            negative_rewrites=negatives,
            positive_sample_size=2,
            negative_sample_size=2,
            collision_sanitization_enabled=True,
            rewrite_type_quota_enabled=True,
            risky_negative_filter_enabled=False,
            risky_negative_overlap_threshold=0.9,
            risky_negative_start_epoch=0,
            risky_negative_downweight=0.5,
            current_epoch=0,
        )
        self.assertEqual(len(pos), 2)
        self.assertEqual(len(neg), 1)
        self.assertEqual(stats.collision_removed_negative, 1)
        self.assertEqual(len(neg_w), 1)
        self.assertEqual(len(deb_w), 1)

    def test_risky_temporal_negative_is_filtered(self):
        anchor = "A person enters the room, sits down, and drinks water."
        positives = [_mk_item("A person walks into the room, sits down, and drinks water.", "hard_positive", "lexical_variation")]
        negatives = [
            _mk_item(
                "A person drinks water, sits down, and enters the room.",
                "hard_negative",
                "temporal_order_flip",
                semantic_relation="changed_core_semantics",
                confidence=0.95,
                severity=2,
            ),
            _mk_item(
                "A person enters the room, sits down, and reads a book.",
                "hard_negative",
                "action_swap",
                semantic_relation="changed_core_semantics",
                confidence=0.95,
                severity=3,
            ),
        ]
        pos, neg, _, _, stats = sanitize_and_sample_rewrites(
            anchor_text=anchor,
            positive_rewrites=positives,
            negative_rewrites=negatives,
            positive_sample_size=1,
            negative_sample_size=2,
            collision_sanitization_enabled=True,
            rewrite_type_quota_enabled=True,
            risky_negative_filter_enabled=True,
            risky_negative_overlap_threshold=0.9,
            risky_negative_start_epoch=0,
            risky_negative_downweight=0.5,
            current_epoch=2,
        )
        self.assertEqual(len(pos), 1)
        self.assertEqual(len(neg), 1)
        self.assertEqual(neg[0]["perturbation_type"], "action_swap")
        self.assertGreaterEqual(stats.risky_negative_filtered, 1)

    def test_type_quota_prefers_non_dominant_negative(self):
        anchor = "A person opens a cabinet in the kitchen."
        positives = [
            _mk_item("A person opens a cabinet in the kitchen.", "hard_positive", "syntax_reorder", confidence=0.99),
            _mk_item("A person opens a cupboard in the kitchen.", "hard_positive", "paraphrase", confidence=0.95),
        ]
        negatives = [
            _mk_item("A person closes a cabinet in the kitchen.", "hard_negative", "action_swap", semantic_relation="contradiction", confidence=0.99, severity=3),
            _mk_item("A person opens a cabinet in the bathroom.", "hard_negative", "object_scene_swap", semantic_relation="changed_core_semantics", confidence=0.99, severity=3),
            _mk_item("A person opens a cabinet before entering the kitchen.", "hard_negative", "temporal_order_flip", semantic_relation="changed_core_semantics", confidence=0.98, severity=2),
        ]
        _, neg, _, _, stats = sanitize_and_sample_rewrites(
            anchor_text=anchor,
            positive_rewrites=positives,
            negative_rewrites=negatives,
            positive_sample_size=1,
            negative_sample_size=2,
            collision_sanitization_enabled=True,
            rewrite_type_quota_enabled=True,
            risky_negative_filter_enabled=False,
            risky_negative_overlap_threshold=0.95,
            risky_negative_start_epoch=0,
            risky_negative_downweight=0.5,
            current_epoch=0,
        )
        neg_types = {x["perturbation_type"] for x in neg}
        self.assertIn("temporal_order_flip", neg_types)
        self.assertGreaterEqual(stats.non_dominant_negative_selected, 1)


if __name__ == "__main__":
    unittest.main()
