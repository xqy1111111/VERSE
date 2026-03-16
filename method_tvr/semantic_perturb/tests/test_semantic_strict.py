import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import h5py
import numpy as np
import torch

from method_tvr.semantic_perturb.dataset_semantic import load_semantic_cache_lookup
from method_tvr.semantic_perturb.hashing import sha256_file
from method_tvr.semantic_perturb.losses import compute_consistency_loss, compute_preference_loss
from method_tvr.start_end_dataset import StartEndDataset


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


class TestSemanticStrict(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = self.tmpdir.name

        self.train_jsonl = os.path.join(self.root, "train.jsonl")
        self.desc_h5 = os.path.join(self.root, "desc.h5")
        self.video_h5 = os.path.join(self.root, "video.h5")

        rows = [{"desc_id": 1, "desc": "a person opens a door", "vid_name": "v1", "duration": 10.0, "ts": [1.0, 3.0]}]
        _write_jsonl(self.train_jsonl, rows)

        with h5py.File(self.desc_h5, "w") as f:
            f.create_dataset("1", data=np.random.randn(8, 768).astype(np.float32))

        with h5py.File(self.video_h5, "w") as f:
            f.create_dataset("v1", data=np.random.randn(16, 2048).astype(np.float32))

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_baseline_dataset_equivalence_when_semantic_disabled(self):
        dataset = StartEndDataset(
            dset_name="charades_fig",
            data_path=self.train_jsonl,
            desc_bert_path_or_handler=self.desc_h5,
            max_desc_len=8,
            max_ctx_len=16,
            vid_feat_path_or_handler=self.video_h5,
            clip_length=1.0,
            ctx_mode="video_tef",
            normalize_vfeat=False,
            normalize_tfeat=False,
            semantic_cache_lookup=None,
        )
        sample = dataset[0]
        self.assertIn("meta", sample)
        self.assertIn("model_inputs", sample)
        self.assertNotIn("semantic", sample["meta"])

    def test_strict_missing_cache_fails_fast(self):
        opt = SimpleNamespace(
            semantic_enable=True,
            semantic_backend="llm",
            semantic_cache_path=os.path.join(self.root, "missing.jsonl"),
            semantic_fail_on_missing_cache=True,
            semantic_fail_on_invalid_cache=True,
            semantic_no_fallback=True,
            semantic_strict_mode=True,
            dset_name="charades_fig",
            semantic_cache_split="train",
            semantic_prompt_version="semantic_generator_v1",
            semantic_schema_version="semantic_schema_v1",
            semantic_generator_model="gpt-4.1-mini",
            semantic_verifier_model="gpt-4.1-mini",
            semantic_neg_types=["attribute_swap"],
            semantic_pos_types=["paraphrase"],
            semantic_severity_levels=[1, 2, 3],
            semantic_num_hard_neg=1,
            semantic_num_hard_pos=1,
        )
        with self.assertRaises(FileNotFoundError):
            load_semantic_cache_lookup(opt, source_data_path=self.train_jsonl)

    def test_invalid_cache_schema_fails(self):
        cache_path = os.path.join(self.root, "semantic_cache.jsonl")
        manifest_path = os.path.join(self.root, "semantic_cache.manifest.json")

        bad_entry = {
            "desc_id": 1,
            "anchor_text": "a person opens a door",
            "source_meta": {"vid_name": "v1", "ts": [1.0, 3.0], "duration": 10.0, "split": "train"},
            "hard_negatives": [],
            # hard_positives is missing on purpose
            "build_meta": {},
        }
        _write_jsonl(cache_path, [bad_entry])

        manifest = {
            "dataset": "charades_fig",
            "split": "train",
            "source_hash": sha256_file(self.train_jsonl),
            "prompt_version": "semantic_generator_v1",
            "schema_version": "semantic_schema_v1",
            "generator_model": "gpt-4.1-mini",
            "verifier_model": "gpt-4.1-mini",
            "backend": "llm",
            "neg_types": ["attribute_swap"],
            "pos_types": ["paraphrase"],
            "severity_levels": [1, 2, 3],
            "no_fallback": True,
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f)

        opt = SimpleNamespace(
            semantic_enable=True,
            semantic_backend="llm",
            semantic_cache_path=cache_path,
            semantic_fail_on_missing_cache=True,
            semantic_fail_on_invalid_cache=True,
            semantic_no_fallback=True,
            semantic_strict_mode=True,
            dset_name="charades_fig",
            semantic_cache_split="train",
            semantic_prompt_version="semantic_generator_v1",
            semantic_schema_version="semantic_schema_v1",
            semantic_generator_model="gpt-4.1-mini",
            semantic_verifier_model="gpt-4.1-mini",
            semantic_neg_types=["attribute_swap"],
            semantic_pos_types=["paraphrase"],
            semantic_severity_levels=[1, 2, 3],
            semantic_num_hard_neg=1,
            semantic_num_hard_pos=1,
        )
        with self.assertRaises(RuntimeError):
            load_semantic_cache_lookup(opt, source_data_path=self.train_jsonl)

    def test_manifest_version_mismatch_fails(self):
        cache_path = os.path.join(self.root, "semantic_cache_ok.jsonl")
        manifest_path = os.path.join(self.root, "semantic_cache_ok.manifest.json")

        ok_entry = {
            "desc_id": 1,
            "anchor_text": "a person opens a door",
            "source_meta": {"vid_name": "v1", "ts": [1.0, 3.0], "duration": 10.0, "split": "train"},
            "hard_negatives": [
                {
                    "text": "a person closes a door",
                    "relation_label": "hard_negative",
                    "perturbation_type": "attribute_swap",
                    "severity": 2,
                    "short_rationale": "core action is changed",
                    "verifier": {
                        "verdict": "pass",
                        "semantic_relation": "changed_core_semantics",
                        "confidence": 0.9,
                        "reason": "action differs",
                    },
                }
            ],
            "hard_positives": [
                {
                    "text": "someone opens the door",
                    "relation_label": "hard_positive",
                    "perturbation_type": "paraphrase",
                    "severity": 1,
                    "short_rationale": "same event with rephrase",
                    "verifier": {
                        "verdict": "pass",
                        "semantic_relation": "equivalent",
                        "confidence": 0.9,
                        "reason": "same meaning",
                    },
                }
            ],
            "build_meta": {},
        }
        _write_jsonl(cache_path, [ok_entry])

        manifest = {
            "dataset": "charades_fig",
            "split": "train",
            "source_hash": sha256_file(self.train_jsonl),
            "prompt_version": "wrong_version",
            "schema_version": "semantic_schema_v1",
            "generator_model": "gpt-4.1-mini",
            "verifier_model": "gpt-4.1-mini",
            "backend": "llm",
            "neg_types": ["attribute_swap"],
            "pos_types": ["paraphrase"],
            "severity_levels": [1, 2, 3],
            "no_fallback": True,
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f)

        opt = SimpleNamespace(
            semantic_enable=True,
            semantic_backend="llm",
            semantic_cache_path=cache_path,
            semantic_fail_on_missing_cache=True,
            semantic_fail_on_invalid_cache=True,
            semantic_no_fallback=True,
            semantic_strict_mode=True,
            dset_name="charades_fig",
            semantic_cache_split="train",
            semantic_prompt_version="semantic_generator_v1",
            semantic_schema_version="semantic_schema_v1",
            semantic_generator_model="gpt-4.1-mini",
            semantic_verifier_model="gpt-4.1-mini",
            semantic_neg_types=["attribute_swap"],
            semantic_pos_types=["paraphrase"],
            semantic_severity_levels=[1, 2, 3],
            semantic_num_hard_neg=1,
            semantic_num_hard_pos=1,
        )
        with self.assertRaises(RuntimeError):
            load_semantic_cache_lookup(opt, source_data_path=self.train_jsonl)

    def test_loss_functions_switch_paths(self):
        anchor = torch.tensor([0.9, 0.6], dtype=torch.float32)
        pos = [torch.tensor([0.85]), torch.tensor([])]
        neg = [torch.tensor([0.2, 0.3]), torch.tensor([0.4])]
        neg_w = [torch.tensor([1.0, 1.5]), torch.tensor([1.0])]

        pref = compute_preference_loss(anchor, pos, neg, neg_w, margin=0.2)
        cons = compute_consistency_loss(anchor, pos)
        self.assertGreaterEqual(float(pref.item()), 0.0)
        self.assertGreaterEqual(float(cons.item()), 0.0)

        zero_cons = compute_consistency_loss(torch.tensor([0.5]), [torch.tensor([])])
        self.assertEqual(float(zero_cons.item()), 0.0)


if __name__ == "__main__":
    unittest.main()
