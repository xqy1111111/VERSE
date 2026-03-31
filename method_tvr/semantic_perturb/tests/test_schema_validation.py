import unittest

from method_tvr.semantic_perturb.schema import _validate_single_sentence_text


class TestSchemaValidation(unittest.TestCase):
    def test_repeated_punctuation_is_not_misclassified(self):
        text = 'A baby in pink shirt sits smiling as "Ellie Love Paris!!!" text appears.'
        _validate_single_sentence_text(text, "anchor_analysis")

    def test_quote_internal_punctuation_is_ignored(self):
        text = 'The sign says "Wait! Stop? Go. Now!" while the baby smiles.'
        _validate_single_sentence_text(text, "anchor_analysis")

    def test_many_sentence_terminator_groups_fail(self):
        text = "One. Two? Three! Four."
        with self.assertRaisesRegex(ValueError, "multiple sentences"):
            _validate_single_sentence_text(text, "anchor_analysis")


if __name__ == "__main__":
    unittest.main()
