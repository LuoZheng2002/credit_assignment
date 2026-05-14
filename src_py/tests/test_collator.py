import unittest

from src_py.train.collator import IGNORE_LABEL, collate_training_samples
from src_py.train.data_sqlite import QuestionNodeId, TrainingSampleTokenized


class TestCollator(unittest.TestCase):
    def test_collate_training_samples_padding_and_mask(self) -> None:
        samples = [
            TrainingSampleTokenized(
                id=QuestionNodeId(question_id=1, node_id=1),
                input_ids=[101, 102],
                labels=[-100, 102],
                reconstructed="a",
                input_length=2,
                advantage=1.2,
                model_official_name="Qwen/Qwen2.5-7B-Instruct",
            ),
            TrainingSampleTokenized(
                id=QuestionNodeId(question_id=1, node_id=2),
                input_ids=[201, 202, 203, 204],
                labels=[-100, 202, 203, 204],
                reconstructed="b",
                input_length=4,
                advantage=-0.5,
                model_official_name="Qwen/Qwen2.5-7B-Instruct",
            ),
        ]

        collated = collate_training_samples(samples=samples, pad_token_id=0)

        self.assertEqual((2, 4), tuple(collated.input_ids.shape))
        self.assertEqual((2, 4), tuple(collated.labels.shape))
        self.assertEqual((2, 4), tuple(collated.attention_mask.shape))
        self.assertEqual((2,), tuple(collated.advantages.shape))

        self.assertEqual([101, 102, 0, 0], collated.input_ids[0].tolist())
        self.assertEqual([-100, 102, IGNORE_LABEL, IGNORE_LABEL], collated.labels[0].tolist())
        self.assertEqual([1, 1, 0, 0], collated.attention_mask[0].tolist())

        self.assertEqual([201, 202, 203, 204], collated.input_ids[1].tolist())
        self.assertEqual([-100, 202, 203, 204], collated.labels[1].tolist())
        self.assertEqual([1, 1, 1, 1], collated.attention_mask[1].tolist())

    def test_collate_rejects_empty_samples(self) -> None:
        with self.assertRaises(AssertionError):
            collate_training_samples(samples=[], pad_token_id=0)


if __name__ == "__main__":
    unittest.main()
