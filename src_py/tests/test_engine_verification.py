import os
import unittest
from unittest import mock

from src_py.train.batch_dataset import ResolvedTrainingBatch
from src_py.train.data_msgpack import QuestionNodeId, TrainingSampleTokenized
from src_py.train.engine import (
    _resolve_attention_backend_from_env,
    _resolve_fsdp_transformer_layer_classes,
    _resolve_pad_token_id,
    _shard_batches_for_rank,
    _verify_tokenizer_model_match,
)


def _build_sample(
    question_id: int,
    node_id: int,
    input_ids: list[int],
    labels: list[int],
    model_official_name: str,
) -> TrainingSampleTokenized:
    return TrainingSampleTokenized(
        id=QuestionNodeId(question_id=question_id, node_id=node_id),
        input_ids=input_ids,
        labels=labels,
        input_length=len(input_ids),
        token_advantages=[0.0] + [0.1] * (len(input_ids) - 1),
        model_official_name=model_official_name,
    )


class TestEngineTokenizerVerification(unittest.TestCase):
    def test_resolve_attention_backend_defaults_to_kernels_flash_attn2(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TRAIN_ATTENTION_BACKEND", None)
            self.assertEqual(
                "kernels-community/flash-attn2",
                _resolve_attention_backend_from_env(),
            )

    def test_resolve_attention_backend_honors_sdpa(self) -> None:
        with mock.patch.dict(
            os.environ, {"TRAIN_ATTENTION_BACKEND": "sdpa"}, clear=False
        ):
            self.assertEqual("sdpa", _resolve_attention_backend_from_env())

    def test_resolve_attention_backend_honors_eager(self) -> None:
        with mock.patch.dict(
            os.environ, {"TRAIN_ATTENTION_BACKEND": "eager"}, clear=False
        ):
            self.assertEqual("eager", _resolve_attention_backend_from_env())

    def test_resolve_attention_backend_honors_kernels_flash_attn2(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"TRAIN_ATTENTION_BACKEND": "kernels-community/flash-attn2"},
            clear=False,
        ):
            self.assertEqual(
                "kernels-community/flash-attn2",
                _resolve_attention_backend_from_env(),
            )

    def test_resolve_attention_backend_rejects_unknown_value(self) -> None:
        with mock.patch.dict(
            os.environ, {"TRAIN_ATTENTION_BACKEND": "flash"}, clear=False
        ):
            with self.assertRaises(AssertionError):
                _resolve_attention_backend_from_env()

    def test_resolve_pad_token_id_prefers_defined_pad(self) -> None:
        self.assertEqual(
            7, _resolve_pad_token_id(tokenizer_pad_token_id=7, tokenizer_eos_token_id=2)
        )

    def test_resolve_pad_token_id_falls_back_to_eos(self) -> None:
        self.assertEqual(
            2,
            _resolve_pad_token_id(
                tokenizer_pad_token_id=None, tokenizer_eos_token_id=2
            ),
        )

    def test_resolve_pad_token_id_raises_when_pad_and_eos_missing(self) -> None:
        with self.assertRaises(AssertionError):
            _resolve_pad_token_id(
                tokenizer_pad_token_id=None, tokenizer_eos_token_id=None
            )

    def test_resolve_fsdp_transformer_layer_classes_detects_decoder_blocks(
        self,
    ) -> None:
        import torch

        class Qwen2DecoderLayer(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.proj = torch.nn.Linear(4, 4)

        class OtherModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.proj = torch.nn.Linear(4, 4)

        model = torch.nn.ModuleList(
            [Qwen2DecoderLayer(), Qwen2DecoderLayer(), OtherModule()]
        )

        classes = _resolve_fsdp_transformer_layer_classes(model)
        self.assertEqual((Qwen2DecoderLayer,), classes)

    def test_shard_batches_for_rank_disjoint_and_complete(self) -> None:
        model_name = "Qwen/Qwen2.5-7B-Instruct"
        batches: list[ResolvedTrainingBatch] = []
        for batch_index in range(7):
            sample = _build_sample(batch_index, 0, [1, 2], [-100, 2], model_name)
            batches.append(
                ResolvedTrainingBatch(
                    batch_index=batch_index,
                    ids=[sample.id],
                    samples=[sample],
                    model_official_name=model_name,
                )
            )

        rank0 = _shard_batches_for_rank(batches, rank=0, world_size=3)
        rank1 = _shard_batches_for_rank(batches, rank=1, world_size=3)
        rank2 = _shard_batches_for_rank(batches, rank=2, world_size=3)

        ids0 = {batch.batch_index for batch in rank0}
        ids1 = {batch.batch_index for batch in rank1}
        ids2 = {batch.batch_index for batch in rank2}
        all_ids = ids0 | ids1 | ids2

        self.assertEqual({0, 1, 2, 3, 4, 5, 6}, all_ids)
        self.assertEqual(set(), ids0 & ids1)
        self.assertEqual(set(), ids0 & ids2)
        self.assertEqual(set(), ids1 & ids2)

    def test_verify_tokenizer_model_match_success(self) -> None:
        model_path = "/tmp/models/model_qwen35_08b"
        sample_a = _build_sample(1, 1, [1, 2, 3], [-100, 2, 3], model_path)
        sample_b = _build_sample(1, 2, [4, 5], [-100, 5], model_path)
        ordered_batches = [
            ResolvedTrainingBatch(
                batch_index=0,
                ids=[sample_a.id, sample_b.id],
                samples=[sample_a, sample_b],
                model_official_name=model_path,
            )
        ]

        result = _verify_tokenizer_model_match(
            model_path=model_path,
            tokenizer_name_or_path=model_path,
            ordered_batches=ordered_batches,
            model_vocab_size=16,
        )

        self.assertEqual(model_path, result["model_official_name"])
        self.assertEqual(model_path, result["tokenizer_name_or_path"])
        self.assertEqual(16, result["model_vocab_size"])
        self.assertEqual(5, result["max_input_token_id"])
        self.assertEqual(5, result["max_label_token_id"])

    def test_verify_tokenizer_model_match_fails_on_data_model_mismatch(self) -> None:
        sample = _build_sample(
            1,
            1,
            [1, 2],
            [-100, 2],
            "Qwen/Qwen3-4B",
        )
        ordered_batches = [
            ResolvedTrainingBatch(
                batch_index=0,
                ids=[sample.id],
                samples=[sample],
                model_official_name="Qwen/Qwen3-4B",
            )
        ]

        with self.assertRaises(AssertionError):
            _verify_tokenizer_model_match(
                model_path="/tmp/models/model_qwen35_08b",
                tokenizer_name_or_path="/tmp/models/model_qwen35_08b",
                ordered_batches=ordered_batches,
                model_vocab_size=16,
            )

    def test_verify_tokenizer_model_match_fails_on_vocab_range(self) -> None:
        model_path = "/tmp/models/model_qwen35_08b"
        sample = _build_sample(1, 1, [1, 9], [-100, 9], model_path)
        ordered_batches = [
            ResolvedTrainingBatch(
                batch_index=0,
                ids=[sample.id],
                samples=[sample],
                model_official_name=model_path,
            )
        ]

        with self.assertRaises(AssertionError):
            _verify_tokenizer_model_match(
                model_path=model_path,
                tokenizer_name_or_path=model_path,
                ordered_batches=ordered_batches,
                model_vocab_size=9,
            )


if __name__ == "__main__":
    unittest.main()
