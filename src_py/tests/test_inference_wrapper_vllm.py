import unittest

from src_py.wrappers.inference_wrapper import (
    _translate_sglang_generate_to_vllm,
    _translate_vllm_completion_to_sglang,
)


class TestInferenceWrapperVllm(unittest.TestCase):
    def test_translate_sglang_generate_to_vllm(self) -> None:
        payload = {
            "input_ids": [10, 20, 30],
            "sampling_params": {
                "temperature": 0.7,
                "max_new_tokens": 5,
                "no_stop_trim": True,
                "stop": ["```\n"],
                "sampling_seed": 123,
            },
            "return_logprob": True,
            "top_logprobs_num": 8,
            "stream": False,
        }

        translated = _translate_sglang_generate_to_vllm(payload, "served-model")

        self.assertEqual("served-model", translated["model"])
        self.assertEqual([10, 20, 30], translated["prompt"])
        self.assertEqual(5, translated["max_tokens"])
        self.assertEqual(0.7, translated["temperature"])
        self.assertEqual(8, translated["logprobs"])
        self.assertEqual(123, translated["seed"])
        self.assertEqual(["```\n"], translated["stop"])
        self.assertTrue(translated["include_stop_str_in_output"])
        self.assertTrue(translated["return_token_ids"])
        self.assertTrue(translated["return_tokens_as_token_ids"])

    def test_translate_vllm_completion_to_sglang(self) -> None:
        response = {
            "choices": [
                {
                    "text": " answer",
                    "token_ids": [220, 17],
                    "prompt_token_ids": [10, 20, 30],
                    "logprobs": {
                        "token_logprobs": [-0.1, -0.2],
                        "top_logprobs": [
                            {"token_id:220": -0.1, "token_id:221": -2.0},
                            {"token_id:17": -0.2, "token_id:18": -3.0},
                        ],
                    },
                }
            ]
        }

        translated = _translate_vllm_completion_to_sglang(response)

        self.assertEqual(" answer", translated["text"])
        self.assertEqual([220, 17], translated["output_ids"])
        meta_info = translated["meta_info"]
        self.assertEqual(
            [[-0.1, 220, None], [-0.2, 17, None]],
            meta_info["output_token_logprobs"],
        )
        self.assertEqual(
            [
                [[-0.1, 220, None], [-2.0, 221, None]],
                [[-0.2, 17, None], [-3.0, 18, None]],
            ],
            meta_info["output_top_logprobs"],
        )


if __name__ == "__main__":
    unittest.main()
