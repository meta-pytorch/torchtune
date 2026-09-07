# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch


@pytest.fixture(scope="module")
def eleuther_eval_module():
    """Load eleuther_eval recipe module with mocked lm_eval dependencies."""
    mock_lm_eval = MagicMock()

    class MockHFLM:
        def __init__(self, pretrained=None, device=None):
            pass

    class MockHFMultimodalLM:
        pass

    mock_lm_eval.models.huggingface.HFLM = MockHFLM
    mock_lm_eval.models.hf_vlms.HFMultimodalLM = MockHFMultimodalLM

    saved_modules = {}
    mock_keys = [
        "lm_eval",
        "lm_eval.evaluator",
        "lm_eval.models",
        "lm_eval.models.hf_vlms",
        "lm_eval.models.huggingface",
        "lm_eval.tasks",
        "lm_eval.utils",
    ]
    for k in mock_keys:
        saved_modules[k] = sys.modules.get(k)
        if k == "lm_eval":
            sys.modules[k] = mock_lm_eval
        elif k == "lm_eval.models.huggingface":
            sys.modules[k] = mock_lm_eval.models.huggingface
        elif k == "lm_eval.models.hf_vlms":
            sys.modules[k] = mock_lm_eval.models.hf_vlms
        else:
            sys.modules[k] = getattr(mock_lm_eval, k.split(".")[-1], MagicMock())

    recipe_path = (
        Path(__file__).parent.parent.parent / "recipes" / "eleuther_eval.py"
    ).resolve()
    spec = importlib.util.spec_from_file_location("eleuther_eval", recipe_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    yield module

    for k, v in saved_modules.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v


class TestEleutherEvalGenerationBudget:
    """Unit tests for configurable generation budget and harness override resolution."""

    @pytest.fixture
    def mock_model(self):
        model = MagicMock()
        model.dtype = torch.float32
        return model

    @pytest.fixture
    def mock_tokenizer(self):
        tokenizer = MagicMock()
        tokenizer.eos_id = 0
        tokenizer.pad_id = 0
        tokenizer.stop_tokens = [0]
        return tokenizer

    def test_default_max_gen_toks(
        self, eleuther_eval_module, mock_model, mock_tokenizer
    ):
        wrapper = eleuther_eval_module._LLMEvalWrapper(
            mock_model,
            mock_tokenizer,
            device="cpu",
            max_seq_length=2048,
            batch_size=2,
        )
        assert wrapper.max_gen_toks == 256

    def test_custom_max_gen_toks(
        self, eleuther_eval_module, mock_model, mock_tokenizer
    ):
        wrapper = eleuther_eval_module._LLMEvalWrapper(
            mock_model,
            mock_tokenizer,
            device="cpu",
            max_seq_length=2048,
            batch_size=2,
            max_gen_toks=512,
        )
        assert wrapper.max_gen_toks == 512

    def test_model_generate_uses_default_when_no_harness_kwargs(
        self, eleuther_eval_module, mock_model, mock_tokenizer
    ):
        wrapper = eleuther_eval_module._LLMEvalWrapper(
            mock_model,
            mock_tokenizer,
            device="cpu",
            max_seq_length=2048,
            batch_size=2,
            max_gen_toks=512,
        )
        context = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)

        with (
            patch.object(eleuther_eval_module, "local_kv_cache"),
            patch.object(eleuther_eval_module, "generate") as mock_generate,
        ):
            mock_generate.return_value = (torch.zeros((2, 10)), None)
            wrapper._model_generate(context)
            assert mock_generate.call_args.kwargs["max_generated_tokens"] == 512

    def test_model_generate_overridden_by_max_gen_toks(
        self, eleuther_eval_module, mock_model, mock_tokenizer
    ):
        wrapper = eleuther_eval_module._LLMEvalWrapper(
            mock_model,
            mock_tokenizer,
            device="cpu",
            max_seq_length=2048,
            batch_size=2,
            max_gen_toks=256,
        )
        context = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)

        with (
            patch.object(eleuther_eval_module, "local_kv_cache"),
            patch.object(eleuther_eval_module, "generate") as mock_generate,
        ):
            mock_generate.return_value = (torch.zeros((2, 10)), None)
            wrapper._model_generate(context, max_gen_toks=64)
            assert mock_generate.call_args.kwargs["max_generated_tokens"] == 64

    def test_model_generate_overridden_by_max_new_tokens(
        self, eleuther_eval_module, mock_model, mock_tokenizer
    ):
        wrapper = eleuther_eval_module._LLMEvalWrapper(
            mock_model,
            mock_tokenizer,
            device="cpu",
            max_seq_length=2048,
            batch_size=2,
            max_gen_toks=256,
        )
        context = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)

        with (
            patch.object(eleuther_eval_module, "local_kv_cache"),
            patch.object(eleuther_eval_module, "generate") as mock_generate,
        ):
            mock_generate.return_value = (torch.zeros((2, 10)), None)
            wrapper._model_generate(context, max_new_tokens=128)
            assert mock_generate.call_args.kwargs["max_generated_tokens"] == 128

    def test_model_generate_precedence_max_gen_toks_over_max_new_tokens(
        self, eleuther_eval_module, mock_model, mock_tokenizer
    ):
        wrapper = eleuther_eval_module._LLMEvalWrapper(
            mock_model,
            mock_tokenizer,
            device="cpu",
            max_seq_length=2048,
            batch_size=2,
            max_gen_toks=256,
        )
        context = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)

        with (
            patch.object(eleuther_eval_module, "local_kv_cache"),
            patch.object(eleuther_eval_module, "generate") as mock_generate,
        ):
            mock_generate.return_value = (torch.zeros((2, 10)), None)
            wrapper._model_generate(context, max_gen_toks=32, max_new_tokens=64)
            assert mock_generate.call_args.kwargs["max_generated_tokens"] == 32
