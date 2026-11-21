# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from tests.test_utils import assert_expected
from torchtune.models.qwen3._component_builders import lora_qwen3
from torchtune.modules import TiedLinear
from torchtune.modules.peft import get_adapter_params, LoRATiedLinear
from torchtune.training.seed import set_seed

RANK = 4
ALPHA = 1.0
BSZ = 2
SEQ_LEN = 32
EMBED_DIM = 64
INTERMEDIATE_DIM = 168
NUM_HEADS = 4
NUM_KV_HEADS = 2
MAX_SEQ_LEN = 64
VOCAB_SIZE = 128
NUM_LAYERS = 2


@pytest.fixture(autouse=True)
def random():
    set_seed(16)


class TestLoRAQwen3Output:
    def _build(self, apply_lora_to_output: bool, use_dora: bool = False):
        return lora_qwen3(
            lora_attn_modules=["q_proj", "v_proj"],
            apply_lora_to_output=apply_lora_to_output,
            vocab_size=VOCAB_SIZE,
            num_layers=NUM_LAYERS,
            num_heads=NUM_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            embed_dim=EMBED_DIM,
            intermediate_dim=INTERMEDIATE_DIM,
            max_seq_len=MAX_SEQ_LEN,
            tie_word_embeddings=True,
            lora_rank=RANK,
            lora_alpha=ALPHA,
            use_dora=use_dora,
        )

    def test_tied_output_uses_lora_tied_linear(self):
        model = self._build(apply_lora_to_output=True)
        assert isinstance(model.output, LoRATiedLinear)
        assert model.output.weight is model.tok_embeddings.weight
        assert "output.lora_a.weight" in get_adapter_params(model)
        assert "output.lora_b.weight" in get_adapter_params(model)
        assert "output.weight" not in dict(model.named_parameters())

    def test_tied_output_disabled_is_plain_tied_linear(self):
        model = self._build(apply_lora_to_output=False)
        assert isinstance(model.output, TiedLinear)

    def test_tied_output_forward(self):
        torch.manual_seed(0)
        model = self._build(apply_lora_to_output=True)
        tokens = torch.randint(0, VOCAB_SIZE, (BSZ, SEQ_LEN))
        logits = model(tokens)
        assert_expected(logits.shape, (BSZ, SEQ_LEN, VOCAB_SIZE))

    def test_tied_output_dora_raises(self):
        with pytest.raises(ValueError, match="DoRA is not supported"):
            self._build(apply_lora_to_output=True, use_dora=True)
