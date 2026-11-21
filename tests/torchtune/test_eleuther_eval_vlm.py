# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import runpy
from pathlib import Path

import pytest
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

pytest.importorskip("lm_eval")

from lm_eval.api.instance import Instance
from torchtune.models.llama3_2_vision._component_builders import (
    llama3_2_vision_decoder,
    llama3_2_vision_encoder,
)
from torchtune.modules.model_fusion import DeepFusionModel


_VLMEvalWrapper = runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "recipes" / "eleuther_eval.py")
)["_VLMEvalWrapper"]


class TinyTokenizer:
    eos_id = 0
    eot_id = 0
    vocabulary = ["<eos>", "P", "A", "B", "C", "D", "<image>", "<pad>"]

    def encode(self, text, **kwargs):
        return [
            self.vocabulary.index(char) if char in self.vocabulary else 1
            for char in text
        ]

    def decode(self, tokens, skip_special_tokens=True):
        return "".join(
            self.vocabulary[token]
            for token in tokens
            if not skip_special_tokens or token != self.eos_id
        )


class TinyTransform:
    tokenizer = TinyTokenizer()
    image_seq_len = 5  # Four image patches and the vision encoder's class token.
    max_num_tiles = 1
    stop_tokens = [0]

    def __call__(self, sample, inference=False):
        content = sample["messages"][0].content
        image = next(block["content"] for block in content if block["type"] == "image")
        text = "".join(block["content"] for block in content if block["type"] == "text")
        tokens = [6] + self.tokenizer.encode(text)
        return {
            "tokens": tokens,
            "encoder_input": {
                "images": [pil_to_tensor(image).float().unsqueeze(0) / 255],
                "aspect_ratio": [torch.tensor([1, 1])],
            },
            "encoder_mask": [
                torch.ones(len(tokens), self.image_seq_len, dtype=torch.bool)
            ],
        }


@pytest.fixture
def make_wrapper():
    def build(answer="AB", max_seq_length=16):
        encoder = llama3_2_vision_encoder(
            patch_size=2,
            num_heads=2,
            clip_embed_dim=8,
            clip_num_layers=1,
            clip_hidden_states=None,
            num_layers_projection=1,
            decoder_embed_dim=8,
            tile_size=4,
            max_num_tiles=1,
        )
        decoder = llama3_2_vision_decoder(
            vocab_size=8,
            num_layers=1,
            fusion_interval=1,
            num_special_tokens=0,
            num_heads=2,
            num_kv_heads=2,
            embed_dim=8,
            intermediate_dim=16,
            max_seq_len=max_seq_length,
            encoder_max_seq_len=5,
        )
        # Make the real decoder deterministic: residual layers preserve the token
        # embedding, and its output projection maps P -> answer -> EOS.
        with torch.no_grad():
            for parameter in decoder.layers.parameters():
                parameter.zero_()
            decoder.tok_embeddings.embedding.weight.copy_(torch.eye(8))
            decoder.output.weight.zero_()
            sequence = [1] + TinyTokenizer().encode(answer) + [0]
            for previous, following in zip(sequence, sequence[1:]):
                decoder.output.weight[following, previous] = 1
        model = DeepFusionModel(decoder=decoder, encoder=encoder).eval()
        return _VLMEvalWrapper(
            model,
            TinyTransform(),
            device=torch.device("cpu"),
            max_seq_length=max_seq_length,
            batch_size=1,
            max_images_per_sample=1,
            dtype=torch.float32,
        )

    return build


@pytest.mark.parametrize(
    "prompt,answer,budget,max_seq_length,expected",
    [
        ("PPPP", "AB", 4, 16, "AB"),  # Prompt longer than answer; normal EOS.
        ("P", "ABCD", 6, 16, "ABCD"),  # Answer longer than prompt.
        ("PPPP", "", 4, 16, ""),  # EOS during prefill.
        ("PP", "ABCD", 1, 16, "A"),
        ("PP", "ABCD", 2, 16, "AB"),
        ("PPP", "ABCD", 4, 8, "ABCD"),  # Total length reaches cache capacity.
        (
            "PPP",
            "ABCD",
            2,
            6,
            "AB",
        ),  # Stop before the next cache index is out of bounds.
    ],
)
def test_generate_until(make_wrapper, prompt, answer, budget, max_seq_length, expected):
    wrapper = make_wrapper(answer, max_seq_length)
    request = Instance(
        request_type="generate_until",
        doc={},
        arguments=(
            "<image>" + prompt,
            {"max_gen_toks": budget, "until": []},
            {"visual": [Image.new("RGB", (4, 4), color="red")]},
        ),
        idx=0,
    )

    # Exercise the inherited harness consumer as well as the real wrapper,
    # image collation, model, sampling and temporary attention caches.
    calls = []
    handle = wrapper.model.register_forward_pre_hook(
        lambda module, args: calls.append(args[0].shape[-1])
    )
    try:
        assert wrapper.generate_until([request]) == [expected]
    finally:
        handle.remove()
    assert len(calls) == min(budget, len(answer) + 1)
    assert calls[0] == len(prompt) + 1
    assert not wrapper.model.caches_are_setup()


@pytest.mark.parametrize("budget", [1, 2, 4])
def test_model_generate_returns_prompt_and_continuation(make_wrapper, budget):
    wrapper = make_wrapper("ABCD", max_seq_length=8)
    batch = wrapper.tok_batch_multimodal_encode(
        ["<image>PPP"], [[Image.new("RGB", (4, 4), color="red")]]
    )
    prompt = batch["input_ids"].clone()

    output = wrapper._model_multimodal_generate(
        batch, max_length=prompt.shape[1] + budget, stop=[]
    )

    assert output.tolist() == [prompt[0].tolist() + [2, 3, 4, 5][:budget]]
    assert output.dtype == prompt.dtype
    assert output.device == prompt.device
    assert not wrapper.model.caches_are_setup()
