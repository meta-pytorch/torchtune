# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from omegaconf import OmegaConf

from torchtune.training.checkpointing._checkpoint_client import CheckpointClient


def test_distributed_checkpointer_cannot_be_configured_as_base_checkpointer(
    tmp_path,
):
    cfg = OmegaConf.create(
        {
            "device": "cpu",
            "checkpointer": {
                "_component_": "torchtune.training.DistributedCheckpointer",
                "checkpoint_dir": str(tmp_path / "checkpoint"),
                "output_dir": str(tmp_path / "output"),
                "model_type": "LLAMA2",
            },
        }
    )

    with pytest.raises(
        ValueError,
        match="DistributedCheckpointer.*base checkpointer",
    ):
        CheckpointClient(cfg)
