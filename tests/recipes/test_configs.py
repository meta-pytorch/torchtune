# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
from pathlib import Path

import torchtune

from omegaconf import OmegaConf
from torchtune import config
from torchtune.utils import torch_version_ge

CONFIG_DIR = Path(torchtune.__file__).parent.parent / "recipes" / "configs"


class TestConfigs:
    def test_instantiate(self) -> None:
        all_configs = [
            os.path.join(CONFIG_DIR, f)
            for f in os.listdir(CONFIG_DIR)
            if f.endswith(".yaml")
        ]
        for config_path in all_configs:
            # QAT config is only compatible with PyTorch 2.4+
            if config_path.endswith("qat_full.yaml") and not torch_version_ge("2.4.0"):
                continue
            cfg = OmegaConf.load(config_path)
            config.validate(cfg)
