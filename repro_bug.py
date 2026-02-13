import sys
import os
import importlib.util
from unittest.mock import MagicMock, patch
from omegaconf import OmegaConf

# --- THE FIX: Load the recipe file directly to bypass the "recipes is not a package" error ---
recipe_path = os.path.join("recipes", "full_finetune_distributed.py")
spec = importlib.util.spec_from_file_location("full_finetune_distributed", recipe_path)
recipe_module = importlib.util.module_from_spec(spec)
sys.modules["full_finetune_distributed"] = recipe_module
spec.loader.exec_module(recipe_module)
FullFinetuneRecipeDistributed = recipe_module.FullFinetuneRecipeDistributed
# ---------------------------------------------------------------------------------------

# Mock distributed environment to look like "GPU #1" (Rank 1)
with patch("torch.distributed.is_available", return_value=True), \
     patch("torch.distributed.is_initialized", return_value=True), \
     patch("torch.distributed.get_rank", return_value=1), \
     patch("torch.distributed.get_world_size", return_value=2):

    # Create a dummy config
    cfg = OmegaConf.create({
        "output_dir": "/tmp/test",
        "metric_logger": {"_component_": "torchtune.training.metric_logging.DiskLogger", "log_dir": "/tmp"}
    })

    print("\n\n>>> STARTING SIMULATION OF RANK 1 (Should remain SILENT) <<<")
    try:
        # Initialize recipe. This calls setup(), which currently has the BUG.
        recipe = FullFinetuneRecipeDistributed(cfg)
        recipe.setup(cfg)
    except Exception as e:
        # We expect it to crash eventually because we don't have real data,
        # but we ONLY care if it prints the config before crashing.
        pass
    print(">>> END SIMULATION <<<\n")
    