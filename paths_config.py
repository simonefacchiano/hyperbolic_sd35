from pathlib import Path

# Project-local root.
PROJECT_ROOT = Path(__file__).resolve().parent

# Edit this root when moving the project to another machine.
FLICKR30K_ROOT = Path("/leonardo_scratch/fast/IscrC_MU4M/datasets/flickr30k")

FLICKR30K_CAPTIONS = FLICKR30K_ROOT / "captions.txt"
FLICKR30K_IMAGES = FLICKR30K_ROOT / "Images"

# SD3.5 paths for generation/training defaults.
SD35_MODEL_PATH = Path("/leonardo_scratch/fast/EUHPC_D25_097/SD35_large")
ADAPTER_SD35_PATH = PROJECT_ROOT / "adapter_SD35_fulldim.pth"
