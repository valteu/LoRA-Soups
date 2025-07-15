from pathlib import Path

RESOURCES_PATH = Path("/sc/projects/sci-herbrich/chair/lora-bp/tunepare/")


def get_dataset_path() -> Path:
    return RESOURCES_PATH / "datasets"


def get_adapter_path() -> Path:
    return RESOURCES_PATH / "adapter_pool"


def get_merging_balanced_adapter_path() -> Path:
    return RESOURCES_PATH / "merging" / "balanced"


def get_merging_unbalanced_adapter_path() -> Path:
    return RESOURCES_PATH / "merging" / "unbalanced"


def get_merging_dataset_path() -> Path:
    return RESOURCES_PATH / "merging_datasets"
