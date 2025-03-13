import json
from datasets import load_dataset


def make_iwslt14_local(split: str, debug: bool = False, debug_size: int = 1000):
    """
    Saves the IWSLT14 dataset as a JSON file.

    Args:
        split (str): The dataset split to save ("train", "validation", or "test").
        debug (bool): If True, saves only a small subset (e.g., 100 examples) for debugging.
        debug_size (int): Number of samples to keep in debug mode.
    """
    dataset = load_dataset("ahazeemi/iwslt14-en-fr")[split]
    # debug mode is enabled
    if debug:
        dataset = dataset.select(range(100))  # Select only 100 samples for debugging

    # Save dataset under the correct split
    full_dataset = {
        split: {
            "en": dataset["en"],
            "fr": dataset["fr"]
        }
    }

    filename = f"iwslt14_{split}_debug.json" if debug else f"iwslt14_{split}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(full_dataset, f, ensure_ascii=False, indent=4)

    print(f"{split} dataset saved as {filename} ({'debug' if debug else 'full'})")


# Generate full and debug datasets for train, validation, and test splits
for split in ["train", "validation", "test"]:
    make_iwslt14_local(split=split, debug=False)  # Full dataset
    make_iwslt14_local(split=split, debug=True)  # Debug dataset
