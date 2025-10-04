import json
import os
import random


def split_dataset(input_filename, output_dir='.'):
    """
    Reads a JSON file, shuffles it randomly, splits it into training, validation,
    and test sets with an 8:1:1 ratio, and generates a dataset_info.json file.
    """
    # --- 1. Read the merged data file ---
    if not os.path.exists(input_filename):
        print(f"Error: Input file '{input_filename}' not found.")
        return

    try:
        with open(input_filename, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
        if not isinstance(all_data, list):
            print(f"Error: The content of file '{input_filename}' is not a list.")
            return
        print(f"Successfully read data, total samples: {len(all_data)}")
    except Exception as e:
        print(f"Error reading or parsing file '{input_filename}': {e}")
        return

    # --- 2. Shuffle the data randomly ---
    # Use a fixed random seed to ensure the shuffling is deterministic for reproducibility.
    random.seed(42)
    random.shuffle(all_data)
    print("Data has been shuffled randomly.")

    # --- 3. Calculate split points ---
    total_size = len(all_data)
    train_end_index = int(total_size * 0.8)
    validation_end_index = train_end_index + int(total_size * 0.1)

    # --- 4. Split the dataset ---
    train_data = all_data[:train_end_index]
    validation_data = all_data[train_end_index:validation_end_index]
    test_data = all_data[validation_end_index:]

    print("\nDataset splitting complete:")
    print(f"  - Training set (train.json):   {len(train_data)} samples")
    print(f"  - Validation set (validation.json): {len(validation_data)} samples")
    print(f"  - Test set (test.json):    {len(test_data)} samples")

    # --- 5. Save the three dataset files ---
    output_files = {
        "train.json": train_data,
        "validation.json": validation_data,
        "test.json": test_data
    }

    print("\nSaving files...")
    for filename, data in output_files.items():
        filepath = os.path.join(output_dir, filename)
        try:
            # Use ensure_ascii=False to correctly handle non-ASCII characters
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"  - Saved '{filename}'")
        except Exception as e:
            print(f"Error saving file '{filename}': {e}")

    # --- 6. Create and save the dataset_info.json file ---
    dataset_info = {
        "hate_speech_detection": {
            "splits": {
                "train": "train.json",
                "validation": "validation.json",
                "test": "test.json"
            },
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output"
            },
            "dataset_size": total_size,
            "split_sizes": {
                "train": len(train_data),
                "validation": len(validation_data),
                "test": len(test_data)
            }
        }
    }

    info_filepath = os.path.join(output_dir, "dataset_info.json")
    try:
        with open(info_filepath, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=4)
        print(f"  - Saved 'dataset_info.json'")
    except Exception as e:
        print(f"Error saving file 'dataset_info.json': {e}")

    print("\nAll tasks completed!")


# --- Main execution block ---
if __name__ == '__main__':
    # --- Configure your filenames here ---

    # Input JSON file (already merged)
    merged_file = 'D:/Pycharm/jargon_detcection/train/algodata/algo_data.json'  # <--- Change this to the merged filename you generated in the previous step

    # Execute the split function
    split_dataset(merged_file)
