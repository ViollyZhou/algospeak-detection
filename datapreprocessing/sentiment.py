# Install required libraries quietly
#!pip install transformers accelerate -q
#!pip install tqdm -q

import torch

# Check for GPU environment
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    print(f"Successfully detected {gpu_count} GPU(s)!")
    for i in range(gpu_count):
        print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("Error: No GPU detected. Please enable the T4 x2 accelerator in your environment settings.")

import pandas as pd
import re
from tqdm import tqdm

# Imports for Transformers and PyTorch
from transformers import (
    pipeline,
    AutoModelForSequenceClassification,
    AutoTokenizer
)


from torch.nn import DataParallel

# --- 1. Configuration ---

# Kaggle input file paths usually start with /kaggle/input/
# !!! Please replace 'your-dataset-folder-name' with the actual folder name of your dataset!!!
INPUT_DIR = "/kaggle/input/sentimentdata"
INPUT_FILENAME = "cleaned_dataset.parquet"  # This is your input filename

# The output file path in Kaggle is /kaggle/working/
OUTPUT_DIR = "/kaggle/working/"
OUTPUT_FILENAME = "final_dataset_multi_gpu.parquet"

INPUT_FILE = f"{INPUT_DIR}/{INPUT_FILENAME}"
OUTPUT_FILE = f"{OUTPUT_DIR}/{OUTPUT_FILENAME}"

TEXT_COLUMN = 'text'
LABEL_COLUMN = 'label'

# --- Sentiment Filtering Parameters ---
SENTIMENT_THRESHOLD = 0.5  # Absolute value threshold for sentiment score

# --- Model Selection (Chinese/English) ---
# Option 1: English Model
MODEL_PATH = "cardiffnlp/twitter-roberta-base-sentiment-latest"


class DataParallelWithConfig(DataParallel):
    """
    A custom DataParallel wrapper that forwards requests for missing attributes
    (e.g., '.config') to its underlying original model.
    """

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


# =============================================================================
#  Step 2: Update the core function to use this new wrapper
# =============================================================================
def sentiment_filter_multi_gpu(df: pd.DataFrame) -> pd.DataFrame:
    """
    Uses multi-GPU parallel processing to compute sentiment scores and filters data
    based on a threshold (final corrected version).
    """
    print("\n--- Starting Multi-GPU Sentiment Analysis Filtering ---")

    print(f"Loading model and tokenizer from '{MODEL_PATH}'...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

    model.to('cuda:0')

    print("Creating pipeline with the original model...")
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=0  # The pipeline itself is anchored to the primary device
    )

    # --- Core Fix: Replace standard 'DataParallel' with our custom 'smart' wrapper ---
    print("Wrapping the model inside the pipeline with our custom DataParallelWithConfig...")
    sentiment_pipeline.model = DataParallelWithConfig(sentiment_pipeline.model)

    print(f"Performing parallel sentiment analysis on {len(df)} records...")
    texts_to_analyze = df[TEXT_COLUMN].tolist()
    results = []

    # The batch_size here can be adjusted according to your VRAM (e.g., 32GB), for example, 256 or 512
    batch_size = 256

    for out in tqdm(sentiment_pipeline(texts_to_analyze, batch_size=batch_size, truncation=True),
                    total=len(texts_to_analyze),
                    desc="Parallel Sentiment Analysis Progress"):
        results.append(out)

    # Subsequent processing logic is identical
    scores = []
    for r in results:
        score = r['score']
        if r['label'] == 'negative':
            scores.append(-score)
        elif r['label'] == 'positive':
            scores.append(score)
        else:
            scores.append(0.0)

    df['sentiment_score'] = scores

    original_count = len(df)
    df_filtered = df[df['sentiment_score'].abs() >= SENTIMENT_THRESHOLD].copy()

    print(f"\nSentiment analysis complete.")
    print(f"Data count after filtering: {len(df_filtered)} rows (removed {original_count - len(df_filtered)} rows)")

    df_filtered = df_filtered.drop(columns=['sentiment_score'])

    return df_filtered


try:
    print(f"Reading file: {INPUT_FILE}")
    data_frame = pd.read_parquet(INPUT_FILE)
    print(f"Current data count: {len(data_frame)} rows")

    # Execute sentiment filtering
    final_df = sentiment_filter_multi_gpu(data_frame)

    # Save the final data
    final_df.to_parquet(OUTPUT_FILE, index=False)

    print("\n--- All cleaning steps are complete! ---")
    print(f"Final remaining data count: {len(final_df)} rows")
    print(f"Final file has been saved to: {OUTPUT_FILE}")
    print("You can find it in the 'Output' directory under the 'Data' panel on the right.")

except FileNotFoundError:
    print(f"Error: File not found at '{INPUT_FILE}'.")
    print("Please check if the 'INPUT_DIR' and 'INPUT_FILENAME' variables in the configuration section are correct.")
except Exception as e:
    print(f"An error occurred during processing: {e}")
