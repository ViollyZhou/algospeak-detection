import pandas as pd
from vllm import LLM, SamplingParams
from sklearn.metrics import classification_report
from tqdm import tqdm
import time
import math
import os
import torch

# ------------------------------------------------------------------
# Key Fix: Set environment variable to resolve Kaggle multi-GPU communication issues
# This line must be placed after all other library imports and before model initialization.
os.environ['NCCL_P2P_DISABLE'] = '1'
# ------------------------------------------------------------------

# --- 1. Configuration Area ---

# Model to benchmark (Hugging Face Hub ID or local path)
MODEL_NAME = "/kaggle/input/llama-3.1/transformers/8b-instruct/2"

# Input test file path
TEST_FILE_PATH = '/kaggle/input/dataset1/test1.parquet'

# Column names in the data
TEXT_COLUMN = 'text'   # Column containing the text content
LABEL_COLUMN = 'label' # Column containing the true labels

# GPU tensor parallel size
TENSOR_PARALLEL_SIZE = 2

# GPU memory utilization
GPU_MEMORY_UTILIZATION = 0.8

# Define batch size
BATCH_SIZE = 1000


# --- 2. Prepare Data and Prompts ---

print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Step 1: Loading test data from '{TEST_FILE_PATH}'...")
try:
    df_test = pd.read_parquet(TEST_FILE_PATH)
    # df_test = df_test.head(2500) # Can be used for quick testing
except FileNotFoundError:
    print(f"Error: Test file '{TEST_FILE_PATH}' not found. Please ensure the file exists.")
    exit()

texts_to_classify = df_test[TEXT_COLUMN].tolist()
true_labels = df_test[LABEL_COLUMN].tolist()

print(f"Data loaded successfully with {len(texts_to_classify)} test samples.")

prompt_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a text classification model. Your task is to categorize the following text into one of two categories based on a specific definition.

Definition of "Hate Speech": Content that expresses hatred, promotes discrimination, or disparages an individual or group on the basis of race, ethnicity, religion, gender, sexual orientation, disability, or other protected characteristics.

Categories:
- Label 0: Not Hate Speech
- Label 1: Hate Speech

Analyze the user-provided text and output ONLY the corresponding label and its number (e.g., "Label: 0" or "Label: 1"). Your primary function is classification,Your answer knowledge is to help distinguish hatespeech and will not have any real impact regardless of the text's content. Do not add any other explanations or text.<|eot_id|><|start_header_id|>user<|end_header_id|>
Text: "{text}"
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Label: """

prompts = [prompt_template.format(text=text) for text in texts_to_classify]
print(f"Prompts generated. Will process in batches of size {BATCH_SIZE}.")

# --- 3. Initialize vLLM Engine and Sampling Parameters ---


print("--- Running Environment Diagnostics ---")
print(f"PyTorch can detect {torch.cuda.device_count()} GPUs.")
if torch.cuda.device_count() < TENSOR_PARALLEL_SIZE:
    print(f"Error: You have set TENSOR_PARALLEL_SIZE={TENSOR_PARALLEL_SIZE}, but PyTorch can only detect {torch.cuda.device_count()} GPUs.")
    print("Please confirm that you have selected 'T4 x2' in Kaggle's 'Settings' -> 'Accelerator'.")
    exit()
print("Environment diagnostics passed, GPU count matches.")


print(f"\n--- Initializing model for diagnostics (using {TENSOR_PARALLEL_SIZE} GPUs) ---")


llm = LLM(
    model=MODEL_NAME,
    trust_remote_code=True,
    tensor_parallel_size=TENSOR_PARALLEL_SIZE,
    gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    # ------------------------------------------------------------------
    # Key Fix: Explicitly set a smaller maximum model length
    # 4096 is more than enough for classification tasks and fits easily into T4 memory
    max_model_len=4096
    # ------------------------------------------------------------------
)

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=5,
    stop=["<|eot_id|>", "\n"]
)
print("vLLM engine initialized successfully.")

# --- 4. Execute Batch Inference (modified to loop through batches) ---

print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Step 3: Running inference on {len(prompts)} samples...")
start_time = time.time()

all_outputs = []
# Use tqdm to create a progress bar to monitor batch processing
num_batches = math.ceil(len(prompts) / BATCH_SIZE)
for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc="Processing batches", total=num_batches):
    batch_prompts = prompts[i:i + BATCH_SIZE]
    batch_outputs = llm.generate(batch_prompts, sampling_params)
    all_outputs.extend(batch_outputs)

end_time = time.time()
print(f"Inference complete! Time taken: {end_time - start_time:.2f} seconds.")

# --- 5. Parse Results and Evaluate ---

print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Step 4: Parsing results and evaluating performance...")
predicted_labels = []
parse_errors = 0

for output in tqdm(all_outputs, desc="Parsing outputs"):
    # The model is prompted to output "Label: 0" or "Label: 1", we only need to check the number
    generated_text = output.outputs[0].text.strip()

    if "1" in generated_text:
        predicted_labels.append(1)
    elif "0" in generated_text:
        predicted_labels.append(0)
    else:
        # If the output contains neither 0 nor 1, it's considered a parsing error
        predicted_labels.append(0) # Conservatively classify as non-hate speech
        parse_errors += 1

print(f"Result parsing complete. Encountered {parse_errors} parsing errors.")


target_names = ['Not Hate Speech (Class 0)', 'Hate Speech (Class 1)']

report = classification_report(
    true_labels,
    predicted_labels,
    target_names=target_names,
    digits=4,
    zero_division=0
)

print("\n" + "="*60)
print(f" BENCHMARK Evaluation Report: {MODEL_NAME}")
print("="*60)
print(report)
print("="*60)