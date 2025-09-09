import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
from torch.optim import AdamW, SGD
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# --- 1. Configurable Area: Adjust your hyperparameters here ---

config = {
    # Model settings
    "model_name": "meta-llama/Llama-3-8b-instruct",
    "use_4bit": True,  # Whether to use 4-bit quantization to save memory

    # Data settings (using a simple example here)
    "max_length": 128,

    # Training settings
    "epochs": 1,
    "batch_size": 2,
    "learning_rate": 5e-5,
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # ===> Key Customization Point 1: Choose the optimizer <===
    # Options: "AdamW", "SGD", or you can add more
    "optimizer": "AdamW",

    # ===> Key Customization Point 2: LoRA Configuration <===
    "lora": {
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        # Specify the module names to apply LoRA to, for Llama2 these are typically "q_proj" and "v_proj"
        "target_modules": ["q_proj", "v_proj"],
        "task_type": "CAUSAL_LM",
    },

    # Output directory
    "output_dir": "./my_lora_model",
}

# --- 2. Data Preparation ---

# Create a simple example dataset
class SimpleDataset(Dataset):
    def __init__(self, tokenizer, num_samples=100):
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        # Assume this is your data
        self.data = ["Human: What is PyTorch?\nAssistant: PyTorch is a popular deep learning framework."] * num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        text = self.data[idx]
        # Tokenize
        tokens = self.tokenizer(
            text,
            max_length=config["max_length"],
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        # Squeeze tensors to remove the batch dimension of 1
        return {
            "input_ids": tokens.input_ids.squeeze(0),
            "attention_mask": tokens.attention_mask.squeeze(0),
            "labels": tokens.input_ids.squeeze(0).clone() # For Causal LM, labels are the same as input_ids
        }

# --- 3. Model and Tokenizer Loading ---

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Configure 4-bit quantization (if needed)
quantization_config = None
if config["use_4bit"]:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

# Load Model
model = AutoModelForCausalLM.from_pretrained(
    config["model_name"],
    quantization_config=quantization_config,
    device_map={"": 0} # Automatically load the model to GPU 0
)

# ===> Key Customization Point 3: Modify model architecture before applying LoRA <===
# If you want to replace activation functions, do it here. This is an advanced operation and requires
# knowledge of the model's internal structure.
# For example (this is a pseudocode example, specific module names depend on the model architecture):
# for name, module in model.named_modules():
#     if "mlp.act_fn" in name: # Assuming this is the name of Llama's activation function module
#         # module -> SiLUActivation()
#         # new_activation = torch.nn.ReLU()
#         # setattr(parent_module, 'act_fn', new_activation)
#         print(f"Found activation function at {name}, you can replace it here.")
#         pass


# --- 4. Apply LoRA (using PEFT) ---

print("\nApplying LoRA to the model...")
peft_config = LoraConfig(**config["lora"])
model = get_peft_model(model, peft_config)
model.print_trainable_parameters() # Print trainable parameters to verify that LoRA was applied successfully

# --- 5. Prepare for Training ---

# Create DataLoader
dataset = SimpleDataset(tokenizer)
data_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

# ===> Key Customization Point 4: Instantiate the chosen optimizer <===
optimizer = None
# PEFT has already handled this for you, it will only return the trainable parameters
trainable_params = model.parameters()

if config["optimizer"].lower() == 'adamw':
    optimizer = AdamW(trainable_params, lr=config["learning_rate"])
elif config["optimizer"].lower() == 'sgd':
    optimizer = SGD(trainable_params, lr=config["learning_rate"], momentum=0.9)
else:
    raise ValueError(f"Optimizer '{config['optimizer']}' is not supported.")

print(f"\nUsing optimizer: {config['optimizer']}")
model.to(config["device"])

# --- 6. Training Loop ---

print("Starting training...")
model.train() # Set the model to training mode

for epoch in range(config["epochs"]):
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{config['epochs']}")
    for batch in progress_bar:
        # Move data to the device
        batch = {k: v.to(config["device"]) for k, v in batch.items()}

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        # Update progress bar
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

print("\nTraining finished!")

# --- 7. Save Model ---

print(f"Saving LoRA adapter to {config['output_dir']}...")
model.save_pretrained(config["output_dir"])
tokenizer.save_pretrained(config["output_dir"])
print("Model saved successfully!")