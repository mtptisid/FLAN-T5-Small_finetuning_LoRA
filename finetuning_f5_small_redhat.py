# Install compatible dependencies
!pip uninstall -y torch torchvision torchaudio bitsandbytes triton transformers datasets peft gcsfs fsspec fastai huggingface_hub
!pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
!pip install bitsandbytes==0.43.3 triton==2.0.0 transformers==4.46.0 datasets==3.0.2 peft==0.13.2 huggingface_hub==0.26.2
!pip install gcsfs==2025.3.2 fsspec==2025.3.2

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset, utils
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig
import shutil
import os

# Clear dataset cache
cache_dir = utils.get_cache_dir()
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    print("Cache cleared.")

# Verify GPU and library setup
print("Torch Version:", torch.__version__)
print("BitsAndBytes Version:", bitsandbytes.__version__)
print("Triton Version:", triton.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

# 1. Set up quantization for memory efficiency
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False
)

# 2. Load tokenizer and model
model_name = "google/flan-t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="cuda",
    torch_dtype=torch.bfloat16
)

# 3. Apply LoRA for parameter-efficient fine-tuning
lora_config = LoraConfig(
    r=8,
    lora_alpha=20,
    target_modules=["q", "v"],
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)
model = get_peft_model(model, lora_config)

# 4. Load dataset from Hugging Face
try:
    dataset = load_dataset("mtpti5iD/redhat-docs", cache_dir="./new_cache")
    print("Dataset loaded successfully:", dataset)
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Attempting to load manually...")
    # Alternative: Load from JSONL file (uncomment if needed)
    # dataset = load_dataset('json', data_files='redhat-docs_dataset.jsonl')
    raise

# Handle null values and format data
def format_example(example):
    command = example['command'] if example['command'] is not None else ""
    title = example.get('title', 'Untitled')  # Use get() for safer access
    content = example.get('content', '')
    return {
        'text': f"Title: {title}\nContent: {content}\nCommand: {command}".strip()
    }

dataset = dataset.map(format_example)

# 5. Tokenize dataset
def tokenize_function(examples):
    inputs = tokenizer(
        examples['text'],
        max_length=512,
        truncation=True,
        padding="max_length"
    )
    inputs["labels"] = inputs["input_ids"].copy()
    return inputs

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(["text", "title", "content", "command", "url"])

# 6. Set up data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# 7. Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,  # Reduced for faster testing
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,  # Adjusted for stability
    fp16=True,
    optim="paged_adamw_8bit",
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
    report_to="none",
)

# 8. Initialize and train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator
)

trainer.train()

# 9. Save the fine-tuned model
model.save_pretrained("./finetuned_flan_t5_small")
tokenizer.save_pretrained("./finetuned_flan_t5_small")
