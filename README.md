# FLAN-T5-Small Fine-Tuned on Red Hat Documentation

## Overview
This repository hosts a fine-tuned version of the **FLAN-T5-Small** model, optimized for question-answering tasks related to Red Hat documentation. The model was fine-tuned using Low-Rank Adaptation (LoRA) and 4-bit quantization to achieve efficient performance on limited computational resources, specifically a Google Colab T4 GPU with approximately 15 GB VRAM. The fine-tuning was performed on the `mtpti5iD/redhat-docs_dataset`, a dataset containing 55,741 rows of Red Hat documentation in JSONL format, with fields for `title`, `content`, `command`, and `url`. The model is designed to provide accurate and concise answers to queries about Red Hat software installation, configuration, and administration.

The project, including the fine-tuning script and documentation, is also available on GitHub: [mtptisid/FLAN-T5-Small_finetuning_LoRA](https://github.com/mtptisid/FLAN-T5-Small_finetuning_LoRA).

## Model Details
- **Base Model**: Google FLAN-T5-Small (`google/flan-t5-small`), a lightweight sequence-to-sequence transformer model with 77 million parameters.
- **Fine-Tuning Method**: Parameter-efficient fine-tuning using LoRA, targeting query (`q`) and value (`v`) projection layers.
- **Quantization**: 4-bit NormalFloat (nf4) quantization with bfloat16 compute dtype, reducing memory usage to approximately 6-8 GB.
- **Task**: Question-answering on Red Hat documentation, leveraging context from `title`, `content`, and `command` fields to generate precise responses.
- **Performance**: The model achieves high accuracy on tasks like extracting commands (e.g., `yum install X`) from provided documentation context.

## Dataset Description
The `mtpti5iD/redhat-docs_dataset` contains 55,741 entries in JSONL format, each representing a piece of Red Hat documentation. The dataset fields are:
- **title**: The title of the documentation section (e.g., "Installing Package X").
- **content**: Detailed text describing the procedure or concept.
- **command**: The command associated with the task, if applicable (may be `null`).
- **url**: A reference URL to the original documentation.

### Preprocessing
- **Null Handling**: Entries with `null` values in the `command` field were assigned an empty string (`""`). Missing `title` or `content` fields were replaced with defaults (`"Untitled"` and `""`, respectively).
- **Formatting**: Each entry was formatted into a single `text` field combining `title`, `content`, and `command` (e.g., "Title: Installing Package X\nContent: To install Package X, use the package manager yum.\nCommand: yum install X").
- **Tokenization**: The formatted text was tokenized using the FLAN-T5 tokenizer, with a maximum length of 512 tokens, applying truncation and padding as needed.

### Dataset Artifacts
The repository includes:
- **data/redhat-docs_dataset.jsonl**: The original dataset (if available).
- **data/formatted_dataset.jsonl**: The preprocessed dataset with formatted `text` fields.
- **data/tokenized_dataset.jsonl**: The tokenized dataset with `input_ids` and `labels` for training.

## Training Details
The fine-tuning process was conducted in Google Colab with the following configuration:
- **Hardware**: NVIDIA T4 GPU with ~15 GB VRAM, CUDA 11.8.
- **Epochs**: 2, balancing performance and training time (~4-8 hours for 55,741 rows).
- **Batch Size**: Effective batch size of 32 (per-device batch size of 4 with 8 gradient accumulation steps).
- **Optimizer**: Paged AdamW with 8-bit precision (`paged_adamw_8bit`).
- **Learning Rate**: Default for LoRA fine-tuning.
- **LoRA Parameters**:
  - Rank (`r`): 8
  - Alpha (`lora_alpha`): 32
  - Dropout (`lora_dropout`): 0.1
  - Target Modules: Query (`q`) and value (`v`) projection layers
- **Mixed Precision**: FP16 training for faster computation and lower memory usage.
- **Dependencies**:
  - PyTorch 2.3.1
  - Transformers 4.46.0
  - BitsAndBytes 0.43.3 (for 4-bit quantization)
  - Triton 2.0.0
  - Datasets 3.0.2
  - PEFT 0.13.2 (for LoRA)
  - Huggingface Hub 0.26.2
  - GCSFS 2025.3.2, FSSPEC 2025.3.2

### Challenges Overcome
- **Dependency Conflicts**: Resolved version mismatches (e.g., `torch 2.7.1` vs. `torchvision 0.21.0`) by aligning dependencies with CUDA 11.8.
- **Triton Errors**: Fixed `ModuleNotFoundError: No module named 'triton.ops'` by using `triton==2.0.0`.
- **Dataset Loading**: Addressed `NotImplementedError` for dataset caching by disabling cache and providing manual JSONL loading as a fallback.

## Repository Structure
- **model/**: Contains the fine-tuned model weights (`adapter_model.bin`, `config.json`) and tokenizer files (`tokenizer.json`, `special_tokens_map.json`).
- **data/**:
  - `redhat-docs_dataset.jsonl`: Original dataset (optional, if uploaded).
  - `formatted_dataset.jsonl`: Preprocessed dataset with formatted text.
  - `tokenized_dataset.jsonl`: Tokenized dataset for training.
- **finetune_script.py**: The Python script used for fine-tuning, including dependency setup, dataset preprocessing, and training configuration.
- **README.md**: This documentation file.

## Usage
The model can be used for question-answering tasks on Red Hat documentation. It expects a prompt with a question and context derived from the dataset’s `title`, `content`, and `command` fields. The model generates concise answers, often extracting commands or summarizing procedures.

### Example Use Case
- **Input Question**: "How do I install Package X?"
- **Input Context**: "Title: Installing Package X\nContent: To install Package X, use the package manager yum. Ensure you have the correct repository configured.\nCommand: yum install X"
- **Expected Output**: "Run `yum install X`."

### Loading the Model
The model and tokenizer can be loaded from this Hugging Face repository using the `transformers` library. Ensure a GPU environment with compatible dependencies for optimal performance.

### Inference Requirements
- **Hardware**: GPU recommended (e.g., NVIDIA T4 or better) for 4-bit quantization.
- **Dependencies**: Same as training (PyTorch 2.3.1, Transformers 4.46.0, etc.).
- **Prompt Format**: Combine question and context in a structured prompt for best results.

## Installation
To use the model, install the required dependencies in a Python environment (preferably with CUDA support). The fine-tuning script (`finetune_script.py`) includes dependency installation commands for reference.

## Evaluation
The model’s performance was evaluated qualitatively during development, with outputs matching expected commands and summaries for Red Hat documentation queries. Quantitative metrics (e.g., BLEU, ROUGE) were not computed due to the task’s focus on command extraction and summarization. Future work may include a test set for formal evaluation.

## Limitations
- **Dataset Availability**: The `mtpti5iD/redhat-docs_dataset` may be private or unavailable, requiring manual JSONL loading.
- **Null Values**: Some entries lack `command` fields, which may affect command-focused queries.
- **Training Scope**: Limited to 2 epochs; additional epochs may improve performance but increase training time.
- **Generalization**: The model is specialized for Red Hat documentation and may not generalize to other domains without further fine-tuning.

## Future Improvements
- **Synthetic Q&A**: Generate ~100-500 question-answer pairs to enhance question-answering performance.
- **Retrieval System**: Implement keyword matching on `title` to fetch relevant context during inference.
- **Extended Training**: Increase epochs or dataset size for better accuracy.
- **Evaluation Metrics**: Compute BLEU, ROUGE, or exact match scores on a test set.

## Contributing
Contributions are welcome! Please refer to the GitHub repository ([mtptisid/FLAN-T5-Small_finetuning_LoRA](https://github.com/mtptisid/FLAN-T5-Small_finetuning_LoRA)) for guidelines on submitting issues, pull requests, or additional datasets. Suggestions for improving the model, dataset, or documentation are appreciated.

## Acknowledgments
- **Google FLAN-T5 Team**: For providing the base `flan-t5-small` model.
- **Hugging Face**: For hosting the model, dataset, and providing the `transformers` and `datasets` libraries.
- **BitsAndBytes**: For enabling 4-bit quantization.
- **PEFT**: For LoRA implementation.
- **Red Hat Documentation Team**: For the original documentation used in the dataset.

## Contact
For questions or support, please open an issue on the GitHub repository: [mtptisid/FLAN-T5-Small_finetuning_LoRA](https://github.com/mtptisid/FLAN-T5-Small_finetuning_LoRA). Alternatively, contact the repository owner (`mtpti5iD`) via Hugging Face or GitHub.

---
*Last Updated: June 14, 2025*
