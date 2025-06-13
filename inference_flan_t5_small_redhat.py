from transformers import T5Tokenizer, T5ForConditionalGeneration
from peft import PeftModel

# Load fine-tuned model
model_path = "./finetuned_flan_t5_small"
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path, device_map="auto")

# Example prompt
question = "How do I install Package X?"
context = "Title: Installing Package X\nContent: To install Package X, use the package manager yum. Ensure you have the correct repository configured.\nCommand: yum install X"
prompt = f"Question: {question} Context: {context}"

# Tokenize and generate
inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to("cuda")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
