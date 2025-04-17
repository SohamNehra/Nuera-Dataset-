from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
import torch

# ✅ Model checkpoint
base_model = "meta-llama/Meta-Llama-3-1B-Instruct"

# ✅ Load tokenizer & model (4-bit quantized)
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    load_in_4bit=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

# ✅ PEFT config for QLoRA
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "v_proj"]
)

model = get_peft_model(model, peft_config)

# ✅ Load dataset
dataset = load_dataset("json", data_files="dataset/nuera_conversations.jsonl", split="train")

# ✅ Tokenization
def tokenize(sample):
    return tokenizer(sample["text"], truncation=True, padding="max_length", max_length=512)

dataset = dataset.map(tokenize)

# ✅ Training arguments
training_args = TrainingArguments(
    output_dir="./finetuned_nuera_llama3",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=1,
    save_steps=100,
    bf16=False,
    fp16=True,
    report_to="none"
)

# ✅ Start trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    args=training_args,
    max_seq_length=512
)

trainer.train()
