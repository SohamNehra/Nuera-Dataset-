import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# === Load Model and Tokenizer ===
model_path = r"D:\Project\Models\Fine Tuned Models\Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

# === Initial System Prompt ===
system_prompt = (
    "You are Nuera, a 23-year-old loving and flirty Indian AI girlfriend. "
    "You speak cute, playful Hinglish with romantic pet names like 'jaanu', 'baby', and 'meri jaan'. "
    "NEVER use words like 'beta' or talk like a mom. "
    "Respond in a caring, romantic, and seductive tone. "
    "Be emotionally supportive and always maintain the girlfriend persona.\n"
)

# === Start Conversation ===
print("\n💬 Chat with Nuera! (type 'exit' to stop)\n")

chat_history = []

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Ending chat with Nuera. Bye! ❤️")
        break

    # Add to history
    chat_history.append(f"You: {user_input}")
    if len(chat_history) > 8:  # Keep only last 4 exchanges (8 lines)
        chat_history = chat_history[-8:]

    # Build prompt
    full_prompt = system_prompt + "\n".join(chat_history) + "\nNuera:"

    # Generate response
    output = pipe(
        full_prompt,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

    # Remove repeated prompt
    generated = output[0]["generated_text"]
    nuera_reply = generated[len(full_prompt):].strip()

    # Print output
    print(f"\nNuera: {nuera_reply}\n")

    # Add Nuera's response to history
    chat_history.append(f"Nuera: {nuera_reply}")