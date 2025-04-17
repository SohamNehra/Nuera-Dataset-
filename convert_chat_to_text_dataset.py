import json

input_file = "nuera_raw.jsonl"         # your existing file
output_file = "nuera_formatted.jsonl"  # output to use for training

with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
    for line in fin:
        try:
            sample = json.loads(line)
            messages = sample.get("messages", [])
            
            # Build a single string with speaker tags
            dialogue = ""
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                
                if role == "user":
                    dialogue += f"<|user|>: {content}\n"
                elif role == "assistant":
                    dialogue += f"<|assistant|>: {content}\n"

            dialogue = dialogue.strip()  # remove last newline

            # Write final formatted sample
            fout.write(json.dumps({"text": dialogue}) + "\n")

        except json.JSONDecodeError:
            print("Skipping invalid JSON line")
