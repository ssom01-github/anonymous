import os
import time
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configuration
MODEL_NAME = "meta-llama/Llama-3.2-3B-instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AVG_LENGTH = 250 # VEify once
NUM_ARTICLES = 1000
LANGUAGE = "Hindi"
INPUT_FILE = "Datasetlink.xlsx"
OUTPUT_FILE = f"Generated_Articles_From_Summary_llama_{NUM_ARTICLES}.csv"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
)

def generate_article(summary, avg_length, language="Hindi"):
    """
    Generate a detailed article in the specified language from the given summary.
    """
    messages = [
        {
            "role": "system",
            "content": f"You are an assistant skilled in writing {language} news articles.",
        },
        {
            "role": "user",
            "content": (
                f"Expand the following summary into a {language} article of approximately "
                f"{avg_length} words.\n\nSummary: \"{summary}\". "
                f"Please make sure the article is detailed and close to {avg_length} words, "
                f"but do not exceed {avg_length + 100} words."
            ),
        },
    ]

    try:
        prompt_ids = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(model.device)

        attention_mask = (prompt_ids != tokenizer.pad_token_id).long()
        approx_tokens = int(avg_length * 1.5)

        output = model.generate(
            input_ids=prompt_ids,
            attention_mask=attention_mask,
            max_new_tokens=int(approx_tokens * 0.9),
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )

        full_text = tokenizer.decode(output[0], skip_special_tokens=True)

        # Remove the prompt from the response if it's repeated
        user_message = messages[-1]["content"].strip()
        response = full_text.split(user_message)[-1].strip() if user_message in full_text else full_text.strip()

        # Clean and format
        return ' '.join(response.replace("\n", " ").split())

    except Exception as e:
        print(f"Error generating article: {e}")
        return None

def main():
    try:
        df = pd.read_excel(INPUT_FILE).head(NUM_ARTICLES)
    except Exception as e:
        print(f"Error loading file {INPUT_FILE}: {e}")
        return

    generated_data = []

    for idx, row in df.iterrows():
        summary = row.get("summary", "")
        print(f"[{idx+1}/{NUM_ARTICLES}] Processing summary: {summary[:60]}...")

        full_article = generate_article(summary, AVG_LENGTH)
        if full_article:
            print(f"Generated article (preview): {full_article[:100]}...")
        else:
            print("Article generation failed, skipping...\n")
            continue

        generated_data.append({
            "summary": summary,
            "original_article": row.get("full_article", ""),
            "__ORIGINAL_ARTICLE_END__": "__ORIGINAL_ARTICLE_END__",
            "generated_article": full_article,
            "__GENERATED_ARTICLE_END__": "__GENERATED_ARTICLE_END__"
        })

        

    if generated_data:
        pd.DataFrame(generated_data).to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
        print(f"\nArticles saved to: {OUTPUT_FILE}")
    else:
        print("\nNo articles generated.")

if __name__ == "__main__":
    main()
