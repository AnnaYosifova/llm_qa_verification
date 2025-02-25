import csv
import os
import pandas as pd
from dotenv import load_dotenv

from functions.prompts_loader import load_prompts
from functions.csv_handler import load_or_create_csv, save_to_csv
from models.openai import openai_model


load_dotenv("config/.env")
prompts = load_prompts()
df = load_or_create_csv(os.getenv("output_csv"))

with open(os.getenv("questions_csv"), "r", encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)

    for row in reader:
        qaid = row["id"]
        lang = row["language"]
        question = row["question"]
        answer = row["answer"]
        score = row["score"]

        for prompt_name, prompt_data in prompts.items():
            prompt_text = prompt_data["text"].format(question=question, answer=answer)
            print(f"=== {prompt_name} ===\n{prompt_text}\n\n")

            model_answer = openai_model.get_response(prompt_text)
            print(f"Ответ модели:\n{model_answer}\n")

            new_row = {
                "model": "openai gpt-4",
                "prompt_type": prompt_data.get("type", "N/A"),
                "persuation": prompt_data.get("persuasion", "N/A"),
                "qaid": qaid,
                "prompt_text": prompt_text,
                "model_answer": model_answer,
                "score": score
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        
        break

save_to_csv(df, os.getenv("output_csv"))
print(f"Данные успешно добавлены в {os.getenv('output_csv')}")
