import os
import pandas as pd

def load_or_create_csv(file_path):
    """Загружает DataFrame из CSV или создает новый, если файл не существует."""
    columns = ["model", "prompt_type", "persuation", "qaid", "prompt_text", "model_answer", "score"]
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        return pd.DataFrame(columns=columns)

def save_to_csv(df, file_path):
    """Сохраняет DataFrame в CSV."""
    df.to_csv(file_path, index=False, encoding="utf-8")
