import yaml

def load_prompts(file_path="prompts.yaml"):
    """Загружает промпты из YAML-файла."""
    with open(file_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)["prompts"]
