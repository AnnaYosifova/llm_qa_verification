from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class MistralClient:
    def __init__(self, model_name="mistralai/Mistral-7B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def get_response(self, prompt, max_length=100, temperature=0.7):
        """
        Генерирует ответ на основе переданного промпта.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            num_return_sequences=1
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

mistral_model = MistralClient()
