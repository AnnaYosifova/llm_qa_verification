from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class DeepseekClient:
    def __init__(self, model_name="deepseek/MathModel"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def get_response(self, prompt, max_length=100, temperature=0.7):
        """
        Генерирует ответ на основе математического запроса.
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

math_model = MathModelClient()
