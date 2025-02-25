import os
from dotenv import load_dotenv
import openai

class OpenAIClient:
    def __init__(self):
        load_dotenv(os.path.join(os.path.dirname(__file__), "../config/.env")) 
        api_key = os.getenv("openai_api_key")
        if not api_key:
            raise ValueError("API-ключ OpenAI не найден! Добавьте его в .env")
        
        self.client = openai.OpenAI(api_key=api_key)

    def get_response(self, prompt, model="gpt-4", temperature=0.7, max_tokens=100):
        """
        Отправляет запрос в OpenAI API и получает ответ.
        """
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Ошибка при запросе в OpenAI: {e}")
            return None

openai_model = OpenAIClient()
