from openai import OpenAI
from dotenv import load_dotenv
import os
from datetime import datetime

load_dotenv()

class LLMClient:

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMClient, cls).__new__(cls)
            cls._instance._init_client()
        return cls._instance
    
    def _init_client(self):

        # Obtener configuración desde variables de entorno
        api_key = os.getenv("LLM_API_KEY")
        api_base_url = os.getenv("LLM_API_BASE_URL")

        # Initializar cliente de LLM
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base_url,
        )

    def generate_text(self, prompt: str) -> str:
        '''
        Método para generar texto usando el modelo LLM
        '''

        # Obtener modelo desde configuración
        model = os.getenv("LLM_MODEL", "deepseek-chat")

        # Medir tiempo de inicio
        start_time = datetime.now()

        # Informar que comienza la generación
        print(f"Generando texto con el modelo {model}...")
        
        # Obtener completion
        response = self.client.chat.completions.create(
            model=model, 
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )

        # Medir tiempo de fin
        end_time = datetime.now()

        # Calcular duración, expresar en segundos
        duration = (end_time - start_time).total_seconds()

        # Entregar feedback al usuario
        print(f"Respuesta generada en {duration} segundos.")

        # Retornar el texto generado
        return response.choices[0].message.content