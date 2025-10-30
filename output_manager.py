from dotenv import load_dotenv
import matplotlib.pyplot as plt
import os
import re
import textwrap
import pandas as pd
from langchain_core.runnables import Runnable
from jira_client import IssueAnalysis
from datetime import datetime

load_dotenv()

class OutputManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(OutputManager, cls).__new__(cls)
            cls._instance._init_manager()
        return cls._instance
    
    def _init_manager(self):
        # Configurar directorio de salida desde variable de entorno
        self.output_dir = os.getenv("OUTPUT_DIR", "outputs")
        
        # Crear el directorio si no existe
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Crear un DataFrame vacío para la tabla de salida
        self.headers = ["HU", "GOBI", "Descripción", "Fecha de liberación", "Valor de negocio", "Métrica impactada"]
        self.data = pd.DataFrame(columns=self.headers)

        # Informar de la ruta de salida
        print(f"Output directory set to: {self.output_dir}")

    def save_output_to_text(self, filename: str, content: str) -> None:
        '''
        Método para guardar contenido en un archivo dentro del directorio de salida
        '''

        # Agrear extensión .txt al nombre del archivo (key del issue)
        if not filename.endswith(".txt"):
            filename += ".txt"

        # Crear la ruta completa del archivo
        file_path = os.path.join(self.output_dir, filename)

        # Escribir el archivo
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        # Informar al usuario
        print(f"Output saved to {file_path}")

    
    def create_visual_output(self, key, metrics_data) -> None:
        '''
        Método para crear una salida visual (placeholder)
        '''

        # Usar mapeos para los niveles de impacto
        impact_levels = {
            "Alto": "🔴",
            "Medio": "🟠",
            "Bajo": "🟡",
            "Nulo": "🟢"
        }

        level_mappging = {
            "Alto": 3,
            "Medio": 2,
            "Bajo": 1,
            "Nulo": 0
        }

        # Preparar datos para visualización
        metrics = list(metrics_data.keys())
        impacts = [level_mappging[metrics_data[m]] for m in metrics]

        # Configuración de colores
        colors_map = {3: '#2e7d32', 2: '#fff176', 1: '#f9a825', 0: '#c62828'}
        colors = [colors_map[i] for i in impacts]

        # Crear gráfico de barras
        plt.figure(figsize=(10, 6))
        bar_width = 0.4
        x_positions = range(len(metrics))

        bars = plt.bar(x_positions, impacts, color=colors, width=bar_width)

        # Etiquetas del eje X (usar nombres de métricas en varias líneas si es necesario)
        wrapped_labels = ['\n'.join(textwrap.wrap(label, 18)) for label in metrics]
        plt.xticks(ticks=x_positions, labels=wrapped_labels, rotation=0, ha='center', fontsize=10)

        # Etiquetas y título
        plt.ylim(0, 3.5)
        plt.ylabel('Nivel de Impacto', fontsize=11)
        plt.title('Impacto de HU en Métricas de Negocio', pad=30, fontsize=14, fontweight='bold')
        plt.yticks([0, 1, 2, 3], ['Nulo', 'Bajo', 'Medio', 'Alto'])
        
        # plt.xticks(rotation=45, ha='right')
        plt.subplots_adjust(bottom=0.3)

        # Rotar etiquetas del eje x para mejor legibilidad
        inv_map = {v: k for k, v in level_mappging.items()}
        for bar, impact in zip(bars, impacts):
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, inv_map[impact],
                 ha='center', va='bottom', fontsize=10, fontweight='medium')

        plt.tight_layout()

        # Crear ruta para archivo de salida
        file_path = os.path.join(self.output_dir, f"{key}.jpg")
        plt.savefig(file_path, dpi=300, bbox_inches='tight', format='png')

        # plt.show()


    def obtain_impact_list(self, text: str) -> dict:
        """
        Transforma la cadena de texto de impactos globales en un diccionario.
        
        El formato de entrada esperado es: 'Métrica 1: Valor A, Métrica 2: Valor B, ...'
        Si el valor es 'Nulo', se reemplaza por 'Bajo' para el ejemplo.
        
        Args:
            impactos_globales_str (str): La cadena de impacto recibida del LLM.
            
        Returns:
            dict: Un diccionario con el formato {'Métrica': 'Valor'}.
        """

        impacts_dict = {}

        # Separar la cadena por cada par
        pairs = text.split(', ')

        # Iterar la lista de pares
        for pair in pairs:
            # Usar ":" como separador
            if ":" in pair:
                # Obtener par de clave y valor
                key, value = pair.split(': ', 1)
                clean_value = value.strip()

            # Agregar en forma de dict (clave, valor)
            impacts_dict[key.strip()] = clean_value

        # Retornar colección
        return impacts_dict


    def add_record_to_table(self, record: dict) -> None:
        '''
        Método para crear una tabla de salida (placeholder)
        '''
        # Validar si el DataFrame está vacío ya tiene encabezados; si no, agregarlos
        if self.data.empty and list(self.data.columns) != self.headers:
            # Crear lista de encabezados
            self.data = pd.DataFrame(columns=self.headers)

        self.data = pd.concat([self.data, pd.DataFrame([record])], ignore_index=True)

            
    def clear_table(self) -> None:
        '''
        Método para limpiar la tabla de salida (placeholder)
        '''
        self.data = pd.DataFrame(columns=self.headers)

    
    def save_table_to_csv(self, filename: str) -> None:
        '''
        Método para guardar la tabla de salida en un archivo CSV
        '''

        # Agrear extensión .csv al nombre del archivo
        if not filename.endswith(".csv"):
            filename += ".csv"

        # Crear la ruta completa del archivo
        file_path = os.path.join(self.output_dir, filename)

        # Guardar el DataFrame como archivo CSV
        self.data.to_csv(file_path, index=False, encoding='utf-8-sig')

        # Informar al usuario
        print(f"Output table saved to {file_path}")


class OutputRunnable(Runnable):
    def __init__(self, output_manager):
        self.output_manager = output_manager

    def invoke(self, result, config=None):

        print(f"Generando salida para issue {result.issue_key}")
        execution = os.getenv("EXECUTION", "asynch")

        # issue_key = config.get("issue_key") if config else "unknown"
        # issue_date = config.get("issue_date") if config else "01-01"
        # epic_key = config.get("epic_key") if config else "unknown"

        # Generar salida especial para la fecha
        # release_date = datetime.fromisoformat(issue_date.replace('Z', '+00:00')).strftime('%d-%m')
            
        # Generar fila para guardar
        row = {
            "HU": result.issue_key,
            "GOBI": result.epic_key,
            "Descripción": result.resumen,
            "Fecha de liberación": result.resolution_date,
            "Valor de negocio": result.valor_negocio,
            "Métrica impactada": result.metrica_impactada
        }

        # Agregar la fila a la tabla de salida
        self.output_manager.add_record_to_table(row)

        # Convertir la respuesta del LLM en reporte (para archivo de texto)
        report = result.to_text_report(result.issue_key)

        # Guardar archivo de texto
        self.output_manager.save_output_to_text(result.issue_key, report)

        # Generar lista de impactos con formato de dict
        impact_list = self.output_manager.obtain_impact_list(result.impactos_globales)

        # Guardar impactos en gráficos
        # Esta línea sólo puede ejecutar en la versión síncrona
        if execution == "synch":
            self.output_manager.create_visual_output(result.issue_key, impact_list)


