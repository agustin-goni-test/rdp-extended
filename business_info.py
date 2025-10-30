from dotenv import load_dotenv
import os

load_dotenv()

class BusinessInfo:
    # Colección que guarda los contextos de negocio ya leídos
    _business_info_files: dict = {}
    _info_folder: str

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(BusinessInfo, cls).__new__(cls)
            cls.instance._init_business_info()
        return cls.instance
    
    def _init_business_info(self):
        self._info_folder = os.getenv("BUSINESS_INFO_FOLDER", "sources")

        print(f"Información del negocio para obtener desde: {self._info_folder}")


    def get_business_info(self, filename: str) -> str:
        '''
        Método para obtener información del negocio desde un archivo.
        En este versión está en desuso
        '''

        # Agregar extensión TXT al archivo
        if not filename.endswith(".txt"):
            filename += ".txt"

        # Si el archivo no está codificado dentro de la colección de contextos
        # (la colección es un par ordenado con nombre de archivo como clave y contenido
        # de archivo como valor)
        if filename not in self._business_info_files:

            # Obtener la ruta hacia el archivo
            file_path = os.path.join(self._info_folder, filename)

            try:
                # Leer el archivo
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                    # Agregar a la colección de contextos de la clase
                    self._business_info_files[filename] = content
                    print(f"Información del negocio cargada desde {file_path}")
            
            except FileNotFoundError:
                print(f"Archivo no encontrado: {file_path}")
                self._business_info_files[filename] = ""
        
        return self._business_info_files[filename]
    

    def epic_already_read(self, epic_key: str) -> bool:
        '''
        Valida si ya fue leída la información de la épica.
        '''
        
        # Nombre del archivo, validar si es txt
        filename = epic_key
        if not filename.endswith(".txt"):
            filename += ".txt"

        # Retorna verdadero si el archivo ya estaba cargado
        return filename in self._business_info_files
    

    def add_epic_to_list(self, epic_key: str, content: str):

        # Usar la clave de épica como nombre de archivo y validar que
        # tiene extensión TXT
        filename = epic_key
        if not filename.endswith(".txt"):
            filename += ".txt"

        # Agregar a la colección con par (llave, valor),
        # donde llave = filename y valor = content
        self._business_info_files[filename] = content

    def get_epic_from_list(self, epic_key:str) -> str:

        # Usar la clave de épica como nombre de archivo y validar que
        # tiene extensión TXT
        filename = epic_key
        if not filename.endswith(".txt"):
            filename += ".txt"

        # Agregar a la colección con par (llave, valor),
        # donde llave = filename y valor = content
        return self._business_info_files[filename]
            
        
    # Método en desuso
    def get_business_info_legacy(self) -> str:
        '''
        Método para obtener información del negocio
        '''

        info = """
        Principales objetivos de la iniciativa Cuota Comercio:
        1. Permitir a los comercios ofrecer cuotas sin interés a sus clientes.
        El objetivo es flexibilizar las opciones de pago que los comercios le ofrece a sus clientes finales,
        con el objetivo de habilitar la venta por cuotas, sin la necesidad de tener que contar con un
        convenio con algún emisor para ofrecer este beneficio.

        2. Incrementar las ventas y el ticket promedio en los comercios.
        Este mecanismo permite a los clientes del comercio segmentar el pago de algunos artículos en cuotas,
        lo que puede incentivar la compra de productos de mayor valor o en mayor cantidad.
        
        3. Mejorar el posicionamiento frente a la competencia.
        Es una necesidad para mantener la competitividad en el mercado, ya que muchos comercios ya ofrecen esta opción,
        y no contar con ella puede resultar en la pérdida de clientes potenciales.


        Principales métricas a impactar:
        1. Cantidad de comercios
        Fomentar a que más comercios contraten el producto de Cuota Comercio.

        2. Completitiud de la información
        Permitir a los comercios que cuenta con el producto acceder a información completa y oportuna sobre sus ventas en cuotas.

        3. Robustez del sistema
        Asegurar que la plataforma soporte el crecimiento en la cantidad de comercios y transacciones

        4. Nivel de servicio
        Minimizar las incidencias relacionadas con la operación del producto. Esto significa que el producto pueda operarar en
        todo su ciclo de vida sin problemas que afecten la disponibilidad del servicio, impidan operar a los clientes, o generen
        resultados confusos que no sean fáciles de entender para los comercios.
        """

        return info

    