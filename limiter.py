import asyncio
import time
from typing import ClassVar, Any
from langchain_core.runnables import Runnable, RunnableConfig
from dotenv import load_dotenv
import os
from logger import Logger

load_dotenv()
logger = Logger()

REQUESTS_PER_MINUTE = os.getenv("REQUESTS_PER_MINUTE", 10)
SECONDS_PER_REQUEST = 60 / int(REQUESTS_PER_MINUTE)

class LangChainRateLimiter:

    last_request_time: ClassVar[float] = 0.0
    rate_limit_lock: ClassVar[asyncio.Lock | None ] = None

    @staticmethod
    def _get_lock() -> asyncio.Lock:
        '''Lazy constructor para el lock'''
        if LangChainRateLimiter.rate_limit_lock is None:
            LangChainRateLimiter.rate_limit_lock = asyncio.Lock()

        return LangChainRateLimiter.rate_limit_lock

    @staticmethod
    async def throttle_delay():
        '''Introducir delay en las llamadas'''

        # Obtener la instancia del lock
        lock_instance = LangChainRateLimiter._get_lock()

        async with lock_instance:
            now = time.monotonic()
            time_elapsed = now - LangChainRateLimiter.last_request_time
            time_to_wait = SECONDS_PER_REQUEST - time_elapsed

            if time_to_wait > 0:
                logger.info(f"RATE LIMITER: Esperando {time_to_wait:.2f} segundos antes de la siguiente llamada...")
                await asyncio.sleep(time_to_wait)

            # Actualizar el tiempo antes de soltar la llamada
            LangChainRateLimiter.last_request_time = time.monotonic()

        return None
    

class RateLimitingRunnable(Runnable):
    '''Clase para insertar un delay dentro de la cadena de LangChain'''

    def get_type(self) -> str:
        return "rate_limiter"
    
    async def ainvoke(self, input: Any, config: RunnableConfig | None = None) -> Any:
        # En este punto de la cadena, sólo capturaremos la llamada para introducir el delay.
        # throttle_delay es estático
        await LangChainRateLimiter.throttle_delay()

        # Pasamos la misma entrada al siguiente paso de la cadena
        return input
    
    def invoke(self, input: Any, config: RunnableConfig | None = None) -> Any:
        '''No se usa, pero debe existir. Levanta excepción'''
        
        raise NotImplementedError(
            "RateLimitingRunnable is designed for asynchronous use via `ainvoke` only "
            "and should not be called synchronously."
        )