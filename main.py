import os
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional


from aiogram import Bot, Dispatcher, types
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.filters import Command
from dotenv import load_dotenv
import google.genai as genai  


load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not BOT_TOKEN or not GEMINI_API_KEY:
    print("Ошибка: Ключи не найдены в переменных окружения!")
    exit(1)


MODEL_NAME = "gemini-1.5-pro"
MAX_RETRIES = 5  
RATE_LIMIT_PER_MINUTE = 10  

# конфиг логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# класс для ассинк клиента
class AsyncGeminiClient:
    def __init__(self, model: str = MODEL_NAME, max_retries: int = MAX_RETRIES):
        self.client = genai.GenerativeModel(model)
        self.max_retries = max_retries
    
    async def generate(self, prompt: str) -> Optional[str]:
        """
        Генерирует ответ от Gemini с retry и защитой от блокировки.
        
        Args:
            prompt (str): Текст запроса
            
        Returns:
            Optional[str]: Ответ или None если ошибка
        """
        for attempt in range(self.max_retries):
            try:
                # синхронный вызов
                response = await asyncio.to_thread(
                    self._sync_generate,
                    prompt
                )
                
                if not response.strip():
                    logging.warning(f"Empty response from Gemini. Attempt {attempt + 1}")
                    continue
                
                return response
            
            except Exception as e:
                logging.error(f"[Attempt {attempt+1}] Ошибка при генерации: {e}", exc_info=True)
            
            # пауза с откатом
            delay = (2 ** attempt) * 1.5 + (time.time() % 3)  
            logging.info(f"Повторная попытка через {delay:.2f} секунды...")
            
            await asyncio.sleep(delay)
        
        return None
    
    def _sync_generate(self, prompt: str) -> str:
        """Синхронный метод для вызова модели"""
        try:
            response = self.client.generate_content(prompt)
            if hasattr(response, "text") and response.text.strip():
                return response.text
            else:
                logging.error("Ответ от Gemini пустой или неструктурированный.")
                return ""
            
        except Exception as e:
            # логирование
            logging.exception(f"Синхронный вызов модели провалился: {e}")
            raise

class RateLimiter:
    def __init__(self, max_per_minute: int = RATE_LIMIT_PER_MINUTE):
        self.max = max_per_minute
        self.requests: list[datetime] = []
    
    async def allow(self) -> bool:
        """Проверяет, разрешено ли выполнение запроса."""
        now = datetime.now()
        
        # очистка старых запросов
        self.requests = [r for r in self.requests if now - r < timedelta(minutes=1)]
        
        if len(self.requests) >= self.max:
            return False
        
        self.requests.append(now)
        return True

# инициализация
session = AiohttpSession()
bot = Bot(token=BOT_TOKEN, session=session)
dp = Dispatcher()

# google-genai с aiohttp для лучшей производительности
genai.configure(api_key=GEMINI_API_KEY)

# инстанцирование
gemini_client = AsyncGeminiClient()
rate_limiter = RateLimiter()

@dp.message(Command("start"))
async def start_cmd(message: types.Message):
    await message.answer("Привет! Я Google Gemini. Спрашивай что угодно.")

@dp.message()
async def handle_message(message: types.Message):
    """Обработчик сообщений"""
    
    # проверка rate limit
    if not await rate_limiter.allow():
        return await message.answer("Очень жалко, но я сейчас перегружен. Подожди минуту и попробуй снова.")
    
    try:
        # статус "печатает"
        await bot.send_chat_action(chat_id=message.chat.id, action="typing")
        
        prompt = message.text.strip()
        if not prompt:
            return await message.answer("Пустой запрос. Что ты хочешь спросить?")
        
        logging.info(f"[User {message.from_user.id}] Получен запрос: '{prompt}'")
        
        # генерация ответа
        response_text = await gemini_client.generate(prompt)
        
        if not response_text:
            return await message.answer("Gemini прислал пустой ответ. Возможно, проблема в запросе.")
        
        await message.reply(response_text[:4096])  # ограничения по длине сообщения
        
    except Exception as e:
        logging.error(f"Ошибка при обработке сообщения: {e}", exc_info=True)
        await message.answer("Произошла ошибка доступа. Попробуй включить VPN на компьютере и перезапустить бота.")

async def main():
    """Запуск бота"""
    logging.info("Бот запущен!")
    
    # добавляем middleware 
    @bot.session.middleware()
    async def log_response_time(
        make_request: types.NextRequestMiddlewareType[types.TelegramType],
        bot: "Bot",
        method: types.TelegramMethod[types.TelegramType]
    ):
        start = time.time()
        try:
            response = await make_request(bot, method)
            duration = time.time() - start
            logging.info(f"Ответ от {method} занял {duration:.2f}s")
        except Exception as e:
            logging.error(f"Ошибка в middleware: {e}", exc_info=True)

    # пулинг с таймаутом
    await dp.start_polling(bot, timeout=30)  

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Бот остановлен пользователем.")
