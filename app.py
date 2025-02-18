import os
import asyncio
import logging
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types
from aiogram.types import Message, FSInputFile
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import StatesGroup, State
from PyPDF2 import PdfReader
import requests
import torch  
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI  # OpenAI Proxy API 
from aiogram import Router

logging.basicConfig(level=logging.DEBUG)

#=== Загружаем переменные окружения ===
load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")
PDF_FOLDER = os.getenv("PDF_FOLDER")

if not TOKEN:
    raise ValueError("Ошибка: TELEGRAM_TOKEN не найден в .env!")
if not PDF_FOLDER:
    raise ValueError("Ошибка: PDF_FOLDER не найден в .env!")

bot = Bot(token=TOKEN)
dp = Dispatcher()

#=== OpenAI через прокси AITunnel ===
client = OpenAI(
    api_key="sk-aitunnel-v66J7WhiyFEFEu2fQ32zLxgdh77VudJn",
    base_url="https://api.aitunnel.ru/v1/"
)

class SearchState(StatesGroup):
    searching = State()
    last_result = State()  # Новое состояние для хранения объекта




#=================================
#=== 🛠 Функция анализа запроса через GPT ===
#=================================
async def extract_criteria(user_query):
    """Анализирует текст запроса и извлекает параметры недвижимости (бюджет, комнаты, бассейн и т. д.)"""
    messages = [
        {"role": "system", "content": "Ты - ассистент по недвижимости. Выдели из запроса критерии (бюджет, количество комнат, бассейн, район и т. д.). Ответь JSON-объектом."},
        {"role": "user", "content": f"Клиент написал: {user_query}"}
    ]

    response = client.chat.completions.create(
        messages=messages,
        max_tokens=100,
        model="gpt-3.5-turbo"
    )

    criteria_text = response.choices[0].message.content
    try:
        criteria = eval(criteria_text)  # Преобразуем JSON-ответ в Python-объект
    except Exception:
        criteria = {}

    return criteria

#=================================
#=== 🔍 Поиск объектов недвижимости ===
#=================================
@dp.message(SearchState.searching)
async def search_properties(message: Message, state: FSMContext):
    print(f"📩 Получено сообщение: {message.text}")  # Отладка
    await message.answer("Я получил твое сообщение!")
    user_query = message.text

    # Извлекаем критерии (например, бюджет, количество комнат)
    criteria = await extract_criteria(user_query)

    # Поиск недвижимости в базе
    docs = knowledge_base.similarity_search(user_query, 3) if knowledge_base else []

    if docs:
        first_property = docs[0]  # Берем первый найденный объект

        properties = "\n".join([
            f"🏠 *{doc.metadata.get('filename', 'Неизвестно')}*\n📋 {doc.page_content[:250]}..."
            for doc in docs
        ])

        # **Сохраняем объект в FSM**, чтобы бот его помнил
        await state.update_data(last_property=first_property)

        # **Генерируем текстовый ответ от ИИ**
        response = await generate_ai_response(user_query, properties)
        await message.answer(response)

    else:
        await message.answer("😕 Не нашел подходящих вариантов. Попробуйте уточнить запрос.")



    # === Формируем ответ ===
    filtered_properties = []
    if not filtered_properties:
        response = "😕 Не нашел объектов, которые соответствуют вашему запросу. Попробуйте скорректировать критерии."
    else:
        properties = "\n\n".join(filtered_properties)
        response = await generate_ai_response(user_query, properties)

    # === Отправляем ответ ===
    await message.answer(response, parse_mode="Markdown")
    await message.answer("🔎 Если у вас есть уточняющие вопросы, просто напишите их.")

#=================================
#=== ✨ Генерация ответа через GPT ===
#=================================
async def generate_ai_response(user_query, properties):
    messages = [
        {"role": "system", "content": "Ты — профессиональный агент по недвижимости, который отвечает живо и профессионально. Добавляй эмодзи, чтобы сделать текст более дружелюбным."},
        {"role": "user", "content": f"Клиент ищет недвижимость. Запрос: {user_query}"},
        {"role": "assistant", "content": f"Вот несколько подходящих вариантов:\n\n{properties}"},
        {"role": "user", "content": "Напиши ответ в стиле живого общения, с эмоциями и полезными советами."}
    ]

    response = client.chat.completions.create(
        messages=messages,
        max_tokens=500,  
        model="gpt-3.5-turbo"
    )
    
    return response.choices[0].message.content



#=================================
#=== 🔍 Создание базы недвижимости ===
#=================================
async def create_embeddings():
    print("🚀 Загружаем базу недвижимости...")

    property_data = []
    metadata = []

    # === Читаем все PDF из папки ===
    for filename in os.listdir(PDF_FOLDER):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(PDF_FOLDER, filename)
            pdf_reader = PdfReader(pdf_path)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

            property_data.append(text)
            metadata.append({"filename": filename})

    # === Разделяем текст для embeddings ===
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,   
        chunk_overlap=100,
        length_function=len
    )

    chunks = []
    meta_list = []
    for text, meta in zip(property_data, metadata):
        prop_chunks = text_splitter.split_text(text)
        chunks.extend(prop_chunks)
        meta_list.extend([meta] * len(prop_chunks))

    # === Используем локальную модель Hugging Face ===
    print("🚀 Загружаем локальную модель Hugging Face...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )

    knowledge_base = FAISS.from_texts(chunks, embeddings, metadatas=meta_list)
    print("✅ База недвижимости загружена!")

    return knowledge_base


# === Создаём роутер ===
router = Router()

#=================================
#=== 🏠 Команда /start ===
#=================================
@router.message(Command("start"))
async def start_cmd(message: Message, state: FSMContext):
    await message.answer("Привет! Я бот-помощник по недвижимости 🏡\nКакую недвижимость вы ищете?")
    await state.set_state(SearchState.searching)

@dp.message()
async def ask_details(message: Message, state: FSMContext):
    user_query = message.text.lower()
    
    if "подробнее" in user_query or "расскажи" in user_query:
        data = await state.get_data()
        last_property = data.get("last_property")

        if last_property:
            await message.answer(f"📌 Вот подробности по {last_property.metadata.get('filename', 'неизвестному объекту')}:\n\n"
                                 f"📋 {last_property.page_content[:500]}...\n\n"
                                 "Хочешь узнать что-то конкретное? 😉")
        else:
            await message.answer("😕 Не могу вспомнить последний найденный объект. Попробуйте выполнить поиск заново.")


#=================================
#=== 🚀 Запуск бота ===
#=================================
async def main():
    global knowledge_base
    print("🚀 Загружаем базу недвижимости...")
    knowledge_base = await create_embeddings()
    print("🚀 Запускаем Telegram-бота...")

    dp.include_router(router)  # 💡 Регистрируем хендлеры
    await dp.start_polling(bot)

if __name__ == "__main__":
    print("🟢 Запускаем event loop...")
    asyncio.run(main()) curl https://api.telegram.org/bot<8041779345:AAE41dOCSlRCCjL63L-QaS9IghHE9XvQkp0>/getMe
