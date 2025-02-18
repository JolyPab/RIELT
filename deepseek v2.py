import os
import asyncio
import logging
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types
from aiogram.types import Message
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import StatesGroup, State
from PyPDF2 import PdfReader
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI
from aiogram import Router

# Настройка логирования
logging.basicConfig(level=logging.DEBUG)

# Загружаем переменные окружения
load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")
PDF_FOLDER = os.getenv("PDF_FOLDER")

if not TOKEN:
    raise ValueError("Ошибка: TELEGRAM_TOKEN не найден в .env!")
if not PDF_FOLDER:
    raise ValueError("Ошибка: PDF_FOLDER не найден в .env!")

# Инициализация бота и диспетчера
bot = Bot(token=TOKEN)
dp = Dispatcher()

# Инициализация OpenAI
client = OpenAI(
    api_key="sk-aitunnel-v66J7WhiyFEFEu2fQ32zLxgdh77VudJn",
    base_url="https://api.aitunnel.ru/v1/"
)

# Состояния бота
class SearchState(StatesGroup):
    searching = State()
    last_result = State()  # Сохраняем последний найденный объект
    dialog_context = State()  # Сохраняем контекст диалога

# Функция для извлечения критериев из запроса пользователя
async def extract_criteria(user_query):
    messages = [
        {"role": "system", "content": "Ты - ассистент по недвижимости. Выдели из запроса критерии (бюджет, количество комнат, бассейн, район и т. д.). Ответь JSON-объектом."},
        {"role": "user", "content": f"Клиент написал: {user_query}"}
    ]

    response = client.chat.completions.create(
        messages=messages,
        max_tokens=100,
        model="gpt-4o-mini"  # Используем вашу модель
    )

    criteria_text = response.choices[0].message.content
    try:
        criteria = eval(criteria_text)
    except Exception:
        criteria = {}

    return criteria

# Функция для извлечения данных из текста объекта недвижимости
async def extract_property_details(text):
    messages = [
        {"role": "system", "content": "Ты - ассистент по недвижимости. Извлеки из текста информацию о цене, количестве комнат, наличии бассейна и других характеристиках. Ответь JSON-объектом."},
        {"role": "user", "content": f"Текст: {text}"}
    ]

    response = client.chat.completions.create(
        messages=messages,
        max_tokens=200,
        model="gpt-4o-mini"  # Используем вашу модель
    )

    details_text = response.choices[0].message.content
    try:
        details = eval(details_text)
    except Exception:
        details = {}

    # Добавляем проверку на наличие ключей
    if "pool" not in details:
        details["pool"] = False  # По умолчанию, если бассейн не указан
    if "rooms" not in details:
        details["rooms"] = 0  # По умолчанию, если количество комнат не указано
    if "price" not in details:
        details["price"] = float("inf")  # По умолчанию, если цена не указана

    return details

# Функция для создания базы недвижимости
async def create_embeddings():
    print("🚀 Загружаем базу недвижимости...")

    property_data = []
    metadata = []

    for filename in os.listdir(PDF_FOLDER):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(PDF_FOLDER, filename)
            pdf_reader = PdfReader(pdf_path)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

            property_data.append(text)
            metadata.append({"filename": filename})

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

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )

    knowledge_base = FAISS.from_texts(chunks, embeddings, metadatas=meta_list)
    print("✅ База недвижимости загружена!")
    return knowledge_base

# Создаём роутер
router = Router()

# Команда /start
@router.message(Command("start"))
async def start_cmd(message: Message, state: FSMContext):
    await state.clear()  # Сбрасываем состояние
    await message.answer("Привет! Я бот-помощник по недвижимости 🏡\nКакую недвижимость вы ищете?")
    await state.set_state(SearchState.searching)

# Обработка запросов пользователя
@router.message(SearchState.searching)
async def handle_search(message: Message, state: FSMContext):
    user_query = message.text

    # Извлекаем критерии из запроса
    criteria = await extract_criteria(user_query)
    await state.update_data(user_criteria=criteria)

    # Ищем похожие объекты в базе FAISS
    docs = knowledge_base.similarity_search(user_query, k=10) if knowledge_base else []
    print(f"Найдено объектов: {len(docs)}")  # Отладочный вывод

    # Фильтруем объекты по критериям
    filtered_properties = []
    for doc in docs:
        details = await extract_property_details(doc.page_content)

        # Проверяем, соответствует ли объект критериям
        if (details.get("price", float("inf")) <= criteria.get("budget", float("inf")) and
            details.get("rooms", 0) >= criteria.get("rooms", 0)):
            filtered_properties.append(doc)

    if filtered_properties:
        # Сохраняем последний найденный объект
        await state.update_data(last_result=filtered_properties[0])

        # Формируем ответ
        properties_text = "\n\n".join([
            f"🏠 *{doc.metadata.get('filename', 'Неизвестно')}*\n📋 {doc.page_content[:250]}..."
            for doc in filtered_properties
        ])

        # Получаем текущий контекст диалога
        data = await state.get_data()
        context = data.get("dialog_context", [])

        # Генерируем ответ с учетом контекста
        response = await generate_ai_response(user_query, properties_text, context)

        # Обновляем контекст диалога
        context.append({"role": "user", "content": user_query})
        context.append({"role": "assistant", "content": response})
        await state.update_data(dialog_context=context)
    else:
        response = "😕 Не нашел подходящих вариантов. Попробуйте уточнить запрос."

    await message.answer(response, parse_mode="Markdown")

# Обработка запросов на подробности
#    data = await state.get_data()
#    last_property = data.get("last_result")
#
 #   if last_property:
 #       # Извлекаем подробности о последнем найденном объекте
  #      details = await extract_property_details(last_property.page_content)
#
 #       # Формируем ответ
 #       response = f"📌 Вот подробности о {last_property.metadata.get('filename', 'неизвестном объекте')}:\n\n"
#        
 #       if "price" in details:
   #         response += f"💰 Цена: {details['price']} MXN\n"
 #       if "rooms" in details:
 #           response += f"🛏️ Комнат: {details['rooms']}\n"
 #       if "location" in details:
  #          response += f"📍 Район: {details['location']}\n"
  #      if "pool" in details:
   #         response += f"🏊 Бассейн: {'есть' if details['pool'] else 'нет'}\n"
        
        # Добавляем текст объекта
   #     response += f"\n📋 Описание: {last_property.page_content[:500]}..."
   # else:
    #    response = "😕 Не могу вспомнить последний найденный объект. Попробуйте выполнить поиск заново."

   # await message.answer(response)

# Функция для генерации ответа через GPT
async def generate_ai_response(user_query, properties, context):
    messages = [
        {"role": "system", "content": "Ты — профессиональный агент по недвижимости.  Отвечай кратко и информативно, используя данные о недвижимости, которые тебе предоставлены. Не давай общих советов, если есть конкретные данные."},
        *context,  # Передаем контекст диалога
        {"role": "user", "content": user_query},
        {"role": "assistant", "content": f"Вот несколько подходящих вариантов:\n\n{properties}"},
        {"role": "user", "content": "Напиши ответ в стиле делового общения"}
    ]

    response = client.chat.completions.create(
        messages=messages,
        max_tokens=500,
        model="gpt-4o-mini"  # Используем вашу модель
    )

    return response.choices[0].message.content

# Запуск бота
async def main():
    global knowledge_base
    print("🚀 Загружаем базу недвижимости...")
    knowledge_base = await create_embeddings()
    print("🚀 Запускаем Telegram-бота...")

    dp.include_router(router)
    await dp.start_polling(bot)

if __name__ == "__main__":
    print("🟢 Запускаем event loop...")
    asyncio.run(main())