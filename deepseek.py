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

#===Настройка логирования===
logging.basicConfig(level=logging.DEBUG)

#===Загружаем переменные окружения===
load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")
PDF_FOLDER = os.getenv("PDF_FOLDER")

if not TOKEN:
    raise ValueError("Ошибка: TELEGRAM_TOKEN не найден в .env!")
if not PDF_FOLDER:
    raise ValueError("Ошибка: PDF_FOLDER не найден в .env!")

#====Инициализация бота и диспетчера===
bot = Bot(token=TOKEN)
dp = Dispatcher()

#===Инициализация OpenAI===
client = OpenAI(
    api_key="sk-aitunnel-v66J7WhiyFEFEu2fQ32zLxgdh77VudJn",
    base_url="https://api.aitunnel.ru/v1/"
)

#===Состояния бота===
class SearchState(StatesGroup):
    searching = State()
    last_result = State()  
    dialog_context = State()

#===Функция для извлечения критериев из запроса пользователя===
async def extract_criteria(user_query):
    messages = [
        {"role": "system", "content": "Eres asistente de bienes raíces. Selecciona de la consulta los criterios (presupuesto, número de habitaciones, piscina, área, etc.). Responde con un objeto JSON."},
        {"role": "user", "content": f"El cliente escribió: {user_query}"}
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

#===Функция для извлечения данных из текста объекта недвижимости===
async def extract_property_details(text):
    messages = [
        {"role": "system", "content": "Eres asistente de bienes raíces. Extraiga del texto información sobre el precio, el número de habitaciones, la disponibilidad de piscina y otras características. Responde con un objeto JSON."},
        {"role": "user", "content": f"Texto: {text}"}
    ]

    response = client.chat.completions.create(
        messages=messages,
        max_tokens=200,
        model="gpt-4o-mini"  
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
        details["price"] = float("inf")  

    return details

#===Функция для создания базы недвижимости===
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

#===Создаём роутер===
router = Router()

#===Команда /start===
@router.message(Command("start"))
async def start_cmd(message: Message, state: FSMContext):
    await state.clear()  
    await message.answer("Hola! Soy Angela, tu asistente de bienes raíces. 🏡\n Qué tipo de propiedad estás buscando?")
    await state.set_state(SearchState.searching)


#=====================================
#===Обработка запросов пользователя===
#=====================================
@router.message(SearchState.searching)
async def handle_search(message: Message, state: FSMContext):
    user_query = message.text

    #===критерии из запроса===
    criteria = await extract_criteria(user_query)
    await state.update_data(user_criteria=criteria)

    #===поиск объектов в базе FAISS===
    docs = knowledge_base.similarity_search(user_query, k=10) if knowledge_base else []
    print(f"Найдено объектов: {len(docs)}")  # Отладочный вывод

    #===Фильтр объекта по критериям===
    filtered_properties = []
    for doc in docs:
        details = await extract_property_details(doc.page_content)

        #===соответствие объекта по критериям
        if (details.get("price", float("inf")) <= criteria.get("budget", float("inf")) and
            details.get("rooms", 0) >= criteria.get("rooms", 0)):
            filtered_properties.append(doc)

    if filtered_properties:
        #==последний найденный объект===
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

        #===контекст диалога===
        context.append({"role": "user", "content": user_query})
        context.append({"role": "assistant", "content": response})
        await state.update_data(dialog_context=context)
    else:
        response = "😕 No encontré las opciones adecuadas. Intenta refinar la consulta."

    await message.answer(response, parse_mode="Markdown")


#=======================================
#===Обработка запросов на подробности===
#=======================================
@router.message(lambda message: any(keyword in message.text.lower() for keyword in ["más", "dime", "qué inmueble", "precio", "costo"]))
async def handle_details(message: Message, state: FSMContext):
    data = await state.get_data()
    last_property = data.get("last_result")

    if last_property:
        #===Извлечение подробности о последнем найденном объекте===
        details = await extract_property_details(last_property.page_content)

        #===ответ===
        response = f"📌 Aquí están los detalles sobre {last_property.metadata.get('filename', 'objeto desconocido')}:\n\n"
        
        if "price" in details:
            response += f"💰 Precio: {details['price']} MXN\n"
        if "rooms" in details:
            response += f"🛏️ Habitación: {details['rooms']}\n"
        if "location" in details:
            response += f"📍 Área: {details['location']}\n"
        if "pool" in details:
            response += f"🏊 Piscina: {'есть' if details['pool'] else 'нет'}\n"
        
        #===текст объекта===
        response += f"\n📋 Descripción: {last_property.page_content[:500]}..."
    else:
        response = "😕 No recuerdo el último objeto encontrado. Intente buscar de nuevo."

    await message.answer(response)

#===Функция для генерации ответа через GPT===
async def generate_ai_response(user_query, properties, context):
    messages = [
        {"role": "system", "content": "Eres un agente inmobiliario profesional. Responda de manera concisa e informativa utilizando los datos inmobiliarios que se le proporcionan. No dé consejos generales si hay datos específicos."},
        *context,
        {"role": "user", "content": user_query},
        {"role": "assistant", "content": f"Aquí hay algunas opciones apropiadas:\n\n{properties}"},
        {"role": "user", "content": "Escribe una respuesta al estilo de la comunicación empresarial"}
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