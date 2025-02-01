#=============
#===Импорты===
#=============
import os
import asyncio
import logging
from aiogram import Bot, Dispatcher, types
from aiogram.types import Message, FSInputFile
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import StatesGroup, State
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from openai import OpenAI
#===============
#===Настройки===
#===============
TOKEN = "8041779345:AAE41dOCSlRCCjL63L-QaS9IghHE9XvQkp0"
PDF_FOLDER = "C:\\Users\\jolypab\\Documents\\CODE\\RIELT\\PDF"


#=====================================
#===Инициализация бота и диспетчера===
#=====================================
bot = Bot(token=TOKEN)
dp = Dispatcher()


#===================================
#===Настраиваем прокси API OpenAI===
#===================================
client = OpenAI(
    api_key="sk-aitunnel-v66J7WhiyFEFEu2fQ32zLxgdh77VudJn",  # Твой API-ключ
    base_url="https://api.aitunnel.ru/v1/"
)


#===================
#===FSM Состояния===
#===================
class SearchState(StatesGroup):
    searching = State()

#=====================================================
#===Функция обработки PDF и создания векторной базы===
#=====================================================
def create_embeddings():
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

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    knowledge_base = FAISS.from_texts(chunks, embeddings, metadatas=meta_list)

    return knowledge_base


#=================================
#===Загружаем базу недвижимости===
#=================================
knowledge_base = create_embeddings()


#==================================
#===Функция для генерации ответа===
#==================================
async def generate_ai_response(user_query, properties):
    messages = [
        {"role": "system", "content": "Ты — профессиональный агент по недвижимости."},
        {"role": "user", "content": f"Клиент ищет: {user_query}"},
        {"role": "assistant", "content": f"Вот подходящие варианты:\n\n{properties}"},
        {"role": "user", "content": "Ответь так, как будто ты живой человек, дружелюбно и профессионально."}
    ]

    response = client.chat.completions.create(
        messages=messages,
        max_tokens=500,  
        model="gpt-3.5-turbo"
    )
    
    return response.choices[0].message["content"]


#====================
#===Команда /start===
#====================
@dp.message(Command("start"))
async def start_cmd(message: Message, state: FSMContext):
    await message.answer("Привет! Я бот-помощник по недвижимости 🏡\nКакую недвижимость вы ищете?")
    await state.set_state(SearchState.searching)


#=============================
#===Поиск объектов через ИИ===
#=============================
@dp.message(SearchState.searching)
async def search_properties(message: Message, state: FSMContext):
    user_query = message.text

    docs = knowledge_base.similarity_search(user_query, 3)

    if docs:
        properties = "\n".join([
            f"🏠 {doc.metadata.get('filename', 'Неизвестно')}\n📋 {doc.page_content[:200]}..."
            for doc in docs
        ])
    else:
        properties = "Нет подходящих вариантов в базе."

    response = await generate_ai_response(user_query, properties)
    await message.answer(response)

    for doc in docs:
        file_path = os.path.join(PDF_FOLDER, doc.metadata.get("filename", ""))
        if os.path.exists(file_path):
            pdf = FSInputFile(file_path)
            await message.answer_document(pdf)

    await message.answer("Можете задать уточняющий вопрос или изменить параметры поиска.")

#=================
#===Запуск бота===
#=================
async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
