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
from langchain_huggingface import HuggingFaceEmbeddings  
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI  
from aiogram import Router

# ===Настройка логирования===
logging.basicConfig(level=logging.DEBUG)

# ===Загружаем переменные окружения===
load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")
PDF_FOLDER = os.getenv("PDF_FOLDER")
OPENAI_API_KEY = "sk-aitunnel-v66J7WhiyFEFEu2fQ32zLxgdh77VudJn"  
OPENAI_API_BASE = "https://api.aitunnel.ru/v1/"  

if not TOKEN:
    raise ValueError("Ошибка: TELEGRAM_TOKEN не найден в .env!")
if not PDF_FOLDER:
    raise ValueError("Ошибка: PDF_FOLDER не найден в .env!")

# ===Инициализация бота и диспетчера===
bot = Bot(token=TOKEN)
dp = Dispatcher()

# ===Состояния бота===
class SearchState(StatesGroup):
    searching = State()

#=== Функция для извлечения текста из PDF===
async def extract_text_from_pdf(pdf_path):
    try:
        pdf_reader = PdfReader(pdf_path)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        logging.error(f"Ошибка при чтении файла {pdf_path}: {e}")
        return ""

#=== Функция для создания и сохранения базы данных===
async def create_embeddings():
    logging.info("🚀 Загружаем базу недвижимости...")

    property_data = []
    metadata = []

    for filename in os.listdir(PDF_FOLDER):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(PDF_FOLDER, filename)
            text = await extract_text_from_pdf(pdf_path)

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
    
    # Сохраняем базу данных
    knowledge_base.save_local("faiss_index")
    logging.info("✅ База недвижимости загружена и сохранена!")
    return knowledge_base

# ===Функция для загрузки базы данных===
def load_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )
    
    if os.path.exists("faiss_index"):
        knowledge_base = FAISS.load_local(
            "faiss_index", 
            embeddings, 
            allow_dangerous_deserialization=True  
        )
        logging.info("✅ База недвижимости загружена из файла!")
    else:
        knowledge_base = None
        logging.warning("⚠️ База недвижимости не найдена. Сначала создайте её.")
    
    return knowledge_base

# === роутер===
router = Router()

# ===Инициализация памяти LangChain===
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# ===Промпт для RAG===
template = """Eres un asistente de bienes raíces. Tu tarea es vender casas de los documentos. Usa los siguientes documentos para responder a la pregunta. Si no hay opciones adecuadas, dilo.

Документы:
{context}

Вопрос: {question}
Ответ:"""
QA_PROMPT = PromptTemplate.from_template(template)

#=== Инициализация ChatOpenAI с прокси===
llm = ChatOpenAI(
    model="gpt-4o-mini",  
    temperature=0,  
    openai_api_key=OPENAI_API_KEY,  
    openai_api_base=OPENAI_API_BASE  
)

# ===Команда /start===
@router.message(Command("start"))
async def start_cmd(message: Message, state: FSMContext):
    await state.clear()  
    await message.answer("Hola! Soy Angela, tu asistente de bienes raíces. 🏡\n Qué tipo de propiedad estás buscando?")
    await state.set_state(SearchState.searching)

#===Обработка запросов пользователя===
@router.message(SearchState.searching)
async def handle_search(message: Message, state: FSMContext):
    try:
        user_query = message.text

        # === RAG для поиска и генерации ответа
        retriever = knowledge_base.as_retriever(search_kwargs={"k": knowledge_base.index.ntotal})
        docs = retriever.get_relevant_documents(user_query)  # Поиск релевантных документов

        # ===контекст из найденных документов
        context = "\n\n".join([doc.page_content for doc in docs])

        # === история диалога из памяти
        chat_history = memory.load_memory_variables({})["chat_history"]

        # ===промпт с контекстом историей и вопросом
        full_prompt = f"Historia del diálogo:\n{chat_history}\n\nDocumentos:\n{context}\n\nPregunta: {user_query}\nRespuesta:"

        # ===ChatOpenAI для генерации ответа
        response = llm.invoke(full_prompt)

        # === текущий вопрос и ответ в истории
        memory.save_context({"question": user_query}, {"answer": response.content})

        # ===ответ пользователю
        await message.answer(response.content, parse_mode="Markdown")
    except Exception as e:
        logging.error(f"Ошибка при обработке запроса: {e}")
        await message.answer("Ocurrió un error al procesar su solicitud. Por favor, inténtelo más tarde.")

# ===Запуск бота===
async def main():
    global knowledge_base
    logging.info("🚀 Загружаем базу недвижимости...")
    knowledge_base = load_embeddings()  # Пытаемся загрузить базу данных
    if knowledge_base is None:
        knowledge_base = await create_embeddings()  # Если базы нет, создаём её

    logging.info("🚀 Запускаем Telegram-бота...")
    dp.include_router(router)
    await dp.start_polling(bot)

if __name__ == "__main__":
    logging.info("🟢 Запускаем event loop...")
    asyncio.run(main())