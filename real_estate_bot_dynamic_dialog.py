
import os
import PyPDF2
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from openai import OpenAI


# Step 1: Extract information from PDF
def extract_pdf_data_and_images(folder_path):
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            text = ""

            # Extract text
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text()

            data.append({"text": text})
    return data

# Step 2: Load data into RAG
class RealEstateRAG:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            openai_api_key="sk-aitunnel-v66J7WhiyFEFEu2fQ32zLxgdh77VudJn",
            openai_api_base="https://api.aitunnel.ru/v1/",
            model="text-embedding-3-small"
        )
        self.db = None

    def load_data(self, documents):
        texts = [doc["text"] for doc in documents]
        self.db = FAISS.from_texts(texts, self.embeddings)

    def get_retriever(self):
        return self.db.as_retriever()

# Step 3: Build conversation model
class RealEstateAgent:
    def __init__(self, retriever):
        self.client = OpenAI(
            api_key="sk-aitunnel-v66J7WhiyFEFEu2fQ32zLxgdh77VudJn",
            base_url="https://api.aitunnel.ru/v1/"
        )
        self.retriever = retriever

    def get_response(self, query, context=None):
        # Формируем динамический промпт для живого диалога
        prompt = f"""
        Ты — умный и дружелюбный агент по недвижимости.
        Клиент ищет дом и задал вопрос: "{query}".

        Если есть контекст, вот информация о доме, которую нужно использовать:
        {context if context else 'Информация о доме отсутствует.'}

        Ответь клиенту живым, дружелюбным тоном. 
        Если клиенту не подходит дом, предложи альтернативный вариант.
        """

        try:
            completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": prompt}
                ],
                max_tokens=1000,
                model="gpt-4o"
            )
            response = completion.choices[0].message.content
        except Exception as e:
            response = f"Произошла ошибка при генерации ответа: {str(e)}"

        return response.strip()

# Telegram bot setup
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_query = update.message.text
    agent = context.application.bot_data['agent']

    # Если нет контекста, создаём его
    if 'session' not in context.user_data:
        context.user_data['session'] = {'current_house': None, 'houses': []}

    session = context.user_data['session']

    # Если это первый запрос, ищем релевантные дома
    if not session['houses']:
        retriever = agent.retriever
        session['houses'] = retriever.get_relevant_documents(user_query)

    # Используем текущий дом или переключаемся на следующий
    if not session['houses']:
        await update.message.reply_text("Извините, я не смог найти подходящих домов.")
        return

    current_house = session['houses'].pop(0)
    session['current_house'] = current_house

    # Генерируем ответ на основе текущего дома и запроса клиента
    context_for_gpt = current_house.page_content if current_house else None
    response = agent.get_response(user_query, context=context_for_gpt)
    await update.message.reply_text(response)

def main():
    # Load and process PDF files
    folder_path = "C:\\Users\\jolypab\\Documents\\CODE\RIELT\\PDF"
    pdf_data = extract_pdf_data_and_images(folder_path)

    # Initialize RAG and agent
    rag = RealEstateRAG()
    rag.load_data(pdf_data)

    agent = RealEstateAgent(retriever=rag.get_retriever())

    # Setup Telegram bot
    application = Application.builder().token("8041779345:AAE41dOCSlRCCjL63L-QaS9IghHE9XvQkp0").build()
    application.bot_data['agent'] = agent

    application.add_handler(CommandHandler("start", lambda update, context: update.message.reply_text("Привет! Чем могу помочь?")))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Bot is running...")
    application.run_polling()

if __name__ == "__main__":
    main()
