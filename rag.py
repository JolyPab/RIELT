import os
import PyPDF2
from io import BytesIO
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
            model="text-embedding-3-small"  # Updated to use a cheaper model
        )
        self.db = None
        self.data = []

    def load_data(self, documents):
        texts = [doc["text"] for doc in documents]
        self.data = documents
        self.db = FAISS.from_texts(texts, self.embeddings)

    def get_retriever(self):
        return self.db.as_retriever()


# Step 3: Build conversation model
class RealEstateAgent:
    def __init__(self, retriever, rag):
        self.client = OpenAI(
            api_key="sk-aitunnel-v66J7WhiyFEFEu2fQ32zLxgdh77VudJn",
            base_url="https://api.aitunnel.ru/v1/"
        )
        self.retriever = retriever
        self.rag = rag

    def get_response(self, query):
        retrieved_docs = self.retriever.get_relevant_documents(query)[:1]
        context = "\n".join([doc.page_content for doc in retrieved_docs])

        prompt = f"""
        представь что ты — умный агент по недвижимости. Клиент описал свои пожелания: "{query}".

        Твои задачи:
        1. кратко описать релевантый дом
        2. в формате чата(диалога) общаться с клиентом
        3. отвечай на вопросы клиента о доме
        4. если клиенту не понравился дом, предложи другой релевантный вариант.
        """

        # Генерация ответа
        completion = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": prompt}
            ],
            max_tokens=1000,
            model="gpt-4o"
        )
        response = completion.choices[0].message.content
        response = response.replace("*", "")


# Telegram bot setup
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Я ваш агент по недвижимости. Спросите меня о доме, который вас интересует.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_query = update.message.text
    agent = context.application.bot_data['agent']

    # Generate response
    response = agent.get_response(user_query)

    # Send text response
    await update.message.reply_text(response)

def main():
    # Example: Load and process PDF files from a folder
    folder_path = "G:\\CODE\\RIELT\\PDF"
    pdf_data = extract_pdf_data_and_images(folder_path)

    # Initialize RAG
    rag = RealEstateRAG()
    rag.load_data(pdf_data)

    # Create agent
    agent = RealEstateAgent(retriever=rag.get_retriever(), rag=rag)

    # Setup Telegram bot
    application = Application.builder().token("8041779345:AAE41dOCSlRCCjL63L-QaS9IghHE9XvQkp0").build()

    application.bot_data['agent'] = agent
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Bot is running...")
    application.run_polling()

if __name__ == "__main__":
    main()

