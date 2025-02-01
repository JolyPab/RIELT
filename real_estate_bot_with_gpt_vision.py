import os
import PyPDF2
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from google.cloud import vision
from openai import OpenAI
import fitz  # PyMuPDF
import io
from PIL import Image
import re

# Step 1: Extract information and images from PDF
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

            # Extract images
            images = extract_images_from_pdf(file_path)
            print(f"Файл {filename}: извлечено {len(images)} изображений")
            data.append({"text": text, "images": images, "filename": filename})
    return data

# Extract images from PDF (helper function)
def extract_images_from_pdf(pdf_path):
    images = []
    with fitz.open(pdf_path) as pdf:
        for page in pdf:
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = pdf.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                images.append((image_bytes, image_ext))
    return images

# Step 2: Load data into RAG
class RealEstateRAG:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            openai_api_key="sk-aitunnel-v66J7WhiyFEFEu2fQ32zLxgdh77VudJn",
            openai_api_base="https://api.aitunnel.ru/v1/",
            model="text-embedding-3-small"
        )
        self.db = None
        self.data = []

    def load_data(self, documents):
        self.data = documents
        texts = [doc["text"] for doc in documents]
        self.db = FAISS.from_texts(texts, self.embeddings)

    def filter_results(self, preferences, exclude_list=[]):
        query = f"{preferences.get('бюджет', '')} {preferences.get('спальни', '')} {preferences.get('бассейн', '')} {preferences.get('тип', '')}"
        results = self.db.similarity_search(query)
        
        if not results:
            return None
        
        budget = int(re.sub("[^0-9]", "", preferences.get('бюджет', '0')))
        property_type = preferences.get('тип', '').lower()
        best_match = None
        min_price_diff = float('inf')
        
        for doc in self.data:
            if doc['filename'] in exclude_list:
                continue  # Пропускаем уже предложенные дома

            if results[0].page_content in doc["text"]:
                price_match = re.search(r'\b([0-9]+[,\.]?[0-9]*)\s*MXN\b', doc["text"])
                price = int(price_match.group(1).replace(',', '')) if price_match else 0
                
                if price <= budget and (property_type in doc["text"].lower() or not property_type):
                    return doc
                
                if price > budget and price - budget < min_price_diff:
                    best_match = doc
                    min_price_diff = price - budget
        
        return best_match if best_match else None

# Step 3: Build conversation model
class RealEstateAgent:
    def __init__(self, retriever):
        self.client = OpenAI(
            api_key="sk-aitunnel-v66J7WhiyFEFEu2fQ32zLxgdh77VudJn",
            base_url="https://api.aitunnel.ru/v1/"
        )
        self.retriever = retriever
        self.questions = [
            "Какой у вас бюджет на покупку недвижимости?",
            "Сколько спален вам необходимо?",
            "Обязателен ли бассейн?",
            "В каком районе вы ищете жилье?",
            "Есть ли дополнительные важные критерии? (например, гараж, терраса)",
            "Какой тип недвижимости вас интересует? (дом, квартира, таунхаус и т.д.)"
        ]

    def ask_questions(self, step):
        if step < len(self.questions):
            return self.questions[step]
        return None

    def generate_response(self, property_info):
        prompt = f"""
        Ты — опытный агент по недвижимости. На основе информации о доме составь краткое, дружелюбное описание для клиента.
        Вот данные объекта:
        {property_info}
        Описание должно быть естественным, словно ты рассказываешь клиенту.
        """
        
        try:
            completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": prompt}
                ],
                max_tokens=500,
                model="gpt-4o"
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            return f"Ошибка генерации описания: {str(e)}"

# Telegram bot setup
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_query = update.message.text
    agent = context.application.bot_data['agent']
    
    if 'session' not in context.user_data:
        context.user_data['session'] = {'step': 0, 'preferences': {}, 'exclude_list': []}

    session = context.user_data['session']
    step = session['step']
    question = agent.ask_questions(step)

    if question:
        if step > 0:
            last_question = agent.questions[step - 1]
            session['preferences'][last_question] = user_query
        session['step'] += 1
        await update.message.reply_text(question)
    else:
        preferences = session['preferences']
        best_match = agent.retriever.filter_results(preferences, session['exclude_list'])
        
        if best_match:
            session['exclude_list'].append(best_match['filename'])
            description = agent.generate_response(best_match["text"])
            await update.message.reply_text(description)
            if best_match["images"]:
                image_bytes, ext = best_match["images"][0]
                compressed_image = compressed_image(image_bytes)
                await update.message.reply_photo(photo=compressed_image)
        else:
            await update.message.reply_text("К сожалению, подходящих вариантов не найдено.")
        

def main():
    folder_path = "C:\\Users\\jolypab\\Documents\\CODE\\RIELT\\PDF"
    pdf_data = extract_pdf_data_and_images(folder_path)
    
    rag = RealEstateRAG()
    rag.load_data(pdf_data)
    agent = RealEstateAgent(retriever=rag)

    application = Application.builder().token("8041779345:AAE41dOCSlRCCjL63L-QaS9IghHE9XvQkp0").build()
    application.bot_data['agent'] = agent

    application.add_handler(CommandHandler("start", lambda update, context: update.message.reply_text("Привет! Чем могу помочь?")))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    print("Bot is running...")
    application.run_polling()

if __name__ == "__main__":
    main()

