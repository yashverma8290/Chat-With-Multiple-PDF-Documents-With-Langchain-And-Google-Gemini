-----------------------------------------------------------------------------------------
💬 Chat With Multiple PDF Documents using Langchain and Google Gemini Pro
-----------------------------------------------------------------------------------------
This project is a Streamlit-based AI chatbot that lets you upload multiple PDF files and ask questions based on their content using Google Gemini Pro (via LangChain). It uses:
- GoogleGenerativeAIEmbeddings for semantic search
- FAISS for local vector storage
- LangChain for the QA pipeline
- PyPDF2 for reading PDF text
- Streamlit for the user interface

-----------------------------------------------
🚀 Features
-----------------------------------------------
- Upload one or more PDF files
- Extracts and chunks text using LangChain
- Creates a local FAISS vector store with Gemini-powered embeddings
- Enables intelligent Q&A using gemini-pro
- User-friendly browser interface using Streamlit

-----------------------------------------------
📁 Project Structure
-----------------------------------------------
Chat-With-Multiple-PDF-Documents-With-Langchain-And-Google-Gemini/
│
├── code.py                     # Main Streamlit app
├── gemini_key.json             # Your Google Cloud service account key (not committed)
├── faiss_index/                # Folder generated after processing PDFs
├── requirements.txt            # Python dependencies
└── README.txt                  # This file

-----------------------------------------------
⚙️ Setup Instructions
-----------------------------------------------

1. 📥 Clone the repository

git clone https://github.com/yashverma8290/Chat-With-Multiple-PDF-Documents-With-Langchain-And-Google-Gemini.git
cd Chat-With-Multiple-PDF-Documents-With-Langchain-And-Google-Gemini

2. 🐍 Create and activate a virtual environment

python -m venv venv
venv\Scripts\activate        # On Windows
# OR
source venv/bin/activate     # On Linux/Mac

3. 📦 Install dependencies

pip install -r requirements.txt

-----------------------------------------------
🔐 Google Gemini API Setup
-----------------------------------------------

✅ Option 1: Using Service Account JSON

1. Go to Google Cloud Console → Enable "Generative Language API"
2. Create a Service Account with permission: Generative Language User
3. Download the JSON key and save it as gemini_key.json in the project root
4. Make sure your file path is set correctly in code.py:

GOOGLE_CREDENTIALS_PATH = r"C:\path\to\gemini_key.json"

-----------------------------------------------
🧠 How It Works
-----------------------------------------------

1. Upload PDF(s)
2. Extract text using PyPDF2
3. Split large text into chunks using RecursiveCharacterTextSplitter
4. Convert chunks into semantic embeddings using GoogleGenerativeAIEmbeddings
5. Store them locally in FAISS vector DB
6. When a question is asked:
   - Embeddings are used to find relevant text chunks
   - Gemini Pro answers the question using that context

-----------------------------------------------
▶️ Run the App
-----------------------------------------------

streamlit run code.py

Then visit http://localhost:8501 in your browser.

-----------------------------------------------
🛠️ Notes
-----------------------------------------------

- Make sure your gemini_key.json is never committed to GitHub.
- You only need to reprocess PDFs when they change.
- The FAISS vector DB is saved as faiss_index/ — auto loaded if it exists.

-----------------------------------------------

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/119111a5-ae2c-4700-a377-afb532f3f195" />

-----------------------------------------------
🧾 Requirements
-----------------------------------------------

- Python 3.8+
- Streamlit
- LangChain
- Google Generative AI SDK
- PyPDF2
- FAISS

-----------------------------------------------
🧑‍💻 Author
-----------------------------------------------

Yash Verma - https://github.com/yashverma8290

-----------------------------------------------
📜 License
-----------------------------------------------

MIT License — free to use and modify.

-----------------------------------------------
🤠 (For Submission Feel)
-----------------------------------------------

Isme Gemini Pro ka API use hua hai, direct Google Cloud se connect karke.
(Bhai yeh backend se powerful model ko access kar raha hai, iska jawaab best context se generate hota hai, default se nahi).
Baaki sab kaam humne code se kar liya, bus ek FAISS index banta hai jisme PDF ka pura gyaan hota hai. 😎
