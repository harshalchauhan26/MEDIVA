# MEDIVA - A Medical RAG Chatbot

**MEDIVA** (Medical Intelligent Virtual Assistant) is a lightweight and modern Retrieval-Augmented Generation (RAG) chatbot designed to assist with medical knowledge and inquiries using custom PDF documents. It leverages powerful embedding models and LLMs for accurate, document-grounded responses.

---

## 🔍 Features

- **RAG Pipeline**: Seamless integration of retrieval and generation.
- **Custom PDF Uploads**: Ask questions based on your own medical documents.
- **Fast Inference**: Powered by GROQ and Langchain for blazing-fast results.
- **Simple UI**: Built with Streamlit, keeping it elegant and responsive.
- **Context-Aware**: Understands medical questions within the scope of provided documents.

---

## 🧠 How It Works

1. **PDF Loader**: Upload medical PDFs (research, notes, articles).
2. **Text Splitter**: Chunks the document into meaningful segments.
3. **Embeddings**: Converts chunks into vector form using HuggingFace embeddings.
4. **Vector Store (FAISS)**: Stores and indexes embeddings for fast retrieval.
5. **Query**: Ask a medical question.
6. **LLM (GROQ)**: Retrieves context chunks and generates a natural language response.

---

## 🚀 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/mediva.git
cd mediva
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your API keys

Create a `.env` file:

```bash
GROQ_API_KEY=your_groq_api_key_here
```
*(Optional if you are using other keys like OPENAI or local LLMs)*

### 5. Run the Streamlit App

```bash
streamlit run mainbot.py
```

---

## 📁 Folder Structure

```
mediva/
├── mainbot.py              # Main Streamlit app
├── database.py             # Handles embeddings and FAISS vector store
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── .env                    # API keys and environment variables
```

---

## 🏥 Use Cases

- Medical research assistants
- Educational tools for students
- Reference chatbot for healthcare workers
- Custom document querying (journal articles, reports, books)

---

## 📌 Future Improvements

- UI refinement with responsive design
- Multi-PDF support
- Caching and optimization
- User authentication
- Advanced analytics and visual explanations

---

## 📫 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## ⚖️ License

This project is licensed under the MIT License.