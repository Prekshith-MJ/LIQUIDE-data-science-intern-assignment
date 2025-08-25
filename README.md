

# LIQUIDE Data Science Intern Assignment

This project implements a hybrid research agent capable of retrieving answers from ingested PDFs and falling back to web search when required. The agent is powered by LangChain, FAISS, and Claude-3-Haiku (via OpenRouter).  

---

## 📂 Project Structure

LIQUIDE-data-science-intern-assignment/

├── Data.pdf             # Sample PDF for testing

├── app.py               # Streamlit application entry point

├── ingest.py            # PDF parsing, cleaning, chunking, and embedding

├── agent.py             # Query agent logic (vectorstore + web fallback)

├── vectorstore.py       # FAISS vector store search

├── websearch.py         # DuckDuckGo search integration

├── testagent.py         # Test script for agent functionality

├── testopenrouter.py    # Test script for OpenRouter API integration

├── tools.py             # Utility functions

├── requirements.txt     # Dependencies

├── faiss_index/         # Folder containing FAISS vectorstore

└── readme.md            # Project documentation

---

## 🔄 System Flow

User Query

│
▼

 Query Cleaning (agent.py)
 │
 ├──► Vector Search (FAISS over PDF embeddings)
 │       │
 │       └──► Relevant Document Passages

 │
 └──► Web Search (DuckDuckGo) [fallback]

  │
  └──► External Results
  │
  ▼
  Claude-3-Haiku (OpenRouter)
                    
                                  │
                                  ▼
                                  Final Response

## ⚙️ Installation

1. Clone the repository:

```bash
git clone https://github.com/Prekshith-MJ/LIQUIDE-data-science-intern-assignment.git
cd LIQUIDE-data-science-intern-assignment

	2.	Create a virtual environment and install dependencies:

python3 -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows

pip install -r requirements.txt


⸻

🔑 API Key Setup

### 🔑 API Key Setup

This project requires an OpenRouter API key to access Claude-3-Haiku.  
You can get your API key by signing up at [OpenRouter](https://openrouter.ai).  

Ensure your key is set before running the app:

```bash
export OPENROUTER_API_KEY="your_key_here"   # Mac/Linux
setx OPENROUTER_API_KEY "your_key_here"     # Windows


⸻

🚀 Usage
	1.	First, ingest your PDFs into the vectorstore:

python ingest.py

	2.	Run the Streamlit app:

streamlit run app.py

	3.	Ask questions in the UI:

	•	The agent first searches the FAISS vectorstore.
	•	If no relevant match is found, it falls back to web search.
	•	Claude-3-Haiku generates the final response.

⸻

📈 Evaluation Criteria Mapping
	•	Data Preprocessing & Extraction → ingest.py (PDF parsing, cleaning, chunking)
	•	Feature Engineering → Embedding creation with Sentence Transformers (all-MiniLM-L6-v2, all-mpnet-base-v2)
	•	Feature Selection → Support for multiple embedding models
	•	Semantic Search → FAISS vector store for retrieval
	•	Tool Selection → Stack includes LangChain, FAISS, DuckDuckGo, OpenRouter
	•	Agent Implementation → Hybrid retrieval + query cleaning in agent.py
	•	LLM Implementation → Claude-3-Haiku response generation

 

