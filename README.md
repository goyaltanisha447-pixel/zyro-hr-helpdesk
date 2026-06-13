# zyro-hr-helpdesk
An AI-powered HR chatbot built with a RAG (Retrieval-Augmented Generation) pipeline. Employees can ask questions about leave, payroll, benefits, and company policies — and get accurate, grounded answers from internal HR documents.


📸 Demo


Ask questions like:


"How many annual leave days do I get?"
"What is the maternity leave policy?"
"How do I claim medical reimbursement?"





🏗️ Architecture

HR Policy Docs (PDF / TXT)
         │
         ▼
   Document Loader
         │
         ▼
   Text Chunker          ← 400-word sliding window, 80-word overlap
         │
         ▼
   Embedder              ← all-MiniLM-L6-v2 (384-dim, normalized)
         │
         ▼
   FAISS Vector Index    ← Inner-product search (cosine similarity)
         │
    Query time:
         │
   User Question
         │
         ▼
   Retrieve top-5 chunks
         │
   Similarity < 0.35? ──► Out-of-scope reply
         │
         ▼
   Claude LLM            ← Grounded generation with strict system prompt
         │
         ▼
       Answer + Sources


🚀 Quick Start

1. Clone the repo

bashgit clone https://github.com/your-username/zyro-hr-helpdesk.git
cd zyro-hr-helpdesk

2. Install dependencies

bashpip install -r requirements.txt

3. Add your HR documents

Create a folder called hr_docs/ and place your PDF or TXT policy files inside:

hr_docs/
  leave_policy.pdf
  payroll_policy.pdf
  benefits_handbook.pdf
  code_of_conduct.txt

4. Set your API key

Create a .env file in the root folder:

ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxx

Get your key from → console.anthropic.com

5. Run the app

bashstreamlit run app.py

Open your browser at http://localhost:8501


📁 Project Structure

zyro-hr-helpdesk/
│
├── app.py                  # Streamlit chatbot app
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── .env                    # Your API key (do NOT commit this)
├── .gitignore              # Ignores .env and hr_docs/
└── hr_docs/                # Your HR policy documents (PDF/TXT)


⚙️ Configuration

All settings are in the CFG dictionary inside app.py:

SettingDefaultDescriptiondata_dir./hr_docsFolder containing HR documentsembed_modelall-MiniLM-L6-v2Sentence embedding modelchunk_size400Words per chunkchunk_overlap80Overlapping words between chunkstop_k5Number of chunks retrieved per queryllm_modelclaude-sonnet-4-20250514Claude model used for generationout_of_scope_threshold0.35Min similarity score to answer


🛡️ Guardrails

GuardrailHow it worksSimilarity thresholdQueries below 0.35 cosine similarity get an out-of-scope reply — no LLM call madeStrict system promptClaude is instructed to answer only from provided context, never hallucinateSource citationsEvery answer shows which HR document it came from


🔧 Tuning Tips


Not finding answers? Lower out_of_scope_threshold or increase top_k
Hallucinating? Raise out_of_scope_threshold or reduce chunk_size
Slow startup? Switch to paraphrase-MiniLM-L3-v2 for faster embedding
Better accuracy? Try BAAI/bge-base-en-v1.5 as the embed model



📦 Dependencies

PackagePurposestreamlitWeb UI frameworkanthropicClaude LLM APIfaiss-cpuVector similarity searchsentence-transformersText embedding modelPyPDF2PDF text extractionpython-dotenvLoad .env API key
