# MUNSIF — Backend

FastAPI + Supabase backend powering **MUNSIF**, an independent, AI-powered assistant that helps everyday bank customers in Pakistan understand their consumer-protection rights — complaint procedures, Banking Mohtasib escalation, credit bureau (eCIB) questions, and more.

> MUNSIF is an independent, unofficial educational project and is not affiliated with, endorsed by, or operated by the State Bank of Pakistan.

Frontend repository: [munsif-frontend](https://github.com/AliHusnainSaqib/munsif-frontend)

## Features
- Deterministic, retrieval-based chatbot — answers are matched from a curated knowledge base, never freely generated, to avoid hallucinated regulatory advice
- JWT-based authentication via Supabase Auth, with mandatory email confirmation
- Optional TOTP two-factor authentication for all users, mandatory for admin accounts
- Persistent multi-session chat history
- Admin dashboard for reviewing unanswered queries and expanding the knowledge base
- A super-admin tier for managing other admins' access
- Identity verification workflow (CNIC/B-Form) with admin-reviewed change requests

## Tech Stack
- **Framework**: FastAPI (Python), Uvicorn
- **Database**: PostgreSQL via Supabase, with the pgvector extension for similarity search
- **Auth**: Supabase Auth
- **NLP**: sentence-transformers (`paraphrase-multilingual-MiniLM-L12-v2`)

## Local Setup
1. Clone this repository
2. Create a virtual environment: `python -m venv .venv`
3. Activate it: `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (Mac/Linux)
4. Install dependencies: `pip install -r requirements.txt`
5. Copy `.env.example` to `.env` and fill in your own Supabase project credentials
6. Run: `uvicorn app.main:app --reload`

## License
MIT — see [LICENSE](./LICENSE)
