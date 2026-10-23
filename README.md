# SBP Chatbot — FastAPI Backend

A minimal FastAPI scaffold for the SBP Chatbot backend.

---

## Prerequisites

- Python 3.10 or later
- Git

---

## Windows Setup

### 1. Create a virtual environment

```powershell
python -m venv .venv
```

### 2. Activate the virtual environment

```powershell
.venv\Scripts\activate
```

> Your prompt should now show `(.venv)` on the left.

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Configure environment variables

```powershell
copy .env.example .env
```

Open `.env` in any text editor and fill in the three values:

```
SUPABASE_URL=https://your-project-ref.supabase.co
SUPABASE_ANON_KEY=your-anon-key
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
```

> ⚠️ **Never commit `.env` to version control.** It is already listed in `.gitignore`.

### 5. Run the development server

```powershell
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`.  
Interactive docs (Swagger UI): `http://localhost:8000/docs`

---

## Project Structure

```
sbp-chatbot-backend/
  app/
    __init__.py
    main.py          # FastAPI app, CORS, routes
    config.py        # Pydantic-settings — reads from .env
    db.py            # Supabase client (service role, server-side only)
    routers/         # Route modules go here
    schemas/         # Pydantic request/response schemas go here
  .env.example       # Template — copy to .env and fill in values
  .gitignore
  requirements.txt
  README.md
```

---

## Health Check

```
GET /health
→ {"status": "ok"}
```
