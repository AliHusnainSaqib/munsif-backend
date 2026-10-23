from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.routers import admin, auth_test, chats
from app.routers import auth_public

# ---------------------------------------------------------------------------
# Model name — must match the one used during ingestion
# ---------------------------------------------------------------------------
_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


# ---------------------------------------------------------------------------
# Lifespan: load model once at startup, release on shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[startup] Loading SentenceTransformer model '{_MODEL_NAME}' …")
    app.state.model = SentenceTransformer(_MODEL_NAME)
    print("[startup] Model ready.")
    yield
    # Nothing to clean up for SentenceTransformer, but the hook is here
    # in case resources need releasing in future.
    print("[shutdown] Shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="SBP Chatbot API", lifespan=lifespan)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
origins = [
    "http://localhost:3000",   # Next.js / CRA default port
    "http://localhost:5173",   # Vite default port
    "http://localhost:5174",   # Vite alternative port
    "http://localhost:5175",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
    "http://127.0.0.1:5175",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Rate limiter (slowapi) — required by auth_public routes
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(chats.router)
app.include_router(admin.router)
app.include_router(auth_test.router)
app.include_router(auth_public.router)

# ---------------------------------------------------------------------------
# Core routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["health"])
def health_check():
    """Returns a simple liveness probe response."""
    return {"status": "ok"}
