from supabase import create_client, Client
from app.config import settings


def get_supabase_client() -> Client:
    """
    Returns a Supabase client authenticated with the service role key.
    This client bypasses Row Level Security and is intended for server-side
    use only. Never expose the service role key to the frontend.
    """
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY)


def get_supabase_anon_client() -> Client:
    """
    Returns a Supabase client authenticated with the anon (public) key.
    This client is used specifically for verifying user-supplied JWTs via
    auth.get_user(token), which is Supabase's intended pattern for backend
    token verification.  It should NOT be used for privileged data queries.
    """
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_ANON_KEY)
