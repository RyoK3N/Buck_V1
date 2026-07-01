"""
UI.backend.auth
─────────────────
Optional bearer-token auth for the FastAPI backend.

The backend has no auth by default — it's meant for localhost development,
matching the rest of the project's "trusted machine" trust model (see
SECURITY.md). Set BUCK_API_AUTH_TOKEN in .env to require every request to
carry ``Authorization: Bearer <token>`` before it reaches a route handler.
This is opt-in: deployments that expose the API beyond localhost (e.g.
behind a reverse proxy) should set it; the default local workflow is
unaffected.
"""

from __future__ import annotations

import hmac

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

# Health checks never need a token.
_EXEMPT_PATHS = {"/health"}

# NOTE: Starlette's BaseHTTPMiddleware only wraps HTTP request/response
# scopes — it does not run for WebSocket connections. /accuracy/ws is
# therefore NOT covered by this middleware even when a token is configured.
# If you expose the API beyond localhost, put it behind a reverse proxy that
# also gates the WebSocket route (or disable that endpoint).


class BearerAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, token: str):
        super().__init__(app)
        self._token = token

    async def dispatch(self, request: Request, call_next):
        if request.method == "OPTIONS" or request.url.path in _EXEMPT_PATHS:
            return await call_next(request)

        header = request.headers.get("authorization", "")
        scheme, _, supplied = header.partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(supplied, self._token):
            return JSONResponse(status_code=401, content={"detail": "Unauthorized"})

        return await call_next(request)


def maybe_add_auth(app, token: str | None) -> None:
    """Attach BearerAuthMiddleware iff a token is configured."""
    if token:
        app.add_middleware(BearerAuthMiddleware, token=token)
