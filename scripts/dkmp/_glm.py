"""Shared GLM-4.7 client for DKMP pipeline.

Reads GLM_API_KEY and GLM_URL from .env.local if not in environment.
Default endpoint: dmxapi.cn (the .com endpoint is unreachable from this machine,
despite memory note suggesting otherwise).

Usage:
    from _glm import call_glm_async, configure
    configure()  # loads .env.local
    text, err = await call_glm_async(client, prompt, max_tokens=120)
"""
from __future__ import annotations
import asyncio
import json
import os
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[2]
DEFAULT_URL = "https://www.dmxapi.cn/v1/chat/completions"
MODEL = os.environ.get("GLM_MODEL", "glm-4.7")


def configure() -> None:
    """Load .env.local into os.environ (idempotent)."""
    env_file = REPO / ".env.local"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k, v)


def _api():
    return os.environ.get("GLM_API_KEY", "").strip()


def _url():
    return os.environ.get("GLM_URL", DEFAULT_URL).strip()


async def call_glm_async(
    client: httpx.AsyncClient,
    prompt: str,
    max_tokens: int = 120,
    temperature: float = 0.0,
    n_retries: int = 4,
    timeout_s: float = 300.0,
) -> tuple[str, str | None]:
    """Returns (content, error). content="" on failure; error=None on success."""
    err = "max_retries"
    for attempt in range(n_retries):
        try:
            r = await client.post(
                _url(),
                headers={
                    "Authorization": f"Bearer {_api()}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "enable_thinking": False,
                },
                timeout=httpx.Timeout(timeout_s, connect=30.0),
            )
            if r.status_code in (429,) or r.status_code >= 500:
                await asyncio.sleep(min(60.0, 2.0 ** (attempt + 1)))
                continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip(), None
        except Exception as e:
            err = str(e)[:200]
            await asyncio.sleep(min(60.0, 2.0 ** (attempt + 1)))
    return "", err


def make_client(timeout_s: float = 300.0) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=httpx.Timeout(timeout_s, connect=30.0),
        limits=httpx.Limits(max_connections=64, max_keepalive_connections=32),
    )
