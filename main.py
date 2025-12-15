import os
import re
import json
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from openai import OpenAI


BLOG_INDEX = "https://www.dateioplatform.com/resources/blog"
STATE_FILE = "state.json"


# ======================
# ENV helper
# ======================
def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Chybí proměnná prostředí {name}. "
            f"V GitHub Actions musí být v jobu:\n"
            f"{name}: ${{{{ secrets.{name} }}}}"
        )
    return value


# ======================
# STATE
# ======================
def load_state() -> set[str]:
    if not os.path.exists(STATE_FILE):
        return set()
    with open(STATE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return set(data.get("processed_urls", []))


def save_state(processed_urls: set[str]) -> None:
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {"processed_urls": sorted(processed_urls)},
            f,
            ensure_ascii=False,
            indent=2,
        )


# ======================
# HTTP
# =================
