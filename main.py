import os
import re
import json
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from openai import OpenAI


# ======================
# KONFIGURACE
# ======================
BLOG_INDEX = "https://www.dateioplatform.com/resources/blog"
STATE_FILE = "state.json"


# ======================
# ENV helper (blbě-odolný)
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
# STATE (už zpracované URL)
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
# HTTP HELPERS
# ======================
def fetch_html(url: str) -> str:
    r = requests.get(
        url,
        timeout=30,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    r.raise_for_status()
    return r.text


# ======================
# BLOG INDEX → URL článků
# ======================
def extract_post_urls_from_index(index_html: str) -> list[str]:
    soup = BeautifulSoup(index_html, "html.parser")
    urls = set()

    for a in soup.select("a[href]"):
        href = a.get("href", "")
        if "/resources/post/" in href:
            urls.add(urljoin(BLOG_INDEX, href))

    return sorted(urls)


# ======================
# DETAIL ČLÁNKU → TEXT
# ======================
def extract_article_text(article_html: str, article_url: str) -> dict:
    soup = BeautifulSoup(article_html, "html.parser")

    title = "Untitled"
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        title = h1.get_text(strip=True)
    elif soup.title and soup.title.get_text(strip=True):
        title = soup.title.get_text(strip=True)

    paragraphs = [
        p.get_text(" ", strip=True)
        for p in soup.find_all("p")
        if p.get_text(strip=True)
    ]
    body = "\n\n".join(paragraphs).strip()

    return {
        "url": article_url,
        "title": title,
        "body": body,
    }


# ======================
# OPENAI – PŘEKLAD
# ======================
def translate_to_hungarian(client: OpenAI, title: str, body: str) -> str:
    prompt = f"""
Přelož následující článek do maďarštiny.

Pravidla:
- zachovej význam i marketingový tón
- nezkracuj
- zachovej odstavce
- vrať výstup v Markdownu

# Název
{title}

# Text
{body}
""".strip()

    response = client.responses.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        input=prompt,
    )

    return response.output_text.strip()


# ======================
# MS TEAMS – Webhook
# ======================
def send_message_to_teams(title: str, article_url: str, translation_md: str) -> None:
    webhook_url = require_env("TEAMS_WEBHOOK_URL").strip()

    # Teams má limit délky zprávy – zkrátíme
    max_chars = 3500
    snippet = translation_md
    if len(snippet) > max_chars:
        snippet = snippet[:max_chars] + "\n\n…(zkráceno)"

    payload = {
        "text": (
            "📄 **Nový blog přeložen do maďarštiny**\n\n"
            f"**Název:** {title}\n\n"
            f"🔗 **Originál:** {article_url}\n\n"
            "---\n\n"
            f"{snippet}"
        )
    }

    r = requests.post(
        webhook_url,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=30,
    )

    if r.status_code >= 400:
        print("TEAMS WEBHOOK ERROR")
        print("Status code:", r.status_code)
        print("Response text:", r.text)

    r.raise_for_status()


# ======================
# MAIN
# ======================
def main():
    # povinné proměnné (fail fast)
    require_env("OPENAI_API_KEY")
    require_env("TEAMS_WEBHOOK_URL")

    processed = load_state()

    index_html = fetch_html(BLOG_INDEX)
    post_urls = extract_post_urls_from_index(index_html)

    new_urls = [u for u in post_urls if u not in processed]

    if not new_urls:
        print("Žádné nové články.")
        save_state(processed)
        return

    client = OpenAI(api_key=require_env("OPENAI_API_KEY"))

    for url in new_urls:
        article_html = fetch_html(url)
        article = extract_article_text(article_html, url)

        if not article["body"]:
            print(f"Nelze extrahovat text: {url}")
            processed.add(url)
            continue

        translation = translate_to_hungarian(
            client, article["title"], article["body"]
        )

        slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", url.split("/")[-1]).strip("-")

        send_message_to_teams(
            title=article["title"],
            article_url=url,
            translation_md=translation,
        )

        print(f"Odesláno do Teams: {slug}")
        processed.add(url)

    save_state(processed)


if __name__ == "__main__":
    main()
