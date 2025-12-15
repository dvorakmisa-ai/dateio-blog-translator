import os
import re
import json
import hashlib
from datetime import datetime
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from openai import OpenAI

BLOG_INDEX = "https://www.dateioplatform.com/resources/blog"
STATE_FILE = "state.json"

def load_state() -> set[str]:
    if not os.path.exists(STATE_FILE):
        return set()
    with open(STATE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return set(data.get("processed_urls", []))

def save_state(processed_urls: set[str]) -> None:
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump({"processed_urls": sorted(processed_urls)}, f, ensure_ascii=False, indent=2)

def fetch_html(url: str) -> str:
    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return r.text

def extract_post_urls_from_index(index_html: str) -> list[str]:
    soup = BeautifulSoup(index_html, "html.parser")
    urls = set()
    for a in soup.select("a[href]"):
        href = a.get("href", "")
        if "/resources/post/" in href:
            urls.add(urljoin(BLOG_INDEX, href))
    # většinou stačí první stránka; vezmeme všechny nalezené
    return sorted(urls)

def extract_article_text(article_html: str, article_url: str) -> dict:
    """
    Jednoduchá extrakce:
    - title: <title> nebo první <h1>
    - body: text z hlavní části (fallback: všechny <p>)
    """
    soup = BeautifulSoup(article_html, "html.parser")

    # title
    title = None
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        title = h1.get_text(strip=True)
    if not title and soup.title and soup.title.get_text(strip=True):
        title = soup.title.get_text(strip=True)
    if not title:
        title = "Untitled"

    # pokus o „nejpravděpodobnější“ obsah: všechny odstavce
    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    body = "\n\n".join([p for p in paragraphs if p])

    if len(body) < 200:
        # fallback – někdy je text v div/span; vezmeme celý viditelný text a zredukujeme
        text = soup.get_text("\n", strip=True)
        body = "\n\n".join([line for line in text.split("\n") if len(line) > 40])

    body = body.strip()

    return {
        "url": article_url,
        "title": title,
        "body": body
    }

def translate_to_hungarian(openai_client: OpenAI, title: str, body: str) -> str:
    # Pokud je článek dlouhý, může být potřeba dělení na části. Pro začátek to necháme jednoduché.
    prompt = f"""
Přelož následující článek do maďarštiny.
- Zachovej smysl, tón a marketingovou terminologii.
- Zachovej odstavce.
- Nezkracuj.
- Vrať výstup v Markdownu.

# Název
{title}

# Text
{body}
""".strip()

    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

    resp = openai_client.responses.create(
        model=model,
        input=prompt,
    )

    # SDK vrací text v resp.output_text
    return resp.output_text.strip()

def jira_auth_headers() -> dict:
    # Jira Cloud: Basic Auth s email:api_token
    # requests to uděláme přes HTTPBasicAuth, ale hlavičky pro JSON dáme sem
    return {"Accept": "application/json", "Content-Type": "application/json"}

def jira_create_issue(summary: str, description_text: str) -> str:
    """
    Vytvoří issue a vrátí issue key (např. ABC-123).
    Použijeme v3 endpoint /rest/api/3/issue. :contentReference[oaicite:4]{index=4}
    Pozn.: description ve v3 je ADF; pro jednoduchost ji dáme jen jako plain text přes ADF minimum.
    """
    base_url = os.environ["JIRA_BASE_URL"].rstrip("/")
    email = os.environ["JIRA_EMAIL"]
    token = os.environ["JIRA_API_TOKEN"]
    project_key = os.environ["JIRA_PROJECT_KEY"]

    url = f"{base_url}/rest/api/3/issue"

    # ADF "doc"
    adf_description = {
        "type": "doc",
        "version": 1,
        "content": [{
            "type": "paragraph",
            "content": [{"type": "text", "text": description_text}]
        }]
    }

    payload = {
        "fields": {
            "project": {"key": project_key},
            "summary": summary,
            "issuetype": {"name": "Task"},
            "description": adf_description,
            "labels": ["dateio-auto-translate"]
        }
    }

    r = requests.post(url, headers=jira_auth_headers(), auth=(email, token), json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data["key"]

def jira_attach_file(issue_key: str, filename: str, content_bytes: bytes) -> None:
    """
    Přidá attachment k issue.
    Jira vyžaduje header X-Atlassian-Token: nocheck a multipart/form-data. :contentReference[oaicite:5]{index=5}
    """
    base_url = os.environ["JIRA_BASE_URL"].rstrip("/")
    email = os.environ["JIRA_EMAIL"]
    token = os.environ["JIRA_API_TOKEN"]

    url = f"{base_url}/rest/api/3/issue/{issue_key}/attachments"

    headers = {
        "X-Atlassian-Token": "nocheck",
        "Accept": "application/json",
    }

    files = {
        "file": (filename, content_bytes, "text/markdown; charset=utf-8")
    }

    r = requests.post(url, headers=headers, auth=(email, token), files=files, timeout=60)
    r.raise_for_status()

def jira_already_has_issue_for_url(article_url: str) -> bool:
    """
    Aby se nevytvářely duplicitní úkoly:
    zkusíme najít issue s labelem a URL v textu.
    (JQL přes /rest/api/3/search) :contentReference[oaicite:6]{index=6}
    """
    base_url = os.environ["JIRA_BASE_URL"].rstrip("/")
    email = os.environ["JIRA_EMAIL"]
    token = os.environ["JIRA_API_TOKEN"]
    project_key = os.environ["JIRA_PROJECT_KEY"]

    jql = f'project = {project_key} AND labels = "dateio-auto-translate" AND text ~ "{article_url}"'
    url = f"{base_url}/rest/api/3/search"

    r = requests.get(
        url,
        headers={"Accept": "application/json"},
        auth=(email, token),
        params={"jql": jql, "maxResults": 1},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    return data.get("total", 0) > 0

def main():
    processed = load_state()

    index_html = fetch_html(BLOG_INDEX)
    post_urls = extract_post_urls_from_index(index_html)

    # vezmeme jen nové URL (oproti state.json)
    new_urls = [u for u in post_urls if u not in processed]

    if not new_urls:
        print("Žádné nové články.")
        return

    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    for url in new_urls:
        # ještě pojistka proti duplicitám přímo v Jira (kdyby se ztratil state.json)
        if jira_already_has_issue_for_url(url):
            print(f"V Jira už existuje issue pro: {url}")
            processed.add(url)
            continue

        article_html = fetch_html(url)
        article = extract_article_text(article_html, url)

        if not article["body"]:
            print(f"Nešlo vytáhnout text článku: {url}")
            processed.add(url)
            continue

        hu_md = translate_to_hungarian(openai_client, article["title"], article["body"])

        # vytvoříme soubor s překladem
        safe_slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", url.split("/")[-1]).strip("-")
        filename = f"{safe_slug}-hu.md"
        content = f"# {article['title']}\n\nOriginál: {url}\n\n---\n\n{hu_md}\n"
        content_bytes = content.encode("utf-8")

        summary = f"[HU] Překlad: {article['title']}"
        description = f"Překlad do maďarštiny.\n\nOriginál: {url}\n\nSoubor s překladem je v příloze (Markdown)."

        issue_key = jira_create_issue(summary, description)
        jira_attach_file(issue_key, filename, content_bytes)

        print(f"Hotovo: {url} → {issue_key}")
        processed.add(url)

    save_state(processed)

if __name__ == "__main__":
    main()
