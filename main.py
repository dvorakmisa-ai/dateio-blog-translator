import os
import re
import json
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from openai import OpenAI


BLOG_INDEX = "https://www.dateioplatform.com/resources/blog"
STATE_FILE = "state.json"


# ----------------------
# STATE (processed URLs)
# ----------------------
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


# ----------------------
# HTTP helper
# ----------------------
def fetch_html(url: str) -> str:
    r = requests.get(
        url,
        timeout=30,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    r.raise_for_status()
    return r.text


# ----------------------
# Blog index -> post URLs
# ----------------------
def extract_post_urls_from_index(index_html: str) -> list[str]:
    soup = BeautifulSoup(index_html, "html.parser")
    urls = set()

    for a in soup.select("a[href]"):
        href = a.get("href", "")
        if "/resources/post/" in href:
            urls.add(urljoin(BLOG_INDEX, href))

    return sorted(urls)


# ----------------------
# Article page -> text
# ----------------------
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


# ----------------------
# OpenAI translation
# ----------------------
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


# ----------------------
# Jira helpers
# ----------------------
def jira_get_valid_issue_type_name() -> str:
    """
    Zjistí, jaké issue typy jsou povolené pro daný projekt,
    a vrátí jeden validní název (preferuje Task/Úkol/Story/Bug).
    """
    base = os.environ["JIRA_BASE_URL"].rstrip("/")
    email = os.environ["JIRA_EMAIL"]
    token = os.environ["JIRA_API_TOKEN"]
    project = os.environ["JIRA_PROJECT_KEY"]

    url = f"{base}/rest/api/3/issue/createmeta"

    r = requests.get(
        url,
        auth=(email, token),
        headers={"Accept": "application/json"},
        params={"projectKeys": project, "expand": "projects.issuetypes"},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()

    projects = data.get("projects", [])
    if not projects:
        raise RuntimeError(
            "Jira createmeta nevrátil žádný projekt. Zkontroluj JIRA_PROJECT_KEY."
        )

    issuetypes = projects[0].get("issuetypes", [])
    if not issuetypes:
        raise RuntimeError(
            "Jira createmeta nevrátil žádné issue typy pro tento projekt."
        )

    available = [it.get("name") for it in issuetypes if it.get("name")]

    preferred = ["Task", "Úkol", "Story", "Bug"]
    for name in preferred:
        if name in available:
            return name

    return available[0]


def jira_already_has_issue_for_url(article_url: str) -> bool:
    """
    Kontrola duplicit přes JQL.
    Používá nový endpoint: /rest/api/3/search/jql
    """
    base = os.environ["JIRA_BASE_URL"].rstrip("/")
    email = os.environ["JIRA_EMAIL"]
    token = os.environ["JIRA_API_TOKEN"]
    project = os.environ["JIRA_PROJECT_KEY"]

    jql = (
        f'project = {project} '
        f'AND labels = "dateio-auto-translate" '
        f'AND text ~ "{article_url}"'
    )

    url = f"{base}/rest/api/3/search/jql"

    r = requests.get(
        url,
        auth=(email, token),
        headers={"Accept": "application/json"},
        params={"jql": jql, "maxResults": 1, "fields": "key"},
        timeout=30,
    )
    r.raise_for_status()
    return r.json().get("total", 0) > 0


def jira_create_issue(summary: str, description: str) -> str:
    base = os.environ["JIRA_BASE_URL"].rstrip("/")
    email = os.environ["JIRA_EMAIL"]
    token = os.environ["JIRA_API_TOKEN"]
    project = os.environ["JIRA_PROJECT_KEY"]

    # IMPORTANT: musí být mimo payload dict!
    issue_type_name = jira_get_valid_issue_type_name()

    url = f"{base}/rest/api/3/issue"

    payload = {
        "fields": {
            "project": {"key": project},
            "summary": summary,
            "issuetype": {"name": issue_type_name},
            "labels": ["dateio-auto-translate"],
            "description": {
                "type": "doc",
                "version": 1,
                "content": [
                    {
                        "type": "paragraph",
                        "content": [{"type": "text", "text": description}],
                    }
                ],
            },
        }
    }

    r = requests.post(
        url,
        json=payload,
        auth=(email, token),
        headers={"Accept": "application/json"},
        timeout=30,
    )

    if r.status_code >= 400:
        print("JIRA CREATE ISSUE ERROR")
        print("Status code:", r.status_code)
        try:
            print("Response JSON:", r.json())
        except Exception:
            print("Response text:", r.text)

    r.raise_for_status()
    return r.json()["key"]


def jira_attach_file(issue_key: str, filename: str, content: bytes) -> None:
    base = os.environ["JIRA_BASE_URL"].rstrip("/")
    email = os.environ["JIRA_EMAIL"]
    token = os.environ["JIRA_API_TOKEN"]

    url = f"{base}/rest/api/3/issue/{issue_key}/attachments"

    r = requests.post(
        url,
        auth=(email, token),
        headers={"X-Atlassian-Token": "nocheck"},
        files={"file": (filename, content)},
        timeout=60,
    )
    r.raise_for_status()


# ----------------------
# Main
# ----------------------
def main():
    processed = load_state()

    index_html = fetch_html(BLOG_INDEX)
    post_urls = extract_post_urls_from_index(index_html)

    new_urls = [u for u in post_urls if u not in processed]

    if not new_urls:
        print("Žádné nové články.")
        save_state(processed)  # vždy vytvoří state.json
        return

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    for url in new_urls:
        if jira_already_has_issue_for_url(url):
            print(f"Jira už má issue pro: {url}")
            processed.add(url)
            continue

        article_html = fetch_html(url)
        article = extract_article_text(article_html, url)

        if not article["body"]:
            print(f"Nelze extrahovat text: {url}")
            processed.add(url)
            continue

        translation = translate_to_hungarian(client, article["title"], article["body"])

        slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", url.split("/")[-1]).strip("-")
        filename = f"{slug}-hu.md"

        md_content = (
            f"# {article['title']}\n\n"
            f"Originál: {url}\n\n"
            f"---\n\n"
            f"{translation}\n"
        ).encode("utf-8")

        issue_key = jira_create_issue(
            summary=f"[HU] Překlad: {article['title']}",
            description=f"Překlad do maďarštiny.\n\nOriginál: {url}\n\nPřeklad je v příloze (Markdown).",
        )

        jira_attach_file(issue_key, filename, md_content)

        print(f"Hotovo → {issue_key}")
        processed.add(url)

    save_state(processed)


if __name__ == "__main__":
    main()
