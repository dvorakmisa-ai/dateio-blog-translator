import os
import re
import json
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from bs4.element import Tag, NavigableString
from openai import OpenAI


BLOG_INDEX = "https://www.dateioplatform.com/resources/blog"
STATE_FILE = "state.json"


# -----------------------
# Env / state
# -----------------------
def require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(
            f"Chybí proměnná {name}. Přidej ji do GitHub Secrets a předej ve workflow env."
        )
    return v


def load_state() -> set[str]:
    if not os.path.exists(STATE_FILE):
        return set()
    with open(STATE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return set(data.get("processed_urls", []))


def save_state(processed_urls: set[str]) -> None:
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump({"processed_urls": sorted(processed_urls)}, f, ensure_ascii=False, indent=2)


# -----------------------
# HTTP helpers
# -----------------------
def fetch_html(url: str) -> str:
    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return r.text


# -----------------------
# Blog parsing
# -----------------------
def extract_post_urls_from_index(index_html: str) -> list[str]:
    soup = BeautifulSoup(index_html, "html.parser")
    urls = set()
    for a in soup.select("a[href]"):
        href = a.get("href", "")
        if "/resources/post/" in href:
            urls.add(urljoin(BLOG_INDEX, href))
    return sorted(urls)


def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _inline_to_md(node: Tag, base_url: str) -> str:
    """
    Inline HTML -> markdown-ish:
    - <strong>/<b> => **text**
    - <em>/<i> => *text*
    - <a href> => [text](absolute_url)
    - <br> => newline
    """
    def walk(n) -> str:
        if isinstance(n, NavigableString):
            return str(n)

        if not isinstance(n, Tag):
            return ""

        name = (n.name or "").lower()

        if name == "br":
            return "\n"

        if name in ("strong", "b"):
            inner = _normalize_ws("".join(walk(c) for c in n.children))
            return f"**{inner}**" if inner else ""

        if name in ("em", "i"):
            inner = _normalize_ws("".join(walk(c) for c in n.children))
            return f"*{inner}*" if inner else ""

        if name == "a":
            href = (n.get("href") or "").strip()
            text = _normalize_ws("".join(walk(c) for c in n.children)) or href
            if href:
                abs_href = urljoin(base_url, href)
                return f"[{text}]({abs_href})"
            return text

        # default: recurse
        return "".join(walk(c) for c in n.children)

    out = walk(node)
    # preserve newlines from <br>, but clean whitespace around them
    out = re.sub(r"[ \t]+\n", "\n", out)
    out = re.sub(r"\n[ \t]+", "\n", out)
    return out.strip()


def _html_to_mdish(root: Tag, base_url: str) -> str:
    """
    Block HTML -> markdown-ish with explicit heading labels:
      H1: Title
      H2: Subtitle
      - bullet
      1. numbered
      > quote
    Keeps inline **bold**, *italic*, and [text](url).
    """
    lines: list[str] = []

    def handle_block(el: Tag, indent: int = 0):
        name = (el.name or "").lower()

        if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            level = int(name[1])
            text = _normalize_ws(el.get_text(" ", strip=True))
            if text:
                lines.append(f"H{level}: {text}")
            return

        if name == "p":
            text = _inline_to_md(el, base_url).strip()
            if text:
                lines.append(text)
            return

        if name == "blockquote":
            text = _normalize_ws(el.get_text(" ", strip=True))
            if text:
                lines.append(f"> {text}")
            return

        if name == "ul":
            for li in el.find_all("li", recursive=False):
                txt = _normalize_ws(_inline_to_md(li, base_url))
                if txt:
                    lines.append(("  " * indent) + f"- {txt}")
                # nested lists directly under li
                for child in li.find_all(["ul", "ol"], recursive=False):
                    handle_block(child, indent=indent + 1)
            return

        if name == "ol":
            i = 1
            for li in el.find_all("li", recursive=False):
                txt = _normalize_ws(_inline_to_md(li, base_url))
                if txt:
                    lines.append(("  " * indent) + f"{i}. {txt}")
                i += 1
                for child in li.find_all(["ul", "ol"], recursive=False):
                    handle_block(child, indent=indent + 1)
            return

        # container: walk direct children to keep order
        for child in el.find_all(recursive=False):
            if isinstance(child, Tag):
                handle_block(child, indent=indent)

    # walk direct children to avoid pulling nav/footer multiple times
    for child in root.find_all(recursive=False):
        if isinstance(child, Tag):
            handle_block(child, indent=0)

    # join blocks
    out_lines: list[str] = []
    for ln in lines:
        # don't normalize list markers / quote markers too aggressively
        if ln.lstrip().startswith(("-", ">")) or re.match(r"^\s*\d+\.\s+", ln):
            out_lines.append(ln.rstrip())
        else:
            out_lines.append(_normalize_ws(ln))
    return "\n\n".join([x for x in out_lines if x.strip()]).strip()


def extract_article(article_html: str, article_url: str) -> dict:
    soup = BeautifulSoup(article_html, "html.parser")

    # Visible title (H1 preferred)
    title = "Untitled"
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        title = h1.get_text(strip=True)
    elif soup.title and soup.title.get_text(strip=True):
        title = soup.title.get_text(strip=True)

    # Meta title (OG preferred, fallback <title>)
    meta_title = None
    og_title = soup.find("meta", attrs={"property": "og:title"})
    if og_title and og_title.get("content"):
        meta_title = og_title["content"].strip()
    elif soup.title and soup.title.get_text(strip=True):
        meta_title = soup.title.get_text(strip=True)

    # Meta description (name=description preferred, fallback og:description)
    meta_description = None
    meta_desc = soup.find("meta", attrs={"name": "description"})
    if meta_desc and meta_desc.get("content"):
        meta_description = meta_desc["content"].strip()
    else:
        og_desc = soup.find("meta", attrs={"property": "og:description"})
        if og_desc and og_desc.get("content"):
            meta_description = og_desc["content"].strip()

    # Body: structured parse (headings, lists, bold, links)
    content_root = soup.find("article") or soup.select_one("main") or soup.body
    body = ""
    if content_root:
        body = _html_to_mdish(content_root, article_url).strip()

    return {
        "url": article_url,
        "title": title,
        "meta_title": meta_title,
        "meta_description": meta_description,
        "body": body,
    }


# -----------------------
# OpenAI translation (incl. metadata)
# -----------------------
def translate_to_hungarian(client: OpenAI, article: dict) -> dict:
    prompt = f"""
Přelož následující obsah do maďarštiny.

Vrať výstup POUZE jako JSON:
{{
  "title": "...",
  "meta_title": "...",
  "meta_description": "...",
  "body": "..."   // text s odstavci (může být Markdown)
}}

Pravidla:
- zachovej význam a marketingový tón
- meta_title ideálně do 60 znaků
- meta_description ideálně do 160 znaků
- body nezkracuj, zachovej odstavce
- zachovej strukturu a značky nadpisů přesně: řádky začínající "H1:", "H2:", ... musí zůstat a jen se přeloží text za dvojtečkou
- zachovej odrážky a číslování (řádky "- ..." a "1. ...")
- zachovej tučné písmo v Markdownu (**...**) a kurzívu (*...*)
- zachovej odkazy ve formátu [text](URL); URL se NESMÍ měnit, text v hranatých závorkách se přeloží
- pokud meta_title nebo meta_description chybí, navrhni je z obsahu
- žádné jiné klíče, žádné komentáře

# TITLE
{article["title"]}

# META TITLE
{article.get("meta_title") or ""}

# META DESCRIPTION
{article.get("meta_description") or ""}

# BODY
{article["body"]}
""".strip()

    resp = client.responses.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        input=prompt,
    )
    text = resp.output_text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            raise RuntimeError(f"Model nevrátil validní JSON. Výstup:\n{text}")
        return json.loads(m.group(0))


# -----------------------
# Jira ADF helpers
# -----------------------
def adf_text(text: str, marks: list[dict] | None = None) -> dict:
    node = {"type": "text", "text": text}
    if marks:
        node["marks"] = marks
    return node


def _adf_inline_from_mdish(text: str) -> list[dict]:
    """
    Parse a very small subset of markdown-ish inline:
      **bold**
      *italic*
      [text](url)
    Produces ADF text nodes with marks (strong/em/link).
    """
    text = text or ""
    out: list[dict] = []

    # split by links first
    link_pat = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

    pos = 0
    for m in link_pat.finditer(text):
        if m.start() > pos:
            out.extend(_adf_inline_parse_bold_italic(text[pos:m.start()]))

        label = m.group(1)
        href = m.group(2).strip()
        if label:
            out.append(adf_text(label, marks=[{"type": "link", "attrs": {"href": href}}]))
        pos = m.end()

    if pos < len(text):
        out.extend(_adf_inline_parse_bold_italic(text[pos:]))

    # merge adjacent plain nodes to keep ADF clean
    merged: list[dict] = []
    for n in out:
        if (
            merged
            and merged[-1].get("type") == "text"
            and n.get("type") == "text"
            and merged[-1].get("marks") == n.get("marks")
        ):
            merged[-1]["text"] += n.get("text", "")
        else:
            merged.append(n)
    return [n for n in merged if n.get("text")]


def _adf_inline_parse_bold_italic(s: str) -> list[dict]:
    """
    Parses **bold** and *italic* in a segment without links.
    Not a full markdown parser; good enough for web content.
    """
    nodes: list[dict] = []

    i = 0
    while i < len(s):
        # bold
        if s.startswith("**", i):
            j = s.find("**", i + 2)
            if j != -1:
                inner = s[i + 2:j]
                if inner:
                    nodes.append(adf_text(inner, marks=[{"type": "strong"}]))
                i = j + 2
                continue

        # italic (single *)
        if s.startswith("*", i) and not s.startswith("**", i):
            j = s.find("*", i + 1)
            if j != -1:
                inner = s[i + 1:j]
                if inner:
                    nodes.append(adf_text(inner, marks=[{"type": "em"}]))
                i = j + 1
                continue

        # plain char run
        # take until next marker
        next_positions = []
        nb = s.find("**", i)
        ni = s.find("*", i)
        if nb != -1:
            next_positions.append(nb)
        if ni != -1:
            next_positions.append(ni)
        nxt = min(next_positions) if next_positions else -1

        if nxt == -1:
            nodes.append(adf_text(s[i:]))
            break
        else:
            if nxt > i:
                nodes.append(adf_text(s[i:nxt]))
            i = nxt

    return nodes


def adf_paragraph(text: str) -> dict:
    # normalize newlines inside paragraph to spaces; blocks are separated elsewhere
    text = (text or "").replace("\n", " ").strip()
    content = _adf_inline_from_mdish(text) if text else [adf_text("")]
    return {"type": "paragraph", "content": content}


def adf_heading(text: str, level: int = 2) -> dict:
    text = (text or "").strip()
    content = _adf_inline_from_mdish(text) if text else [adf_text("")]
    return {"type": "heading", "attrs": {"level": level}, "content": content}


def adf_list_item(text: str) -> dict:
    return {"type": "listItem", "content": [adf_paragraph(text)]}


def adf_bullet_list(items: list[str]) -> dict:
    return {"type": "bulletList", "content": [adf_list_item(i) for i in items]}


def adf_ordered_list(items: list[str]) -> dict:
    return {"type": "orderedList", "content": [adf_list_item(i) for i in items]}


def mdish_to_adf_blocks(text: str) -> list[dict]:
    """
    Conversion of our md-ish blocks:
    - blocks separated by blank lines
    - headings: "H2: ..." or "# ..."
    - bullet list: lines "- ..."
    - ordered list: lines "1. ..."
    - quote: lines starting with "> "
    - everything else paragraph
    """
    blocks: list[dict] = []
    text = (text or "").strip()
    if not text:
        return [adf_paragraph("(empty)")]

    for raw_block in text.split("\n\n"):
        b = raw_block.strip()
        if not b:
            continue

        # Headings "H2: ..."
        m2 = re.match(r"^H([1-6]):\s+(.*)$", b)
        if m2:
            level = int(m2.group(1))
            blocks.append(adf_heading(m2.group(2).strip(), level=level))
            continue

        # Markdown headings "# ..."
        m = re.match(r"^(#{1,6})\s+(.*)$", b)
        if m:
            level = min(6, max(1, len(m.group(1))))
            blocks.append(adf_heading(m.group(2).strip(), level=level))
            continue

        # Quote
        if all(ln.strip().startswith(">") for ln in b.split("\n") if ln.strip()):
            # join quote lines into one paragraph (without >)
            q = "\n".join([re.sub(r"^\s*>\s?", "", ln).rstrip() for ln in b.split("\n") if ln.strip()])
            blocks.append({"type": "blockquote", "content": [adf_paragraph(q)]})
            continue

        lines = [ln.rstrip() for ln in b.split("\n") if ln.strip()]

        # Bullet list
        if lines and all(re.match(r"^\s*-\s+.+$", ln) for ln in lines):
            items = [re.sub(r"^\s*-\s+", "", ln).strip() for ln in lines]
            blocks.append(adf_bullet_list(items))
            continue

        # Ordered list
        if lines and all(re.match(r"^\s*\d+\.\s+.+$", ln) for ln in lines):
            items = [re.sub(r"^\s*\d+\.\s+", "", ln).strip() for ln in lines]
            blocks.append(adf_ordered_list(items))
            continue

        # Paragraph fallback (keep soft line breaks as spaces)
        b = re.sub(r"\n+", "\n", b)
        blocks.append(adf_paragraph(b))

    return blocks


def build_description_adf(article: dict, hu: dict) -> dict:
    content: list[dict] = []

    content.append(adf_heading("HU překlad blogu", level=2))

    content.append(adf_heading("Originál", level=3))
    content.append(adf_paragraph(f"URL: {article['url']}"))
    content.append(adf_paragraph(f"Title: {article.get('title') or ''}"))
    content.append(adf_paragraph(f"Meta title: {article.get('meta_title') or ''}"))
    content.append(adf_paragraph(f"Meta description: {article.get('meta_description') or ''}"))

    content.append(adf_heading("Překlad (HU) – metadata", level=3))
    content.append(adf_paragraph(f"Title (HU): {(hu.get('title') or '').strip()}"))
    content.append(adf_paragraph(f"Meta title (HU): {(hu.get('meta_title') or '').strip()}"))
    content.append(adf_paragraph(f"Meta description (HU): {(hu.get('meta_description') or '').strip()}"))

    content.append(adf_heading("Překlad (HU) – obsah", level=3))
    content.extend(mdish_to_adf_blocks((hu.get("body") or "").strip()))

    return {"type": "doc", "version": 1, "content": content}


# -----------------------
# Jira API
# -----------------------
def jira_request(method: str, path: str, **kwargs) -> requests.Response:
    base = require_env("JIRA_BASE_URL").rstrip("/")
    email = require_env("JIRA_EMAIL")
    token = require_env("JIRA_API_TOKEN")

    url = f"{base}{path}"
    headers = kwargs.pop("headers", {})
    headers.setdefault("Accept", "application/json")

    r = requests.request(
        method,
        url,
        auth=(email, token),
        headers=headers,
        timeout=30,
        **kwargs,
    )

    if r.status_code == 401:
        raise RuntimeError(
            "Jira vrátila 401 Unauthorized.\n"
            "Zkontroluj:\n"
            "- JIRA_BASE_URL (např. https://firma.atlassian.net)\n"
            "- JIRA_EMAIL (stejný účet, který token vytvořil)\n"
            "- JIRA_API_TOKEN (zkus vygenerovat nový)\n"
        )

    return r


def jira_diagnostic() -> None:
    """
    Optional debug: prints whoami + visible projects.
    Enable with env JIRA_DIAGNOSTIC=1
    """
    if os.getenv("JIRA_DIAGNOSTIC", "").strip() != "1":
        return

    print("\n=== JIRA DIAGNOSTIC ===")
    me = jira_request("GET", "/rest/api/3/myself")
    print("JIRA /myself:", me.status_code)
    if me.status_code < 400:
        j = me.json()
        print("displayName:", j.get("displayName"))
        print("emailAddress:", j.get("emailAddress"))

    proj = jira_request("GET", "/rest/api/3/project/search", params={"maxResults": 50})
    print("JIRA /project/search:", proj.status_code)
    if proj.status_code < 400:
        values = proj.json().get("values", [])
        print("Visible projects count:", len(values))
        for p in values:
            print("-", p.get("key"), ":", p.get("name"))
    print("=== END DIAGNOSTIC ===\n")


def jira_issue_exists_for_url(project_key: str, article_url: str) -> bool:
    # Keep your original logic for now (duplicates fix is a separate topic you postponed)
    jql = f'project = {project_key} AND labels = "dateio-auto-translate" AND text ~ "{article_url}"'
    r = jira_request(
        "GET",
        "/rest/api/3/search/jql",
        params={"jql": jql, "maxResults": 1, "fields": "key"},
    )
    r.raise_for_status()
    return r.json().get("total", 0) > 0


def jira_get_issue_type_name(project_key: str) -> str:
    """
    Zjistí validní issue type pro projekt.
    - ignoruje sub-task typy (vyžadují parent)
    - preferuje doménové typy jako 'Článek'
    """
    desired = (os.getenv("JIRA_ISSUE_TYPE") or "").strip()
    project_key = (project_key or "").strip().strip('"').strip("'")

    r = jira_request(
        "GET",
        f"/rest/api/3/project/{project_key}",
        params={"expand": "issueTypes"},
    )

    if r.status_code == 404:
        try:
            print("JIRA PROJECT LOOKUP 404 JSON:", r.json())
        except Exception:
            print("JIRA PROJECT LOOKUP 404 TEXT:", r.text)
        raise RuntimeError(
            f"Projekt s key '{project_key}' nebyl nalezen nebo není viditelný pro účet v JIRA_EMAIL."
        )

    r.raise_for_status()
    data = r.json()

    issue_types = data.get("issueTypes", [])

    non_subtasks = [
        it for it in issue_types
        if it.get("name") and not it.get("subtask", False)
    ]

    if not non_subtasks:
        raise RuntimeError("Projekt nemá žádný non-subtask issue type.")

    names = [it["name"] for it in non_subtasks]
    print("Dostupné issue types (bez sub-task):", ", ".join(names))

    if desired:
        if desired in names:
            print("Používám issue type z JIRA_ISSUE_TYPE:", desired)
            return desired
        else:
            print(f"JIRA_ISSUE_TYPE='{desired}' není validní pro projekt.")

    preferred = [
        "Platform content",
    ]

    for p in preferred:
        if p in names:
            print("Vybraný issue type (preferred):", p)
            return p

    print("Vybraný issue type (fallback):", names[0])
    return names[0]


def jira_create_issue(summary: str, description_adf: dict) -> str:
    project_key = require_env("JIRA_PROJECT_KEY").strip().strip('"').strip("'")
    issue_type = jira_get_issue_type_name(project_key)

    payload = {
        "fields": {
            "project": {"key": project_key},
            "summary": summary,
            "issuetype": {"name": issue_type},
            "labels": ["dateio-auto-translate"],
            "description": description_adf,
        }
    }

    r = jira_request(
        "POST",
        "/rest/api/3/issue",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        json=payload,
    )

    if r.status_code >= 400:
        print("JIRA CREATE ISSUE ERROR", r.status_code)
        try:
            print(r.json())
        except Exception:
            print(r.text)

    r.raise_for_status()
    return r.json()["key"]


# -----------------------
# Main
# -----------------------
def main():
    require_env("OPENAI_API_KEY")
    require_env("JIRA_BASE_URL")
    require_env("JIRA_EMAIL")
    require_env("JIRA_API_TOKEN")
    require_env("JIRA_PROJECT_KEY")

    jira_diagnostic()

    processed = load_state()

    index_html = fetch_html(BLOG_INDEX)
    post_urls = extract_post_urls_from_index(index_html)
    new_urls = [u for u in post_urls if u not in processed]

    if not new_urls:
        print("Žádné nové články.")
        save_state(processed)
        return

    openai_client = OpenAI(api_key=require_env("OPENAI_API_KEY"))
    project_key = require_env("JIRA_PROJECT_KEY").strip().strip('"').strip("'")

    for url in new_urls:
        if jira_issue_exists_for_url(project_key, url):
            print(f"V Jira už existuje issue pro: {url}")
            processed.add(url)
            continue

        article_html = fetch_html(url)
        article = extract_article(article_html, url)

        if not article["body"]:
            print(f"Nelze extrahovat text: {url}")
            processed.add(url)
            continue

        hu = translate_to_hungarian(openai_client, article)
        description_adf = build_description_adf(article, hu)

        summary = f"[HU] {article['title']}"
        issue_key = jira_create_issue(summary, description_adf)

        print(f"Vytvořeno: {issue_key} ← {url}")
        processed.add(url)

    save_state(processed)


if __name__ == "__main__":
    main()
