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


# ===== Structured HTML -> md-ish (headings, lists, bold, links, images) =====
def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _inline_to_mdish(node: Tag, base_url: str) -> str:
    """
    Inline HTML -> markdown-ish:
      <strong>/<b> -> **text**
      <em>/<i>     -> *text*
      <a href>     -> [text](absolute_url)
      <br>         -> newline
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
            inner = _norm_ws("".join(walk(c) for c in n.children))
            return f"**{inner}**" if inner else ""

        if name in ("em", "i"):
            inner = _norm_ws("".join(walk(c) for c in n.children))
            return f"*{inner}*" if inner else ""

        if name == "a":
            href = (n.get("href") or "").strip()
            text = _norm_ws("".join(walk(c) for c in n.children)) or href
            if href:
                abs_href = urljoin(base_url, href)
                return f"[{text}]({abs_href})"
            return text

        return "".join(walk(c) for c in n.children)

    out = walk(node)
    out = re.sub(r"[ \t]+\n", "\n", out)
    out = re.sub(r"\n[ \t]+", "\n", out)
    return out.strip()


def _img_src_from_tag(img: Tag) -> str:
    """
    Try common src attributes including lazy-load variants.
    """
    if not img:
        return ""
    for key in ("src", "data-src", "data-original", "data-lazy-src", "data-srcset"):
        v = (img.get(key) or "").strip()
        if v:
            if key.endswith("srcset") and "," in v:
                v = v.split(",")[0].strip().split(" ")[0].strip()
            return v
    return ""


def _direct_text_only(el: Tag) -> str:
    """
    Returns only direct text nodes (not descendant text), normalized.
    Useful for Webflow divs where text isn't wrapped in <p>.
    """
    if not isinstance(el, Tag):
        return ""
    parts = []
    for c in el.contents:
        if isinstance(c, NavigableString):
            parts.append(str(c))
    return _norm_ws("".join(parts))


def _html_to_mdish(root: Tag, base_url: str) -> str:
    """
    Block HTML -> md-ish with explicit heading labels:
      H1: ...
      H2: ...
      - bullet
      1. numbered
      IMG1: alt="..." caption="..." src=...
    Keeps inline **bold**, *italic*, and [text](url).
    """
    blocks: list[str] = []
    img_counter = {"n": 0}

    def add_image_placeholder(img: Tag, caption: str = ""):
        img_counter["n"] += 1
        src = _img_src_from_tag(img)
        alt = _norm_ws(img.get("alt") or "") if img else ""
        abs_src = urljoin(base_url, src) if src else ""

        parts = [f"IMG{img_counter['n']}:"]
        if alt:
            parts.append(f'alt="{alt}"')
        if caption:
            parts.append(f'caption="{_norm_ws(caption)}"')
        if abs_src:
            parts.append(f"src={abs_src}")

        blocks.append(" ".join(parts))

    def handle(el: Tag, indent: int = 0):
        name = (el.name or "").lower()

        # Headings
        if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            level = int(name[1])
            text = _norm_ws(el.get_text(" ", strip=True))
            if text:
                blocks.append(f"H{level}: {text}")
            return

        # Figure with img + caption
        if name == "figure":
            img = el.find("img")
            if img:
                cap_el = el.find("figcaption")
                caption = cap_el.get_text(" ", strip=True) if cap_el else ""
                add_image_placeholder(img, caption=caption)
            return

        # Standalone img
        if name == "img":
            add_image_placeholder(el, caption="")
            return

        # Paragraph
        if name == "p":
            text = _inline_to_mdish(el, base_url).strip()
            if text:
                blocks.append(text)
            return

        # Lists
        if name == "ul":
            for li in el.find_all("li", recursive=False):
                txt = _norm_ws(_inline_to_mdish(li, base_url))
                if txt:
                    blocks.append(("  " * indent) + f"- {txt}")
                for child in li.find_all(["ul", "ol"], recursive=False):
                    handle(child, indent=indent + 1)
            return

        if name == "ol":
            i = 1
            for li in el.find_all("li", recursive=False):
                txt = _norm_ws(_inline_to_mdish(li, base_url))
                if txt:
                    blocks.append(("  " * indent) + f"{i}. {txt}")
                i += 1
                for child in li.find_all(["ul", "ol"], recursive=False):
                    handle(child, indent=indent + 1)
            return

        # capture direct text inside div/section when not wrapped in <p>
        if name in ("div", "section", "article", "main"):
            direct = _direct_text_only(el)
            if direct:
                blocks.append(direct)
            for child in el.find_all(recursive=False):
                if isinstance(child, Tag):
                    handle(child, indent=indent)
            return

        # Default: containers recurse
        for child in el.find_all(recursive=False):
            if isinstance(child, Tag):
                handle(child, indent=indent)

    # Walk direct children to reduce nav/footer noise
    for child in root.find_all(recursive=False):
        if isinstance(child, Tag):
            handle(child, indent=0)

    # Join as paragraphs separated by blank lines
    out: list[str] = []
    for b in blocks:
        if (
            b.lstrip().startswith("-")
            or re.match(r"^\s*\d+\.\s+", b)
            or re.match(r"^IMG\d+:\s+", b)
        ):
            out.append(b.rstrip())
        else:
            out.append(_norm_ws(b))
    return "\n\n".join([x for x in out if x.strip()]).strip()
# ===== END structured parsing =====


# ===== FAQ post-processing from the md-ish body =====
def _rewrite_faq_section(body: str) -> str:
    text = (body or "").strip()
    if not text:
        return text

    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]

    def is_heading(b: str) -> bool:
        return bool(re.match(r"^H[1-6]:\s+", b)) or bool(re.match(r"^(#{1,6})\s+", b))

    def is_faq_heading(b: str) -> bool:
        if re.match(r"^H[1-6]:\s*FAQs?\s*$", b, flags=re.IGNORECASE):
            return True
        m = re.match(r"^(#{1,6})\s*(FAQs?)\s*$", b, flags=re.IGNORECASE)
        return bool(m)

    out: list[str] = []
    i = 0
    while i < len(blocks):
        b = blocks[i]
        if not is_faq_heading(b):
            out.append(b)
            i += 1
            continue

        out.append("H2: FAQs")
        i += 1

        faq_items: list[str] = []
        while i < len(blocks) and not is_heading(blocks[i]):
            faq_items.append(blocks[i])
            i += 1

        qn = 1
        j = 0
        while j < len(faq_items):
            q = faq_items[j].strip()
            a = faq_items[j + 1].strip() if j + 1 < len(faq_items) else ""
            if q:
                out.append(f"FAQ Q{qn}: {q}")
                if a:
                    out.append(f"FAQ A{qn}: {a}")
                qn += 1
            j += 2

    return "\n\n".join(out).strip()
# ===== END FAQ post-processing =====


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

    # Body: structured parse with fallback
    content_root = soup.find("article") or soup.select_one("main") or soup.body
    body = ""
    if content_root:
        body = _html_to_mdish(content_root, article_url).strip()

    if not body:
        paragraphs = [
            p.get_text(" ", strip=True)
            for p in soup.find_all("p")
            if p.get_text(strip=True)
        ]
        body = "\n\n".join(paragraphs).strip()

    body = _rewrite_faq_section(body)

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
- zachovej strukturu:
  - řádky začínající "H1:", "H2:", ... ponech a přelož jen text za dvojtečkou
  - zachovej FAQ strukturu: řádky "FAQ Q1:", "FAQ A1:" ponech, přelož jen text za dvojtečkou
  - zachovej odrážky "- ..." a číslování "1. ..."
  - zachovej **tučné** a *kurzívu* (markdown značky)
  - zachovej odkazy ve formátu [text](URL); URL se NESMÍ měnit
  - zachovej řádky začínající "IMG" (např. "IMG1: ...") – neupravuj je a nepřekládej URL/atributy
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
def adf_text(text: str) -> dict:
    return {"type": "text", "text": text}


def adf_paragraph(text: str) -> dict:
    return {"type": "paragraph", "content": [adf_text(text)]}


def adf_heading(text: str, level: int = 2) -> dict:
    return {"type": "heading", "attrs": {"level": level}, "content": [adf_text(text)]}


def adf_list_item(text: str) -> dict:
    return {"type": "listItem", "content": [adf_paragraph(text)]}


def adf_bullet_list(items: list[str]) -> dict:
    return {"type": "bulletList", "content": [adf_list_item(i) for i in items]}


def adf_ordered_list(items: list[str]) -> dict:
    return {"type": "orderedList", "content": [adf_list_item(i) for i in items]}


def mdish_to_adf_blocks(text: str) -> list[dict]:
    blocks: list[dict] = []
    text = (text or "").strip()
    if not text:
        return [adf_paragraph("(empty)")]

    for raw_block in text.split("\n\n"):
        b = raw_block.strip()
        if not b:
            continue

        m_h = re.match(r"^H([1-6]):\s+(.*)$", b)
        if m_h:
            level = int(m_h.group(1))
            blocks.append(adf_heading(m_h.group(2).strip(), level=level))
            continue

        m = re.match(r"^(#{1,6})\s+(.*)$", b)
        if m:
            level = min(6, max(1, len(m.group(1))))
            blocks.append(adf_heading(m.group(2).strip(), level=level))
            continue

        if re.match(r"^IMG\d+:\s+", b):
            blocks.append(adf_paragraph("🖼️ " + b))
            continue

        if re.match(r"^FAQ\s+[QA]\d+:\s+", b):
            blocks.append(adf_paragraph("❓ " + b))
            continue

        lines = [ln.rstrip() for ln in b.split("\n") if ln.strip()]

        if lines and all(re.match(r"^\s*-\s+.+$", ln) for ln in lines):
            items = [re.sub(r"^\s*-\s+", "", ln).strip() for ln in lines]
            blocks.append(adf_bullet_list(items))
            continue

        if lines and all(re.match(r"^\s*\d+\.\s+.+$", ln) for ln in lines):
            items = [re.sub(r"^\s*\d+\.\s+", "", ln).strip() for ln in lines]
            blocks.append(adf_ordered_list(items))
            continue

        b = re.sub(r"\n+", "\n", b)
        blocks.append(adf_paragraph(b))

    return blocks


def build_description_adf(article: dict, hu: dict) -> dict:
    content: list[dict] = []

    content.append(adf_heading("HU překlad blogu", level=2))

    content.append(adf_heading("Originál", level=3))

    # NEW: stable dedupe marker
    content.append(adf_paragraph(f"SOURCE_URL: {article['url']}"))

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


# NEW: robust JQL escaping
def jql_escape(s: str) -> str:
    return (s or "").replace("\\", "\\\\").replace('"', '\\"')


# CHANGED: dedupe by SOURCE_URL marker in description (stable)
def jira_issue_exists_for_url(project_key: str, article_url: str) -> bool:
    project_key = (project_key or "").strip().strip('"').strip("'")
    needle = f"SOURCE_URL: {article_url}"

    jql = (
        f'project = "{project_key}" '
        f'AND labels = dateio-auto-translate '
        f'AND description ~ "{jql_escape(needle)}"'
    )

    r = jira_request(
        "GET",
        "/rest/api/3/search",
        params={"jql": jql, "maxResults": 1, "fields": "key"},
    )
    r.raise_for_status()
    return r.json().get("total", 0) > 0


def jira_get_issue_type_name(project_key: str) -> str:
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
        "Článek",
        "Platform content",
        "Tapix content",
        "Epic",
        "Task",
        "Úkol",
        "Story",
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
        # robust Jira dedupe
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
