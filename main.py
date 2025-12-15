def jira_already_has_issue_for_url(article_url: str) -> bool:
    """
    Kontrola duplicit v Jira:
    Atlassian odstranil /rest/api/3/search (vrací 410 Gone) a místo toho je /rest/api/3/search/jql. :contentReference[oaicite:1]{index=1}
    """
    base_url = os.environ["JIRA_BASE_URL"].rstrip("/")
    email = os.environ["JIRA_EMAIL"]
    token = os.environ["JIRA_API_TOKEN"]
    project_key = os.environ["JIRA_PROJECT_KEY"]

    jql = f'project = {project_key} AND labels = "dateio-auto-translate" AND text ~ "{article_url}"'
    url = f"{base_url}/rest/api/3/search/jql"

    r = requests.get(
        url,
        headers={"Accept": "application/json"},
        auth=(email, token),
        params={
            "jql": jql,
            "maxResults": 1,
            "fields": "key",   # ať je odpověď malá
        },
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    return data.get("total", 0) > 0
