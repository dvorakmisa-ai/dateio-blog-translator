import os
import requests

def jira_whoami() -> None:
    base = os.environ["JIRA_BASE_URL"].rstrip("/")
    email = os.environ["JIRA_EMAIL"]
    token = os.environ["JIRA_API_TOKEN"]

    url = f"{base}/rest/api/3/myself"
    r = requests.get(url, auth=(email, token), headers={"Accept": "application/json"}, timeout=30)

    print("\n=== JIRA WHOAMI (/myself) ===")
    print("HTTP:", r.status_code)
    try:
        print(r.json())
    except Exception:
        print(r.text)

    r.raise_for_status()


def jira_print_visible_projects() -> None:
    base = os.environ["JIRA_BASE_URL"].rstrip("/")
    email = os.environ["JIRA_EMAIL"]
    token = os.environ["JIRA_API_TOKEN"]

    url = f"{base}/rest/api/3/project/search"

    print("\n=== JIRA PROJECTS (/project/search) ===")

    start_at = 0
    total_printed = 0

    while True:
        r = requests.get(
            url,
            auth=(email, token),
            headers={"Accept": "application/json"},
            params={"maxResults": 50, "startAt": start_at},
            timeout=30,
        )

        print("Page startAt:", start_at, "HTTP:", r.status_code)

        if r.status_code >= 400:
            try:
                print(r.json())
            except Exception:
                print(r.text)
            r.raise_for_status()

        data = r.json()
        values = data.get("values", [])
        total = data.get("total", 0)

        if not values:
            break

        for p in values:
            print("-", p.get("key"), ":", p.get("name"))
            total_printed += 1

        start_at += len(values)
        if start_at >= total:
            break

    print("=== Printed:", total_printed, "projects ===\n")


def main():
    jira_whoami()
    jira_print_visible_projects()
    print("DIAGNOSTIKA HOTOVÁ – tady program končí schválně.")
    return


if __name__ == "__main__":
    main()
