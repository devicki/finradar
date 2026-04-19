"""Verify all FastAPI endpoints with real data."""
import httpx
import json
import sys

BASE_URL = "http://localhost:8000"


def test_endpoint(method: str, path: str, body: dict | None = None) -> dict:
    url = f"{BASE_URL}{path}"
    try:
        if method == "GET":
            resp = httpx.get(url, timeout=10)
        else:
            resp = httpx.post(url, json=body, timeout=10)

        data = resp.json()
        return {"path": path, "status": resp.status_code, "ok": resp.status_code == 200, "data": data}
    except Exception as e:
        return {"path": path, "status": 0, "ok": False, "data": str(e)}


def main():
    results = []

    # Health check
    results.append(test_endpoint("GET", "/health"))

    # News list
    results.append(test_endpoint("GET", "/api/v1/news/?page_size=3"))

    # News by topic
    results.append(test_endpoint("GET", "/api/v1/news/topics/"))

    # Feed
    results.append(test_endpoint("GET", "/api/v1/feed/?page_size=3"))

    # Feed summary
    results.append(test_endpoint("GET", "/api/v1/feed/summary?hours=24"))

    # Search
    results.append(test_endpoint("POST", "/api/v1/search/", {"query": "economy", "page_size": 3}))

    # Print results
    print(f"{'상태':>4}  {'경로':<40} {'결과'}")
    print("-" * 90)

    all_ok = True
    for r in results:
        status_icon = "OK" if r["ok"] else "FAIL"
        detail = ""
        if r["ok"]:
            data = r["data"]
            if "items" in data:
                detail = f"items={len(data['items'])}, total={data.get('total', '?')}"
            elif "status" in data:
                detail = f"status={data['status']}"
            elif isinstance(data, list):
                detail = f"count={len(data)}"
            else:
                detail = json.dumps(data, ensure_ascii=False)[:60]
        else:
            detail = str(r["data"])[:60]
            all_ok = False

        print(f"{status_icon:>4}  {r['path']:<40} {detail}")

    print()
    if all_ok:
        print("All endpoints OK")
    else:
        print("Some endpoints FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
