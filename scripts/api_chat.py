"""
Simple client for calling the running FastAPI server.
"""
import argparse
import json
from urllib import request


def post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Call the /chat endpoint of the local API.")
    ap.add_argument("--endpoint", default="http://localhost:8000/chat")
    ap.add_argument("--session", default="s1")
    ap.add_argument("--input", required=True)
    args = ap.parse_args()

    payload = {"session_id": args.session, "input": args.input}
    res = post_json(args.endpoint, payload)
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

