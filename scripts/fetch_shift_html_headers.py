from __future__ import annotations

import re
import requests


def main() -> int:
    gid = 2025020705
    season_dir = "20252026"
    suffix = str(gid)[4:]
    urls = [
        f"https://www.nhl.com/scores/htmlreports/{season_dir}/TV{suffix}.HTM",
        f"https://www.nhl.com/scores/htmlreports/{season_dir}/TH{suffix}.HTM",
    ]

    for url in urls:
        print("\nURL", url)
        r = requests.get(url, timeout=30)
        print("status", r.status_code, "len", len(r.text or ""))
        html = r.text or ""

        # grab some colspan header texts (these are usually player headers)
        headers = re.findall(r"<td[^>]*colspan=\"?\d+\"?[^>]*>(.*?)</td>", html, flags=re.I | re.S)
        cleaned: list[str] = []
        for h in headers:
            t = re.sub(r"<[^>]+>", " ", h)
            t = re.sub(r"\s+", " ", t).strip()
            if t and t not in cleaned:
                cleaned.append(t)
            if len(cleaned) >= 30:
                break
        print("sample colspan texts:")
        for t in cleaned[:20]:
            print(" ", t)

        # find examples that look like player headers
        playerish = [t for t in cleaned if re.match(r"^\d{1,2}\s+", t)]
        print("playerish headers (first 20):")
        for t in playerish[:20]:
            print(" ", t)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
