import re
import time
import random
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin


HEADERS = {
    "User-Agent": "Academic Chatbot"
}

def check_availability(doi: str, max_retries: int = 5) -> int:
    url = f"https://api.elsevier.com/content/article/doi/{doi}?view=FULL"

    backoff = 1.0  # starting delay in seconds

    for attempt in range(max_retries + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)

            # Success
            if r.status_code == 200:
                return 1

            # Rate limit -> retry with backoff (prefer Retry-After if present)
            if r.status_code == 429:
                retry_after = r.headers.get("Retry-After")
                if retry_after is not None:
                    try:
                        sleep_time = float(retry_after)
                    except ValueError:
                        sleep_time = backoff
                else:
                    sleep_time = backoff

                # small jitter to avoid synchronized retries
                sleep_time += random.uniform(0, 0.25)
                time.sleep(sleep_time)
                backoff = min(backoff * 2, 30)
                continue

            # Temporary server errors -> retry
            if 500 <= r.status_code < 600:
                time.sleep(backoff + random.uniform(0, 0.25))
                backoff = min(backoff * 2, 30)
                continue

            # Anything else (401/403/404 etc.) -> don't retry
            return 0

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            # Network issues -> retry
            if attempt < max_retries:
                time.sleep(backoff + random.uniform(0, 0.25))
                backoff = min(backoff * 2, 30)
                continue
            return 0

        except Exception as e:
            print(f"Error for {doi}: {e}")
            return 0

    return 0

def safe_filename(name: str) -> str:
    # Windows-illegal characters: <>:"/\|?*
    return re.sub(r'[<>:"/\\|?*]', "_", name)

def save2pdf(doi: str):
    base_url = f"https://wellesu.com/{doi}"

    try:
        resp = requests.get(base_url, headers=HEADERS, timeout=20)
        resp.raise_for_status()

        if "application/pdf" in (resp.headers.get("Content-Type") or ""):
            pdf_bytes = resp.content

        else:
            
            soup = BeautifulSoup(resp.text, "lxml")

            pdf_url = None

            embed = soup.find("embed", {"type": "application/pdf"})
            if embed and embed.get("src"):
                pdf_url = embed["src"]

            
            if not pdf_url:
                iframe = soup.find("iframe", src=True)
                if iframe and iframe["src"].lower().endswith(".pdf"):
                    pdf_url = iframe["src"]

            # <object data="...pdf">
            if not pdf_url:
                obj = soup.find("object", {"type": "application/pdf"})
                if obj and obj.get("data"):
                    pdf_url = obj["data"]

            if not pdf_url:
                raise RuntimeError("PDF link not found in HTML")

            pdf_url = urljoin(base_url, pdf_url)

            pdf_resp = requests.get(pdf_url, headers=HEADERS, timeout=30)
            pdf_resp.raise_for_status()

            if "application/pdf" not in (pdf_resp.headers.get("Content-Type") or "") \
               and not pdf_resp.content.startswith(b"%PDF"):
                raise RuntimeError("Fetched file is not a valid PDF")

            pdf_bytes = pdf_resp.content

        filename = safe_filename(doi) + ".pdf"
        with open(f"papers/{filename}", "wb") as f:
            f.write(pdf_bytes)

        print(f"Saved paper: {filename}")
        return filename

    except Exception as e:
        print(f"Failed for {doi}: {e}")
