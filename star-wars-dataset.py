import argparse
import json
import os
import time
import datetime
import requests
import pandas as pd
import mwparserfromhell

API_URL = "https://starwars.fandom.com/api.php"
HEADERS = {"User-Agent": "VaderDatasetBuilder/1.0 (personal research project)"}

BATCH_SIZE = 20
FLUSH_EVERY = 100
MIN_TEXT_LEN = 200
MAX_RETRIES = 5
RETRY_BACKOFF = 2.0


def api_get(params):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(API_URL, params=params, headers=HEADERS, timeout=30)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as e:
            if attempt == MAX_RETRIES:
                raise
            wait = RETRY_BACKOFF * (2 ** (attempt - 1))
            print(f"[retry] request failed ({e}), attempt {attempt}/{MAX_RETRIES}, waiting {wait:.0f}s")
            time.sleep(wait)


# --------------------------------------------------------------------------
# scrape
# --------------------------------------------------------------------------
def bootstrap_from_previous_run(out_dir, resume_from_input):
    if resume_from_input and not os.path.exists(out_dir):
        if os.path.exists(resume_from_input):
            import shutil
            shutil.copytree(resume_from_input, out_dir)
            print(f"[resumed] copied prior output forward from {resume_from_input}")
        else:
            print(f"[warn] --resume-from set but not found at {resume_from_input}, starting fresh")


def get_all_page_titles(titles_path):
    if os.path.exists(titles_path):
        with open(titles_path) as f:
            titles = json.load(f)
        print(f"[cache] {len(titles):,} titles already fetched, skipping phase 1")
        return titles

    titles = []
    apcontinue = None
    while True:
        params = {
            "action": "query", "list": "allpages", "apnamespace": 0,
            "apfilterredir": "nonredirects",
            "aplimit": "500", "format": "json",
        }
        if apcontinue:
            params["apcontinue"] = apcontinue
        data = api_get(params)
        titles.extend(p["title"] for p in data["query"]["allpages"])
        if "continue" in data:
            apcontinue = data["continue"]["apcontinue"]
        else:
            break
        time.sleep(0.2)

    with open(titles_path, "w") as f:
        json.dump(titles, f)
    print(f"[done] {len(titles):,} titles fetched, cached to {titles_path}")
    return titles


def get_plaintext(titles_batch):
    params = {
        "action": "query", "prop": "revisions|info",
        "rvprop": "content", "rvslots": "main", "inprop": "url",
        "titles": "|".join(titles_batch), "format": "json",
    }
    data = api_get(params)
    pages = data.get("query", {}).get("pages", {})

    result = {}
    for p in pages.values():
        title = p.get("title", "")
        url = p.get("fullurl", "")
        revisions = p.get("revisions", [])
        wikitext = revisions[0].get("slots", {}).get("main", {}).get("*", "") if revisions else ""

        if wikitext.strip().upper().startswith("#REDIRECT"):
            continue  # belt-and-braces, apfilterredir should already exclude these

        text = mwparserfromhell.parse(wikitext).strip_code() if wikitext else ""
        result[title] = {"text": text, "url": url}
    return result


def load_progress(progress_path):
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            return json.load(f)
    return {"titles_done": 0, "articles_written": 0}


def save_progress(progress_path, state):
    tmp = progress_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, progress_path)


def cmd_scrape(args):
    out_dir = args.out_dir
    titles_path = os.path.join(out_dir, "titles.json")
    progress_path = os.path.join(out_dir, "progress.json")
    articles_path = os.path.join(out_dir, "star_wars_articles.jsonl")
    scrape_date = datetime.date.today().isoformat()

    os.makedirs(out_dir, exist_ok=True)
    bootstrap_from_previous_run(out_dir, args.resume_from)

    titles = get_all_page_titles(titles_path)
    state = load_progress(progress_path)

    if state["titles_done"] > 0:
        print(f"[resumed] {state['titles_done']:,}/{len(titles):,} titles already processed, "
              f"{state['articles_written']:,} articles written so far")

    buffer = []
    with open(articles_path, "a", encoding="utf-8") as out_f:
        for i in range(state["titles_done"], len(titles), BATCH_SIZE):
            batch = titles[i:i + BATCH_SIZE]
            pages = get_plaintext(batch)

            for title, page in pages.items():
                if len(page["text"]) < MIN_TEXT_LEN:
                    continue
                buffer.append(json.dumps({
                    "title": title,
                    "text": page["text"],
                    "url": page["url"],
                    "source": "Wookieepedia",
                    "license": "CC BY-SA 3.0",
                    "scrape_date": scrape_date,
                }) + "\n")

            state["titles_done"] = min(i + BATCH_SIZE, len(titles))

            if len(buffer) >= FLUSH_EVERY:
                out_f.writelines(buffer)
                out_f.flush()
                state["articles_written"] += len(buffer)
                buffer = []
                save_progress(progress_path, state)

            if state["titles_done"] % 2000 == 0:
                print(f"{state['titles_done']:,}/{len(titles):,} titles processed, "
                      f"{state['articles_written']:,} articles written")

            time.sleep(0.3)

        if buffer:
            out_f.writelines(buffer)
            state["articles_written"] += len(buffer)
            save_progress(progress_path, state)

    print(f"[complete] {state['articles_written']:,} articles in {articles_path}")


# --------------------------------------------------------------------------
# sanity check
# --------------------------------------------------------------------------
def cmd_sanity(args):
    sample = get_plaintext(["Luke Skywalker", "Anakin Skywalker", "Tatooine"])
    ok = True
    for title, page in sample.items():
        length = len(page["text"])
        print(f"{title}: {length} chars, url={page['url']}")
        if length < MIN_TEXT_LEN:
            ok = False
    if not ok:
        raise SystemExit("[FAILED] one or more sample pages returned no real text, do not run scrape yet")
    print("[passed] safe to run 'scrape'")


# --------------------------------------------------------------------------
# convert
# --------------------------------------------------------------------------
def cmd_convert(args):
    if not os.path.exists(args.input) or os.path.getsize(args.input) == 0:
        raise SystemExit(
            f"[error] {args.input} is missing or empty. "
            f"Has 'scrape' actually written any articles yet? Check progress.json."
        )

    df = pd.read_json(args.input, lines=True)

    if "text" not in df.columns:
        raise SystemExit(
            f"[error] parsed {len(df)} rows but no 'text' column found, "
            f"file may be truncated or corrupted, check the raw JSONL manually."
        )

    print(f"{len(df):,} articles, {df['text'].str.len().sum():,} characters total")

    before = len(df)
    df = df.drop_duplicates(subset="title", keep="last")
    if len(df) != before:
        print(f"dropped {before - len(df):,} duplicate titles")

    df.to_parquet(args.output, index=False)
    print(f"wrote {args.output}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Star Wars (Wookieepedia) dataset builder")
    sub = parser.add_subparsers(dest="command", required=True)

    p_sanity = sub.add_parser("sanity", help="quick check that the API is returning real text before committing to a full scrape")
    p_sanity.set_defaults(func=cmd_sanity)

    p_scrape = sub.add_parser("scrape", help="scrape Wookieepedia via the Fandom API")
    p_scrape.add_argument("--out-dir", default="/kaggle/working/star_wars_raw")
    p_scrape.add_argument("--resume-from", default=None)
    p_scrape.set_defaults(func=cmd_scrape)

    p_convert = sub.add_parser("convert", help="convert scraped JSONL into a Parquet file")
    p_convert.add_argument("--input", default="/kaggle/working/star_wars_raw/star_wars_articles.jsonl")
    p_convert.add_argument("--output", default="/kaggle/working/star_wars_corpus.parquet")
    p_convert.set_defaults(func=cmd_convert)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
