import argparse
import json
import os
import time
import datetime
import requests
import pandas as pd

API_URL = "https://starwars.fandom.com/api.php"
HEADERS = {"User-Agent": "VaderDatasetBuilder/1.0 (personal research project)"}

BATCH_SIZE = 20
FLUSH_EVERY = 100
MIN_TEXT_LEN = 200


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
            "aplimit": "500", "format": "json",
        }
        if apcontinue:
            params["apcontinue"] = apcontinue
        r = requests.get(API_URL, params=params, headers=HEADERS, timeout=30)
        data = r.json()
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
        "action": "query", "prop": "extracts|info", "explaintext": 1,
        "exsectionformat": "plain", "inprop": "url",
        "titles": "|".join(titles_batch), "format": "json",
    }
    r = requests.get(API_URL, params=params, headers=HEADERS, timeout=30)
    pages = r.json().get("query", {}).get("pages", {})
    return {
        p["title"]: {"text": p.get("extract", ""), "url": p.get("fullurl", "")}
        for p in pages.values()
    }


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
# convert
# --------------------------------------------------------------------------
def cmd_convert(args):
    df = pd.read_json(args.input, lines=True)
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

    p_scrape = sub.add_parser("scrape", help="scrape Wookieepedia via the Fandom API")
    p_scrape.add_argument("--out-dir", default="/kaggle/working/star_wars_raw",
                           help="output directory for titles.json, progress.json, articles.jsonl")
    p_scrape.add_argument("--resume-from", default=None,
                           help="path to a previous session's output dir, e.g. an attached Kaggle input")
    p_scrape.set_defaults(func=cmd_scrape)

    p_convert = sub.add_parser("convert", help="convert scraped JSONL into a Parquet file")
    p_convert.add_argument("--input", default="/kaggle/working/star_wars_raw/star_wars_articles.jsonl",
                            help="path to the JSONL produced by the scrape command")
    p_convert.add_argument("--output", default="/kaggle/working/star_wars_corpus.parquet",
                            help="path to write the output Parquet file")
    p_convert.set_defaults(func=cmd_convert)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
