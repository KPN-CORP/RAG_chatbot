"""Dev helper: run the REAL production extraction path on one PDF and print the
Gemini-produced chunks, so you can eyeball how tables (incl. merged cells) chunk.

Usage (run from the application/ directory, with the project venv):

    python validate_gemini_chunks.py "<path/to/file.pdf>"          # all pages
    python validate_gemini_chunks.py "<path/to/file.pdf>" 6        # only page 6

Notes:
- Reads GEMINI_API_KEY from application/.env (override=True).
- Makes real Gemini API calls (one per page that contains a table) -> uses quota.
- The Textract OCR fallback is disabled here (set WITH_OCR=1 to enable it). It is
  off by default because it needs valid AWS creds + a reachable S3 bucket, and is
  unrelated to inspecting Gemini table chunking.
"""

import io
import os
import sys

# Print UTF-8 so Indonesian text / symbols (≥, ⇒, •) don't crash the console.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Make sibling modules importable (config, ingest, modules, ...).
APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)


def _resolve_poppler_path() -> None:
    """Ensure Poppler (pdftoppm/pdfinfo) is findable by pdf2image.

    pdf2image shells out to Poppler to rasterise each PDF page. If it's already
    on PATH we do nothing; otherwise we point POPPLER_PATH at the winget install.
    """
    from shutil import which

    if which("pdfinfo") or os.environ.get("POPPLER_PATH"):
        return
    winget_bin = os.path.join(
        os.environ.get("LOCALAPPDATA", ""),
        "Microsoft", "WinGet", "Packages",
        "oschwartz10612.Poppler_Microsoft.Winget.Source_8wekyb3d8bbwe",
        "poppler-25.07.0", "Library", "bin",
    )
    if os.path.isdir(winget_bin):
        os.environ["POPPLER_PATH"] = winget_bin


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    pdf_path = sys.argv[1]
    only_page = int(sys.argv[2]) if len(sys.argv) > 2 else None

    _resolve_poppler_path()

    from dotenv import load_dotenv
    load_dotenv(os.path.join(APP_DIR, ".env"), override=True)

    if not os.environ.get("GEMINI_API_KEY"):
        print("!! GEMINI_API_KEY is empty. Add it to application/.env and re-run.")
        return 1

    import ingest.pdf_with_tables as pwt
    from ingest.pdf_with_tables import extract_pdf_content

    if os.environ.get("WITH_OCR") != "1":
        # Neutralise the Textract OCR fallback (see module docstring).
        pwt._apply_scanned_pdf_ocr_fallback = (
            lambda pdf_path, pages_output, profile="": pages_output
        )

    print(f"### FILE: {pdf_path}")
    print("### PATH: Gemini image extractor (use_gemini='use', production default)\n")

    pages = extract_pdf_content(pdf_path, use_gemini="use")

    any_chunks = False
    for pg in pages:
        page_no = pg.get("page")
        if only_page and page_no != only_page:
            continue
        gchunks = pg.get("gemini_chunks")
        if not gchunks:
            continue
        any_chunks = True
        print(f"===== page {page_no}: {len(gchunks)} Gemini chunks =====")
        for i, c in enumerate(gchunks, 1):
            text = c if isinstance(c, str) else (c.get("chunk_text") or c.get("text") or str(c))
            print(f"  [{i}] {text}")
        print()

    if not any_chunks:
        print("(No Gemini table chunks on the selected page(s). "
              "The page may have no tables, or Gemini returned nothing.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
