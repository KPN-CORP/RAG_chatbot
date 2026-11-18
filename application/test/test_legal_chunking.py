import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ingest.legal_chunking import build_legal_chunks, is_legal_document  # noqa: E402

# Textract lines as they come out of a scanned UU page (OCR errors kept on purpose).
PAGE_95 = """\
PRESIDEN
Refublik Indonesia
BAB V
PEMUTUSAN HUBUNGAN KERJA
Pasal
229
(3) Perundingan bipartit sebagaimana dimaksud dalam Pasal 228
ayat (3) tidak perlu dilakukan oleh Pemberi Kerja dalam hal:
a.
Pekerja/ Buruh mengundurkan diri atas kemauan sendiri;
b.
Pekerja/Buruh meninggal dunia.
Pasal 23O
(l)
Pemberi Kerja dilarang melakukan Pemutusan Hubungan
Kerja kepada Pekerja/Buruh dengan alasan:
a.
berhalangan masuk kerja karena sakit menurut keterangan dokter selama waktu tidak melampaui 12 (dua belas) bulan secara terus-menerus;
b.
berhalangan menjalankan pekerjaannya karena memenuhi kewajiban terhadap negara;
e.
hamil, melahirkan, gugur kandungan, atau menyusui bayinya;
f.
mempunyai pertalian darah
dan/
atau
ikatan
perkawinan dengan Pekerja/ Buruh lainnya di dalam 1 (satu) Perusahaan;
i.
status perkawinan; atau
SK No292928A
j.
dalam
"""

PAGE_96 = """\
PRESIDEN
Refubuk Indonesi,a
keadaan disabilitas tetap, sakit akibat kecelakaan kerja.
(21 Pemutusan Hubungan Kerja yang dilakukan dengan
alasan sebagaimana dimaksud pada ayat (1) batal demi hukum.
Pasal231
Pemutusan Hubungan Kerja dapat terjadi karena alasan:
a.
Perusahaan pailit;
- 96 -
Pasal 232
Cukup jelas.
Pasal 233
Cukup jelas.
"""

PAGES = [
    {"page": 95, "narrative_text": PAGE_95},
    {"page": 96, "narrative_text": PAGE_96},
]


def _by_pasal(chunks):
    return {c["breadcrumb"].split(" > ")[-1]: c for c in chunks}


def test_detects_legal_document():
    assert is_legal_document(PAGES)
    assert not is_legal_document([{"page": 1, "narrative_text": "Cuti tahunan 12 hari.\nPasal 3 ayat (1)"}])


def test_whole_pasal_is_one_chunk_across_pages():
    chunks = build_legal_chunks(PAGES, doc_id="uu", source="UU Nomor 27 Tahun 2026 - Copy.pdf")
    by_pasal = _by_pasal(chunks)

    assert {"Pasal 229", "Pasal 230", "Pasal 231"} <= set(by_pasal)
    p230 = by_pasal["Pasal 230"]
    text = p230["chunk_text"]

    assert p230["page"] == 95
    assert p230["element_type"] == "legal_article"
    assert p230["breadcrumb"] == (
        "UU Nomor 27 Tahun 2026 > BAB V Pemutusan Hubungan Kerja > Pasal 230"
    )
    # Ayat marker repaired and joined with its sentence, list items kept together.
    assert "(1) Pemberi Kerja dilarang melakukan Pemutusan Hubungan Kerja kepada" in text
    assert "a. berhalangan masuk kerja karena sakit" in text
    assert "e. hamil, melahirkan" in text
    assert "f. mempunyai pertalian darah dan/ atau ikatan perkawinan" in text
    # Item j continues on the next page; ayat (2) OCR "(21" is repaired.
    assert "j. dalam keadaan disabilitas tetap" in text
    assert "(2) Pemutusan Hubungan Kerja yang dilakukan" in text
    # Page furniture removed.
    for noise in ("PRESIDEN", "Refublik", "Refubuk", "SK No"):
        assert noise not in text


def test_bare_pasal_number_on_next_line_is_kept():
    chunks = build_legal_chunks(PAGES, doc_id="uu", source="UU.pdf")
    p229 = _by_pasal(chunks)["Pasal 229"]
    assert "a. Pekerja/ Buruh mengundurkan diri" in p229["chunk_text"]
    assert "Pasal 231" in _by_pasal(chunks)
    assert "- 96 -" not in _by_pasal(chunks)["Pasal 231"]["chunk_text"]


def test_long_pasal_split_at_ayat_boundaries():
    long_page = "Pasal 1\n" + "\n".join(f"({i}) " + "kata " * 60 for i in range(1, 8)) + "\nPasal 2\nx\nPasal 3\ny\nPasal 4\nz\nPasal 5\nw"
    chunks = build_legal_chunks([{"page": 1, "narrative_text": long_page}], doc_id="d", source="PP.pdf", max_chars=800)
    p1 = [c for c in chunks if c["breadcrumb"].endswith("Pasal 1")]
    assert len(p1) > 1
    assert "Pasal 1 (lanjutan)" in p1[1]["chunk_text"]
    assert all(len({c["chunk_id"] for c in chunks}) == len(chunks) for _ in [0])


def test_page_catchword_not_duplicated_and_long_lines_split():
    pages = [
        {"page": 1, "narrative_text": "Pasal 1\n(1) Larangan:\ni. status; atau\nj. dalam\n"},
        {"page": 2, "narrative_text": "j. dalam keadaan disabilitas tetap.\nPasal 2\n" + "Kalimat panjang sekali. " * 200
         + "\nPasal 3\nx\nPasal 4\ny\nPasal 5\nz"},
    ]
    chunks = build_legal_chunks(pages, doc_id="d", source="UU.pdf", max_chars=1000)
    p1 = _by_pasal(chunks)["Pasal 1"]["chunk_text"]
    assert p1.count("j. dalam") == 1
    assert "j. dalam keadaan disabilitas tetap." in p1
    assert max(len(c["chunk_text"]) for c in chunks) < 1200


def test_in_text_pasal_reference_is_not_a_new_article():
    text = (
        "Pasal 310\nx.\nPasal 311\nPada saat Undang-Undang ini mulai berlaku:\n"
        "b. Bab IV Bagian Kesatu Pasal 80 huruf a dan Bagian Kedua\nPasal\n81\n"
        "dalam lampiran Undang-Undang Nomor 6 Tahun 2023, dicabut.\n"
        "Pasal 23\nayat (2) juga dicabut.\nPasal 312\nPeraturan pelaksanaan.\nPasal 313\nBerlaku."
    )
    chunks = build_legal_chunks([{"page": 127, "narrative_text": text}], doc_id="d", source="UU.pdf")
    heads = [c["breadcrumb"].split(" > ")[-1] for c in chunks]
    assert heads == ["Pasal 310", "Pasal 311", "Pasal 312", "Pasal 313"]
    p311 = chunks[1]["chunk_text"]
    assert "Bagian Kedua Pasal 81 dalam lampiran" in p311
    assert "Pasal 23 ayat (2) juga dicabut." in p311
