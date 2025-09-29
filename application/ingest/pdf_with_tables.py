import pdfplumber
from fuzzywuzzy import fuzz, process
from typing import List, Tuple, Any, Iterable
# from collections import Counter
import re

def bbox_intersects(a, b):
    # a and b = [x0, top, x1, bottom]
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    horiz = not (ax1 <= bx0 or bx1 <= ax0)
    vert = not (ay1 <= by0 or by1 <= ay0)
    return horiz and vert

def get_page_elements(pdf_path):
    pages_output = []
    with pdfplumber.open(pdf_path) as pdf:
        for pidx, page in enumerate(pdf.pages, start=1):
            # 1) find table bboxes (pdfplumber Table objects via find_tables)
            table_bboxes = []
            tables = []
            try:
                table_objs = page.find_tables()
            except Exception:
                table_objs = []
            for t_idx, t in enumerate(table_objs):
                # pdfplumber Table has .bbox and .extract() method
                tbbox = t.bbox  # (x0, top, x1, bottom)
                table_bboxes.append({"bbox": tbbox, "table_index": t_idx})
                # get rows as list of lists
                try:
                    rows = t.extract()
                except Exception:
                    # fallback to page.extract_tables
                    rows = None
                tables.append({"table_index": t_idx, "bbox": tbbox, "rows": rows})

            # 2) extract words with bboxes and filter out words that fall into any table bbox
            words = page.extract_words()  # each item has x0, x1, top (y0), bottom (y1), text
            narrative_words = []
            for w in words:
                wbox = (float(w["x0"]), float(w["top"]), float(w["x1"]), float(w["bottom"]))
                in_table = any(bbox_intersects(wbox, tb["bbox"]) for tb in table_bboxes)
                if not in_table:
                    narrative_words.append(w)
            # 3) rebuild narrative text using word order grouping by line (simple)
            # group by 'top' with small tolerance to form lines
            lines = {}
            tol = 3  # pixels tolerance
            for w in narrative_words:
                top = round(float(w["top"]) / tol) * tol
                lines.setdefault(top, []).append((float(w["x0"]), w["text"]))
            # sort lines by top, then words by x0
            sorted_lines = [ " ".join([t[1] for t in sorted(line, key=lambda x: x[0])]) for top, line in sorted(lines.items()) ]

            narrative_text = "\n".join(sorted_lines).strip()
            # 4) for the page output, include tables (as DataFrames if available)
            table_records = []
            for t in tables:
                if t["rows"]:
                    table_dct = t['rows']
                else:
#                     # fallback: try page.extract_tables and pick the matching by bbox proximity
                    table_dct = None
                table_records.append({"table_index": t["table_index"], "bbox": t["bbox"], "table_dct": table_dct})

            pages_output.append({
                "page": pidx,
                "narrative_text": narrative_text,
                "tables": table_records
            })
    return pages_output

def get_all_tables(pdf_pages: Iterable[dict]) -> Tuple[List[Any], List[str]]:
    """
    Return (tables, tables_flatted)
    - tables: list of each table['table_dct'] (or None if missing)
    - tables_flatted: string for each table where cells in a row are joined by spaces
      and rows are joined by newlines.
    Behavior:
      - treats None cells as empty string (but preserves 0 and False as values).
      - tolerant if 'tables' or 'table_dct' keys are missing.
    """
    tables: List[Any] = []
    tables_flatted: List[str] = []

    append_table = tables.append
    append_flat = tables_flatted.append

    for page in pdf_pages:
        # safer access to page['tables']
        for table in page.get("tables", []):
            # prefer dict-style access but be robust to different shapes
            td = table.get("table_dct") if isinstance(table, dict) else getattr(table, "table_dct", None)
            append_table(td)

            # guard: if td is not an iterable of rows, use empty list
            if td is None:
                append_flat("")  # keep alignment: one flat string per table entry
                continue

            # build flattened text efficiently using join() (linear time)
            # convert None -> '' but keep falsy numerics (0) as '0'
            flat = "\n".join(
                " ".join(
                    ("" if cell is None else str(cell))
                    for cell in row
                )
                for row in td
            )
            flat = re.sub(r'\s+', ' ', flat)

            append_flat(flat)

    return tables, tables_flatted

def clean_table(table):
    '''
    clean None with previous cell (based on result, None value is the previous. Ex: CP_Perjalanan Dinas)
    '''
    for row in table:
        for i, cell in enumerate(row):
            if cell==None and i>0:
                row[i] = row[i-1]
    return table

def change_to_records(table):
    '''change table format from a list of list to records'''
    table = clean_table(table)
    headers = table[0]
    data_rows = table[1:]
    # Use a list comprehension to build the list of dictionaries
    records = [dict(zip(headers, row)) for row in data_rows]
    return records

def fix_nested_table(table, tables, tables_flatted):
    '''fix for a table inside a table
    if there is a table inside a table, 
    returning a fixed table that the value is a table, not messy text'''
    for index_row, row in enumerate(table):
        for index_col, cell in enumerate(row):
            if cell and isinstance(cell, str):
                if len(cell)>20: # only check for long text
                    one,score = process.extractOne(cell, tables_flatted, scorer=fuzz.token_sort_ratio)
                    if score>80:
                        idx = tables_flatted.index(one)
                        # update the table inside a table with real table
                        cleaned_table = clean_table(tables[idx])
                        cleaned_table = change_to_records(cleaned_table)
                        table[index_row][index_col] = cleaned_table
    return table

def check_double_header(table_data):
    """
    Analyzes a table's structure and returns its type.
    """
    if len(table_data) < 2:
        return False # or "too_small"

    header_row_1 = table_data[0]
    header_row_2 = table_data[1]

    # Heuristic for the complex table:
    # Does the second row have non-None values under None values in the first?
    # This is a strong sign of a merged, hierarchical header.

    for i, item in enumerate(header_row_1):
        if item is None and header_row_2[i] is not None:
            return True
    
    # Heuristic for a simple table:
    # The first row is full and the second row looks like data.
    if all(isinstance(h, str) for h in header_row_1):
        return False
        
    return False # Default fallback

def check_triple_header(table_data):
    """
    Analyzes a table's structure and returns its type.
    """
    if len(table_data) < 2:
        return False # or "too_small"

    header_row_2 = table_data[1]
    header_row_3 = table_data[2]

    # Heuristic for the complex table:
    # Does the third row have non-None values under None values in the second?
    # This is a strong sign of a merged, hierarchical header.

    for i, item in enumerate(header_row_2):
#         print('item:', item, "below:", header_row_3[i])
        if item is None and header_row_3[i] is not None and i>0:
#             print('   ^True')
            return True
    
    # Heuristic for a simple table:
    # The first row is full and the second row looks like data.
    if all(isinstance(h, str) for h in header_row_2):
        return False
        
    return False # Default fallback


def process_double_header_table(table_data):
    """
    Processes a complex, multi-header table into a list of dictionaries (records),
    where each dictionary represents a row.
    """
    if len(table_data) < 3:
        # Not enough rows for headers and data
        return []

    # --- Step 1: Reconstruct and Flatten the Headers (No changes here) ---
    header_row_1 = table_data[0]
    header_row_2 = table_data[1]
    
    # Forward-fill the top-level headers
    filled_header_1 = []
    last_header = ""
    for header in header_row_1:
        if header is not None:
            last_header = header
        filled_header_1.append(last_header)

    # Combine headers into a final, flat list
    flattened_headers = []
    for i, sub_header in enumerate(header_row_2):
        main_header = filled_header_1[i]
        if sub_header:
            clean_sub_header = sub_header.replace('\n', ' ')
            flattened_headers.append(f"{main_header} - {clean_sub_header}")
        else:
            flattened_headers.append(main_header)

    # --- Step 2: Build a List of Records (This part is new) ---
    records = []
    data_rows = table_data[2:] # The rest of the list is data

    for row in data_rows:
        # Skip any row that is empty or doesn't have a primary identifier
        if not row or row[0] is None:
            continue

        # Create a dictionary for the current row by zipping headers and row values
        # The dict() constructor combined with zip() is a concise way to do this.
        record = dict(zip(flattened_headers, row))
        records.append(record)
            
    return records

def process_triple_header_table(table_data):
    """
    Processes a complex, TRIPLE-header table into a list of dictionaries (records).
    """
    if len(table_data) < 4:  # 3 header rows + at least 1 data row
        return []

    # --- Step 1: Reconstruct and Flatten the Three-Level Headers ---
    header_row_1 = table_data[0]
    header_row_2 = table_data[1]
    header_row_3 = table_data[2]

    # Level 1: Forward-fill the top-most header row
    filled_header_1 = []
    last_header = ""
    for header in header_row_1:
        if header is not None:
            last_header = header.replace('\n', ' ')
        filled_header_1.append(last_header)

    # Level 2: Forward-fill the middle header row
    filled_header_2 = []
    last_header = ""
    for i, header in enumerate(header_row_2):
        # Reset fill if the top-level header changes, making it more robust
        if filled_header_1[i] != filled_header_1[i-1] and i > 0:
            last_header = ""
        if header is not None:
            last_header = header.replace('\n', ' ')
        filled_header_2.append(last_header)

    # Level 3: Combine all three filled headers into a final, flat list
    flattened_headers = []
    for i in range(len(header_row_1)):
        # Collect parts from all three levels, cleaning newlines
        h1 = filled_header_1[i]
        h2 = filled_header_2[i]
        h3 = header_row_3[i].replace('\n', ' ') if header_row_3[i] else ""

        # Build the final header by joining the parts that are not empty
        parts = [part for part in [h1, h2, h3] if part]
        unique_parts = list(dict.fromkeys(parts)) # Remove duplicates like 'Gol.' - 'Gol.'
        final_header = " - ".join(unique_parts)
        flattened_headers.append(final_header)

    # --- Step 2: Build a List of Records (Same logic as before) ---
    records = []
    data_rows = table_data[3:]  # The rest is data

    for row in data_rows:
        if not row or row[0] is None:
            continue

        record = dict(zip(flattened_headers, row))
        records.append(record)
            
    return records

def check_and_fix_nested_tables(pdf_pages):
    '''check and fix nested table'''
    tables, tables_flatted = get_all_tables(pdf_pages)
    for page_idx, page in enumerate(pdf_pages):
        if page['tables']:
            for i, dct in enumerate(page['tables']):
                if dct['table_dct']:
                    table = dct['table_dct']
                    fixed_table = fix_nested_table(table, tables, tables_flatted)
                pdf_pages[page_idx]['tables'][i]['table_dct'] = fixed_table
    return pdf_pages

def change_table_format(pdf_pages):
    '''change table format from a list of list to records and handle double header'''
    for page_idx, page in enumerate(pdf_pages):
        if page['tables']:
            for i, dct in enumerate(page['tables']):
                if dct['table_dct']:
                    is_double_header = check_double_header(dct['table_dct'])
                    is_triple_header = check_triple_header(dct['table_dct']) if is_double_header else False

                    table = clean_table(dct['table_dct'])

                    if is_triple_header:
                        records = process_triple_header_table(dct['table_dct'])
                    elif is_double_header:
                        records = process_double_header_table(dct['table_dct'])
                    else:
                        headers = table[0]
                        data_rows = table[1:]
                        # Use a list comprehension to build the list of dictionaries
                        records = [dict(zip(headers, row)) for row in data_rows]
                pdf_pages[page_idx]['tables'][i]['table_dct'] = records
    return pdf_pages

def extract_pdf_content(file):
    """
    Extract all texts from a PDF file, including narrative text and tables.

    Steps:
    1. Parse each page to extract narrative text and tables.
    2. Fix nested tables (tables embedded as text within other tables).
    3. Convert table format from list of lists to list of records (dicts).

    Args:
        file (str): Path to the PDF file.

    Returns:
        List[dict]: List of page dictionaries containing narrative text and tables.
    """
    pdf_pages = get_page_elements(file)                # Step 1: Extract page elements
    pdf_pages = check_and_fix_nested_tables(pdf_pages) # Step 2: Fix nested tables
    pdf_pages = change_table_format(pdf_pages)         # Step 3: Format tables as records

    return pdf_pages
