# filename: utils.py
import re
import json
# from typing import List

def extract_and_convert_dictionary_string(s):
    """
    Extracts the content between the first '{' and the last '}',
    and then attempts to convert it to a dictionary.
    """
    # 1. Extraction/Filtering using regex
    # Pattern: \{ (literal '{') .* (any characters) \} (literal '}')
    match = re.search(r'\{.*\}', s, re.DOTALL)

    if not match:
        print(f"Error: Could not find content enclosed in '{{...}}' in string: {s}")
        return None

    dict_string = match.group(0)

    # 2. Conversion
    try:
        data = json.loads(dict_string)
        if isinstance(data, dict):
            return data
        else:
            print(f"Error: Extracted content was valid JSON but resulted in a {type(data).__name__}, not a dictionary.")
            return None
    except json.JSONDecodeError as e:
        print(f"Error: Extracted content is not valid JSON. {e}")
        return None


