import re
import random

def get_random_int(min_val = 0, max_val = 1000):
    return random.randint(min_val, max_val)


def extract_and_clean_text(input_string):
    # Regular expression pattern to find text between square brackets
    pattern = r'\[(.*?)\]'
    match = re.search(pattern, input_string)
    if match:
        extracted_text = match.group(1)  # Get the text between brackets
        cleaned_text = extracted_text.replace('"', '')  # Remove all double quotes
        return cleaned_text
    else:
        return " "  # Return None if no match is found
