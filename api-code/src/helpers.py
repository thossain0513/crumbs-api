import re


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
