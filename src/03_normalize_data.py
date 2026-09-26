import re
import unicodedata


def normalize_text(text):
    """
    Basic normalization for business names and addresses.
    """

    if text is None:
        return ""

    text = str(text)

    # Convert Unicode characters into a consistent form
    text = unicodedata.normalize("NFKC", text)

    # Convert to lowercase
    text = text.lower()

    # Replace '&' with 'and'
    text = text.replace("&", " and ")

    # Remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================
# TEST THE NORMALIZATION
# ============================================

examples = [
    "ABC Technologies Pvt. Ltd.",
    "ABC Technologies Private Limited",
    "ABC TECHNOLOGIES PVT LTD",
    "Road No. 10, Banjara Hills, Hyderabad",
    "Rd No 10, Banjara Hills, Hyderabad",
    "Tata & Sons",
    "Tata and Sons",
    "Moyna's Coffee"
]


print("========== NORMALIZATION TEST ==========")

for value in examples:

    print("\nOriginal :", value)
    print("Normalized:", normalize_text(value))