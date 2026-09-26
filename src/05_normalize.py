import re
import unicodedata
import pandas as pd


def normalize_text(text):
    """
    Language-independent normalization.
    Preserves Hindi, Tamil, Telugu, etc.
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Case normalization
    text = text.casefold()

    # Make & and "and" more consistent
    text = text.replace("&", " and ")

    # Replace punctuation with spaces
    text = "".join(
        " " if unicodedata.category(ch).startswith("P") else ch
        for ch in text
    )

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_dataframe(df):

    df = df.copy()

    df["name_normalized"] = (
        df["business_name"]
        .fillna("")
        .apply(normalize_text)
    )

    df["address_normalized"] = (
        df["business_address"]
        .fillna("")
        .apply(normalize_text)
    )

    return df


# -----------------------------------------
# TEST
# -----------------------------------------

if __name__ == "__main__":

    test_data = [
        "ABC Technologies Pvt. Ltd.",
        "ABC Technologies Private Limited",
        "ABC TECHNOLOGIES PVT LTD",
        "राम मार्केटिंग प्राइवेट लिमिटेड",
        "एसएस फूड प्राइवेट लिमिटेड",
        "Raj Investments LLP",
        "Tata & Sons",
        "Moyna's Coffee"
    ]

    print("\n========== MULTILINGUAL NORMALIZATION ==========\n")

    for value in test_data:
        print("Original   :", value)
        print("Normalized :", normalize_text(value))
        print()