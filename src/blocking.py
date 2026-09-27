import re
import unicodedata
from collections import defaultdict


# =========================================================
# NORMALIZATION
# =========================================================

def normalize_text(text):
    """
    Normalize text while preserving multilingual scripts.
    No translation is performed.
    """

    if text is None:
        return ""

    text = str(text)

    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Case-insensitive normalization
    text = text.casefold()

    # Treat '&' like 'and'
    text = text.replace("&", " and ")

    # Replace punctuation with spaces
    text = "".join(
        " " if unicodedata.category(ch).startswith("P")
        else ch
        for ch in text
    )

    # Remove repeated whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# =========================================================
# TOKENIZATION
# =========================================================

def get_tokens(text):
    """
    Return unique tokens from normalized text.
    """

    if not text:
        return set()

    return set(text.split())


# =========================================================
# NUMBER EXTRACTION
# =========================================================

def get_numbers(text):
    """
    Extract numeric components from an address.

    Example:
        '630 45th Terrace'
        -> {'630', '45'}
    """

    if not text:
        return set()

    return set(re.findall(r"\d+", text))


# =========================================================
# EXACT VALUE INDEX
# =========================================================

def build_exact_index(df, column):
    """
    Build:

        normalized value -> row indices

    Useful for exact normalized-name blocking.
    """

    index = defaultdict(list)

    for idx, value in df[column].items():

        if value:
            index[value].append(idx)

    return index


# =========================================================
# TOKEN INDEX
# =========================================================

def build_token_index(df, column):
    """
    Build:

        token -> row indices

    Used for name/address token blocking.
    """

    index = defaultdict(list)

    for idx, value in df[column].items():

        for token in get_tokens(value):

            index[token].append(idx)

    return index


# =========================================================
# NUMBER INDEX
# =========================================================

def build_number_index(df, column):
    """
    Build:

        number -> row indices

    Used for address-number blocking.
    """

    index = defaultdict(list)

    for idx, value in df[column].items():

        for number in get_numbers(value):

            index[number].append(idx)

    return index


# =========================================================
# TOKEN FREQUENCY
# =========================================================

def calculate_token_frequency(df, column):
    """
    Count how many records contain each token.

    Each token is counted at most once per record.
    """

    frequency = defaultdict(int)

    for value in df[column]:

        tokens = get_tokens(value)

        for token in tokens:
            frequency[token] += 1

    return dict(frequency)


# =========================================================
# RARE TOKEN LOOKUP
# =========================================================

def get_rare_tokens(text, frequency, max_frequency):
    """
    Return tokens whose frequency is <= max_frequency.
    """

    tokens = get_tokens(text)

    return {
        token
        for token in tokens
        if frequency.get(token, 0) <= max_frequency
    }


# =========================================================
# CANDIDATE LOOKUP FROM INDEX
# =========================================================

def lookup_candidates(tokens, index):
    """
    Retrieve all row indices associated with tokens.
    """

    candidates = set()

    for token in tokens:
        candidates.update(index.get(token, []))

    return candidates


# =========================================================
# EXACT NAME BLOCK
# =========================================================

def exact_name_block(name, name_index):
    """
    Return candidates having the same normalized name.
    """

    if not name:
        return set()

    return set(name_index.get(name, []))


# =========================================================
# RARE NAME TOKEN BLOCK
# =========================================================

def rare_name_block(
    name,
    name_index,
    name_frequency,
    max_frequency=100
):
    """
    Generate candidates using rare business-name tokens.
    """

    rare_tokens = get_rare_tokens(
        name,
        name_frequency,
        max_frequency
    )

    return lookup_candidates(
        rare_tokens,
        name_index
    )


# =========================================================
# RARE ADDRESS TOKEN BLOCK
# =========================================================

def rare_address_block(
    address,
    address_index,
    address_frequency,
    max_frequency=100
):
    """
    Generate candidates using rare address tokens.
    """

    rare_tokens = get_rare_tokens(
        address,
        address_frequency,
        max_frequency
    )

    return lookup_candidates(
        rare_tokens,
        address_index
    )


# =========================================================
# ADDRESS NUMBER BLOCK
# =========================================================

def address_number_block(
    address,
    number_index
):
    """
    Generate candidates sharing an address number.
    """

    numbers = get_numbers(address)

    return lookup_candidates(
        numbers,
        number_index
    )


# =========================================================
# COMBINE BLOCKS
# =========================================================

def combine_candidates(*candidate_sets):
    """
    Union candidates generated by multiple blocking rules.
    """

    result = set()

    for candidates in candidate_sets:
        result.update(candidates)

    return result