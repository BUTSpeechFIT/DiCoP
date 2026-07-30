import re

DISFLUENCIES = {
    'oh', 'ha', 'um', 'uh', 'ah', 'hmm', 'haahaa', 'mmm', 'ohhh', 'ohh', 'ahh', 'hahaha', 'ohhhh', 'haaa',
    'hmmm', 'haa', 'ahhh', 'umm', 'haha', 'mmmm', 'ummm', 'hah', 'hhh', 'ahw', 'hm', 'haahaahaa',
    'hahahaha', 'hmmmm', 'hmmmmmmmm', 'aah', 'haaaa', 'uhh', 'hahah', 'hai', 'uhhh', 'ohw', 'ahhhh',
    'haahaaa', 'hahahah', 'hhhh', 'hahahahaha', 'mmmmm', 'ummmm', 'aaaa', 'ohhhhh', 'sss', 'uuu', '000',
    'aaah', 'hhhhh', 'hmmmmm', 'hmmmmmmm', 'www', 'aaahh', 'haaaaaa', 'huu', 'ohhhhhhh', 'ohhhhhhhh',
    'ohhhhhhhhhhhhhh', 'aaa', 'aahw', 'eee', 'hahahha', 'hh', 'hmmmmmm', 'hoo', 'ooo', 'uhhhh', 'uhhhhh',
    'aaaaa', 'aahhh', 'haaaaa', 'haah', 'hahahahahahaha', 'hahhaha', 'ohhhhhh', 'rrr', 'ummmmm', 'uuuu',
    'wwww', 'aahm', 'ahhhhhhhhh', 'er', 'haaaaaaa', 'haaaaaaaa', 'hahaa', 'hahaaaa', 'hahahaa',
    'hahahahahaha', 'hahahahha', 'hahahuh', 'hahhah', 'hahhhh', 'hahuhu', 'hooo', 'mmmmmmm', 'oooo',
    'ssssss', 'ummmmmmm', 'yah', 'yyyyyyyyyyyy', '999', 'aaaahhm', 'aaahhh', 'aaahhhmmm', 'aahh', 'aahmm',
    'ahhhhh', 'ahhhhhhhhhh', 'ahhhhhhhhhhh', 'eeee', 'ffff', 'haaaaaaaaa', 'haaaaaaaaaa',
    'haaaaaaaaaaaaaaaaaaa', 'haahaha', 'haahahaha', 'haahuuuuu', 'hahaaa', 'hahaaaaa', 'hahaaha',
    'hahahaaah', 'hahahahaahahha', 'hahahahah', 'hahahahahah', 'hahahahahahahaha', 'hahahahahha',
    'hahahahu', 'hahahahuh', 'hahahahuhu', 'hahahhaa', 'hahahoho', 'hahahu', 'hahahuha', 'hahha',
    'hahhaaha', 'hahhh', 'hahu', 'hahuh', 'hahuhahuh', 'hahuhuhu', 'haisho', 'hap', 'haummm', 'hhhhhh',
    'hhhhhhh', 'huhahihi', 'huhuhuha', 'huuu', 'huuuu', 'huuuuu', 'lll', 'mchhh', 'mmmmmm', 'nnn', 'nnnnn',
    'nnnnnn', 'ohahahahhu', 'ohhhhhhhhh', 'ohhhhhhhhhhh', 'ohhhhhhhhhhhh', 'ohhhhhhhhhhhhhhhhh', 'ohhn',
    'ohhp', 'ohooo', 'ooooo', 'oooooo', 'ooooooooo', 'oooooooooooooooooooooooooo', 'ppppppp', 'ssss',
    'sssss', 'uhhhhhhh', 'uhhhhhhhhhhhh', 'ummmmmmmm', 'ummmmmmmmm', 'yyy', 'yyyyyyy', 'yay', 'hehehe',
    'shhhhh', 'uhhhhmm', 'huh', 'hhahaha', 'huhuhuh', 'hmmhmm', 'huhhhhhhh', 'wow', 'huhuuu', 'yea',
    'huhhhhh', 'huhh', 'huhuhu', 'whoa', 'huhhh', 'yeah', 'huhuu', 'sshhh', 'huhuuhhu', 'uhm', 'huhmmmm',
    'huhhu', 'onnnnnn', 'huhummm', 'oohh', 'mmmhmmm', 'oohhoa', 'ss', 'mhmm', 'sshhhhh', 'uhmm', 'ssshh',
    'mmhmm', 'huhuhh', 'oohhh',
}
STRIP_CHARS = '.,!?;:\'"-][~+'
SPECIAL_CHARS = {
    '%', '$', '!', '"', '&', '*', '+', ':', '£', '|', '<', '>', '/', ']', ')', '~', '[', '_', '(', '-',
    '.', ',', '\'', ';', '?', '=', '@', '#', '^', '\\', '`', '{', '}', '’',
}
PATTERNS = {
    "word_end_with_punct": re.compile(r'^\w+[.,!?;:]+$'),
    "word_with_contractions": re.compile(r"^[A-Za-z]?[a-z]+(?:['’](?:[a-z]{1,2}|m|re|ve|ll|s|t))?$"),
    "word_with_hyphen": re.compile(r"^[a-zA-Z]+(?:-[a-zA-Z]+)+$"),
    "number_and_percentage": re.compile(r"^[0-9]+(?:\.[0-9]+)?%$"),
    "special_whisper": re.compile(r"^[a-zA-Z]+[.,?!']*<\|\w+\|><\|(translate|transcribe)\|>$"),
    "float_number": re.compile(r"^[0-9]+[\.,]+[0-9]+$"),
    "abbreviation": re.compile(r"[a-z]{1}(\.[a-z]{1})+$"),
    "domain_name": re.compile(r"^[a-zA-Z0-9]+(?:\.[a-zA-Z0-9]+)+$"),
}
NUMBER_AND_DOLLAR_PATTERNS = (
    re.compile(r"\d{1,10}[\.,]*(?:,\d{3})*\d*\$$"),
    re.compile(r"\$\d{1,10}[\.,]*(?:,\d{3})*\d*$"),
)
POUND_AND_NUMBER_PATTERNS = (
    re.compile(r"\d{1,10}[\.,]*(?:,\d{3})*\d*\£$"),
    re.compile(r"\£\d{1,10}[\.,]*(?:,\d{3})*\d*$"),
)


def remove_disfluencies(text):
    return " ".join(word for word in text.split() if word.lower() not in DISFLUENCIES)


def _normalize_number_token(word: str) -> str:
    word = word.replace(',', '')
    return word.replace('.', ' point ')


def format_text(word, w_type):
    word = word.upper()
    if w_type == "special_whisper":
        normalized = word.split("<")[0].strip(STRIP_CHARS)
    else:
        word = word.strip(STRIP_CHARS)
        if w_type in {"word_end_with_punct", "word_with_contractions"}:
            normalized = word
        elif w_type == "word_with_hyphen":
            normalized = word.replace("-", " ")
        elif w_type == "number_and_percentage":
            normalized = _normalize_number_token(word).replace('%', ' percent')
        elif w_type == "number_and_dollar":
            normalized = _normalize_number_token(word.replace('$', '')) + " dollar"
        elif w_type == "pound_and_number":
            normalized = _normalize_number_token(word.replace('£', '')) + " pound"
        elif w_type == "float_number":
            normalized = _normalize_number_token(word)
        elif w_type == "domain_name":
            normalized = word.replace('.', ' dot ')
        elif w_type == "abbreviation":
            normalized = word.replace('.', '')
        else:
            normalized = re.sub(r"[^a-zA-Z0-9' ]", " ", word)
    return re.sub(r"\s+", " ", normalized).upper()


def is_valid_word(word):
    word = word.lower()
    if PATTERNS["word_end_with_punct"].match(word):
        return True, "word_end_with_punct"

    stripped_word = word.strip(STRIP_CHARS)
    for word_type in (
        "word_with_contractions",
        "word_with_hyphen",
        "number_and_percentage",
        "special_whisper",
        "float_number",
        "abbreviation",
        "domain_name",
    ):
        if PATTERNS[word_type].match(stripped_word):
            return True, word_type

    if any(pattern.match(stripped_word) for pattern in NUMBER_AND_DOLLAR_PATTERNS):
        return True, "number_and_dollar"
    if any(pattern.match(stripped_word) for pattern in POUND_AND_NUMBER_PATTERNS):
        return True, "pound_and_number"

    return False, "unknown"


def norm_string(text):
    norm_words = []
    for word in text.strip().split():
        if set(word) & SPECIAL_CHARS:
            _, w_type = is_valid_word(word)
        else:
            w_type = "unknown"
        norm_words.append(format_text(word, w_type))
    return " ".join(norm_words)
