import json
import re
import unicodedata
from collections import Counter
from pathlib import Path


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def load_json(path: str):
    from pathlib import Path
    import json

    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file JSON: {file_path}")

    raw = file_path.read_text(encoding="utf-8").strip()

    if not raw:
        raise ValueError(f"File JSON đang rỗng: {file_path}")

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"JSON không hợp lệ tại file: {file_path}\n"
            f"line={e.lineno}, column={e.colno}, message={e.msg}"
        ) from e


def save_json(path: str, data) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.lower().strip()
    text = re.sub(r"[^\wÀ-ỹ\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str):
    text = normalize_text(text)
    tokens = re.findall(r"[a-zA-ZÀ-ỹ0-9_]+", text, flags=re.UNICODE)
    return tokens


def build_samples(intents_data: dict):
    samples = []
    responses_by_intent = {}

    for intent in intents_data["intents"]:
        tag = intent["tag"]
        responses_by_intent[tag] = intent["responses"]

        for pattern in intent["patterns"]:
            samples.append((pattern, tag))

    return samples, responses_by_intent


def build_vocab(samples, min_freq: int = 1):
    counter = Counter()

    for text, _ in samples:
        counter.update(tokenize(text))

    vocab = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1
    }

    for token, freq in counter.items():
        if freq >= min_freq and token not in vocab:
            vocab[token] = len(vocab)

    return vocab


def encode_text(text: str, vocab: dict, max_len: int):
    tokens = tokenize(text)

    if not tokens:
        tokens = [UNK_TOKEN]

    ids = [vocab.get(token, vocab[UNK_TOKEN]) for token in tokens[:max_len]]

    if len(ids) < max_len:
        ids.extend([vocab[PAD_TOKEN]] * (max_len - len(ids)))

    return ids