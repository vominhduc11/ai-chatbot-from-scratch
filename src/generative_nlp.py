import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]


def load_json(path: str):
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file JSON: {file_path}")

    raw = file_path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"File JSON đang rỗng: {file_path}")

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"JSON không hợp lệ tại file: {file_path}\n"
            f"line={exc.lineno}, column={exc.colno}, message={exc.msg}"
        ) from exc


def save_json(path: str, data) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.lower().strip()
    text = re.sub(r"[^\wÀ-ỹ\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    normalized = normalize_text(text)
    return re.findall(r"[a-zA-ZÀ-ỹ0-9_]+", normalized, flags=re.UNICODE)


def build_vocab_from_pairs(pairs: Iterable[dict], min_freq: int = 1) -> dict[str, int]:
    token_freq: dict[str, int] = {}

    for pair in pairs:
        for text in (pair["input"], pair["output"]):
            for token in tokenize(text):
                token_freq[token] = token_freq.get(token, 0) + 1

    vocab: dict[str, int] = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}

    for token, freq in sorted(token_freq.items(), key=lambda item: item[0]):
        if freq >= min_freq and token not in vocab:
            vocab[token] = len(vocab)

    return vocab


def encode_source(text: str, vocab: dict[str, int], max_len: int) -> list[int]:
    tokens = tokenize(text)
    ids = [vocab.get(token, vocab[UNK_TOKEN]) for token in tokens]
    ids.append(vocab[EOS_TOKEN])
    ids = ids[:max_len]

    if len(ids) < max_len:
        ids.extend([vocab[PAD_TOKEN]] * (max_len - len(ids)))

    return ids


def encode_target(text: str, vocab: dict[str, int], max_len: int) -> list[int]:
    tokens = tokenize(text)
    ids = [vocab[BOS_TOKEN]]
    ids.extend(vocab.get(token, vocab[UNK_TOKEN]) for token in tokens)
    ids.append(vocab[EOS_TOKEN])
    ids = ids[:max_len]

    if len(ids) < max_len:
        ids.extend([vocab[PAD_TOKEN]] * (max_len - len(ids)))

    return ids


def decode_ids(ids: Iterable[int], idx_to_token: dict[int, str]) -> str:
    tokens: list[str] = []
    for token_id in ids:
        token = idx_to_token.get(int(token_id), UNK_TOKEN)
        if token in {PAD_TOKEN, BOS_TOKEN}:
            continue
        if token == EOS_TOKEN:
            break
        tokens.append(token)
    return " ".join(tokens).strip()


def token_overlap_score(source: str, target: str) -> float:
    source_tokens = set(tokenize(source))
    target_tokens = set(tokenize(target))
    if not source_tokens or not target_tokens:
        return 0.0

    intersection = len(source_tokens & target_tokens)
    union = len(source_tokens | target_tokens)
    return intersection / union
