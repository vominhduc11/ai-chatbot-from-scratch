import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.generative_model import Seq2SeqChatbot
from src.generative_nlp import (
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    build_vocab_from_pairs,
    decode_ids,
    encode_source,
    encode_target,
    load_json,
    save_json,
)

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

BASE_DIR = Path(__file__).resolve().parent
INTENTS_PATH = BASE_DIR / "data" / "intents.json"
ARTIFACT_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "level2_model.pt"
METADATA_PATH = ARTIFACT_DIR / "level2_metadata.json"

MAX_SOURCE_LEN = 20
MAX_TARGET_LEN = 28
EMBED_DIM = 128
HIDDEN_DIM = 256
DROPOUT = 0.2
BATCH_SIZE = 16
EPOCHS = 35
LEARNING_RATE = 1e-3
PATIENCE = 8
TEACHER_FORCING_RATIO = 0.6
MIN_OVERLAP_SCORE = 0.15
FALLBACK_RESPONSE = "Mình chưa chắc về câu này. Bạn hãy hỏi ngắn gọn hơn hoặc mở rộng thêm dữ liệu để mình học tốt hơn."


class PairDataset(Dataset):
    def __init__(self, pairs, vocab, max_source_len, max_target_len):
        self.pairs = pairs
        self.vocab = vocab
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        source_ids = encode_source(pair["input"], self.vocab, self.max_source_len)
        target_ids = encode_target(pair["output"], self.vocab, self.max_target_len)
        return (
            torch.tensor(source_ids, dtype=torch.long),
            torch.tensor(target_ids, dtype=torch.long),
        )


def build_level2_pairs(intents_data: dict) -> list[dict]:
    pairs: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for intent in intents_data["intents"]:
        for pattern in intent["patterns"]:
            for response in intent["responses"]:
                key = (pattern.strip().lower(), response.strip().lower())
                if key in seen:
                    continue
                seen.add(key)
                pairs.append({
                    "input": pattern.strip(),
                    "output": response.strip(),
                    "intent": intent["tag"],
                })

    return pairs


def split_pairs(pairs: list[dict], train_ratio: float = 0.85):
    shuffled = pairs[:]
    random.shuffle(shuffled)

    if len(shuffled) < 2:
        return shuffled, shuffled

    split_idx = max(1, int(len(shuffled) * train_ratio))
    split_idx = min(split_idx, len(shuffled) - 1)
    return shuffled[:split_idx], shuffled[split_idx:]


def evaluate(model, dataloader, criterion, device, vocab_size):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for source_ids, target_ids in dataloader:
            source_ids = source_ids.to(device)
            target_ids = target_ids.to(device)

            logits = model(source_ids, target_ids, teacher_forcing_ratio=0.0)
            loss = criterion(
                logits.reshape(-1, vocab_size),
                target_ids[:, 1:].reshape(-1)
            )

            valid_tokens = (target_ids[:, 1:] != criterion.ignore_index).sum().item()
            total_loss += loss.item() * max(valid_tokens, 1)
            total_tokens += max(valid_tokens, 1)

    return total_loss / max(total_tokens, 1)


def preview_generation(model, pair, vocab, idx_to_token, device):
    source_ids = torch.tensor(
        [encode_source(pair["input"], vocab, MAX_SOURCE_LEN)],
        dtype=torch.long,
        device=device
    )

    with torch.no_grad():
        generated_ids = model.generate(
            source_ids,
            bos_idx=vocab[BOS_TOKEN],
            eos_idx=vocab[EOS_TOKEN],
            max_new_tokens=MAX_TARGET_LEN - 1
        )

    return decode_ids(generated_ids, idx_to_token)


def main():
    print("=== LEVEL 2 TRAINING START ===")
    print(f"INTENTS_PATH   : {INTENTS_PATH}")
    print(f"MODEL_PATH     : {MODEL_PATH}")
    print(f"METADATA_PATH  : {METADATA_PATH}")

    if not INTENTS_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy dữ liệu intent: {INTENTS_PATH}")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DEVICE         : {device}")

    intents_data = load_json(str(INTENTS_PATH))
    pairs = build_level2_pairs(intents_data)
    print(f"TOTAL PAIRS    : {len(pairs)}")

    train_pairs, val_pairs = split_pairs(pairs)
    print(f"TRAIN PAIRS    : {len(train_pairs)}")
    print(f"VAL PAIRS      : {len(val_pairs)}")

    vocab = build_vocab_from_pairs(train_pairs)
    idx_to_token = {idx: token for token, idx in vocab.items()}
    print(f"VOCAB SIZE     : {len(vocab)}")

    train_dataset = PairDataset(train_pairs, vocab, MAX_SOURCE_LEN, MAX_TARGET_LEN)
    val_dataset = PairDataset(val_pairs, vocab, MAX_SOURCE_LEN, MAX_TARGET_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = Seq2SeqChatbot(
        vocab_size=len(vocab),
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        pad_idx=vocab[PAD_TOKEN],
        dropout=DROPOUT,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=vocab[PAD_TOKEN])
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_train_loss = 0.0
        total_tokens = 0

        for source_ids, target_ids in train_loader:
            source_ids = source_ids.to(device)
            target_ids = target_ids.to(device)

            optimizer.zero_grad()
            logits = model(
                source_ids,
                target_ids,
                teacher_forcing_ratio=TEACHER_FORCING_RATIO
            )
            loss = criterion(
                logits.reshape(-1, len(vocab)),
                target_ids[:, 1:].reshape(-1)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            valid_tokens = (target_ids[:, 1:] != vocab[PAD_TOKEN]).sum().item()
            total_train_loss += loss.item() * max(valid_tokens, 1)
            total_tokens += max(valid_tokens, 1)

        train_loss = total_train_loss / max(total_tokens, 1)
        val_loss = evaluate(model, val_loader, criterion, device, len(vocab))

        preview = preview_generation(model, val_pairs[0], vocab, idx_to_token, device) if val_pairs else ""
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | preview={preview!r}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({"model_state_dict": model.state_dict()}, str(MODEL_PATH))
            print(f"[SAVED MODEL] {MODEL_PATH}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping.")
                break

    metadata = {
        "vocab": vocab,
        "idx_to_token": {str(idx): token for idx, token in idx_to_token.items()},
        "pairs": pairs,
        "fallback_response": FALLBACK_RESPONSE,
        "min_overlap_score": MIN_OVERLAP_SCORE,
        "config": {
            "max_source_len": MAX_SOURCE_LEN,
            "max_target_len": MAX_TARGET_LEN,
            "embed_dim": EMBED_DIM,
            "hidden_dim": HIDDEN_DIM,
            "dropout": DROPOUT,
            "vocab_size": len(vocab),
        },
    }
    save_json(str(METADATA_PATH), metadata)
    print(f"[SAVED METADATA] {METADATA_PATH}")
    print("=== LEVEL 2 TRAINING DONE ===")


if __name__ == "__main__":
    main()
