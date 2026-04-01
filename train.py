import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.model import ChatbotNet
from src.nlp import (
    load_json,
    save_json,
    build_samples,
    build_vocab,
    encode_text,
    PAD_TOKEN
)


SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "intents.json"
ARTIFACT_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "model.pt"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"

MAX_LEN = 12
EMBED_DIM = 128
HIDDEN_DIM = 128
DROPOUT = 0.2
BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 1e-3
PATIENCE = 8
THRESHOLD = 0.60

FALLBACK_RESPONSE = "Mình chưa hiểu rõ câu hỏi này. Bạn hãy hỏi lại ngắn gọn hơn hoặc bổ sung dữ liệu rồi train lại."


class IntentDataset(Dataset):
    def __init__(self, samples, vocab, label_to_idx, max_len):
        self.samples = samples
        self.vocab = vocab
        self.label_to_idx = label_to_idx
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        input_ids = encode_text(text, self.vocab, self.max_len)
        label_id = self.label_to_idx[label]

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(label_id, dtype=torch.long)
        )


def split_data(samples, train_ratio=0.8):
    shuffled = samples[:]
    random.shuffle(shuffled)

    if len(shuffled) < 2:
        return shuffled, shuffled

    split_idx = max(1, int(len(shuffled) * train_ratio))
    split_idx = min(split_idx, len(shuffled) - 1)

    train_samples = shuffled[:split_idx]
    val_samples = shuffled[split_idx:]

    return train_samples, val_samples


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for input_ids, labels in dataloader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            logits = model(input_ids)
            loss = criterion(logits, labels)

            total_loss += loss.item() * input_ids.size(0)

            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def main():
    print("=== TRAINING START ===")
    print(f"BASE_DIR      : {BASE_DIR}")
    print(f"DATA_PATH     : {DATA_PATH}")
    print(f"ARTIFACT_DIR  : {ARTIFACT_DIR}")
    print(f"MODEL_PATH    : {MODEL_PATH}")
    print(f"METADATA_PATH : {METADATA_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DEVICE        : {device}")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {DATA_PATH}")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    intents_data = load_json(str(DATA_PATH))
    samples, responses_by_intent = build_samples(intents_data)

    print(f"TOTAL SAMPLES : {len(samples)}")

    if not samples:
        raise ValueError("Không có sample nào trong intents.json")

    train_samples, val_samples = split_data(samples, train_ratio=0.8)

    print(f"TRAIN SAMPLES : {len(train_samples)}")
    print(f"VAL SAMPLES   : {len(val_samples)}")

    vocab = build_vocab(train_samples, min_freq=1)
    print(f"VOCAB SIZE    : {len(vocab)}")

    labels = sorted(list({label for _, label in samples}))
    if not labels:
        raise ValueError("Không có label nào trong dữ liệu")

    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}

    print(f"LABELS        : {labels}")

    train_dataset = IntentDataset(train_samples, vocab, label_to_idx, MAX_LEN)
    val_dataset = IntentDataset(val_samples, vocab, label_to_idx, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = ChatbotNet(
        vocab_size=len(vocab),
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=len(labels),
        pad_idx=vocab[PAD_TOKEN],
        dropout=DROPOUT
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    patience_counter = 0
    best_saved = False

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for input_ids, labels_batch in train_loader:
            input_ids = input_ids.to(device)
            labels_batch = labels_batch.to(device)

            optimizer.zero_grad()

            logits = model(input_ids)
            loss = criterion(logits, labels_batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item() * input_ids.size(0)
            preds = torch.argmax(logits, dim=1)
            correct_train += (preds == labels_batch).sum().item()
            total_train += labels_batch.size(0)

        train_loss = total_train_loss / max(total_train, 1)
        train_acc = correct_train / max(total_train, 1)

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            torch.save(
                {"model_state_dict": model.state_dict()},
                str(MODEL_PATH)
            )
            best_saved = True
            print(f"[SAVED MODEL] {MODEL_PATH}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping.")
                break

    if not best_saved:
        torch.save(
            {"model_state_dict": model.state_dict()},
            str(MODEL_PATH)
        )
        print(f"[FORCE SAVED MODEL] {MODEL_PATH}")

    metadata = {
        "vocab": vocab,
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
        "responses_by_intent": responses_by_intent,
        "fallback_response": FALLBACK_RESPONSE,
        "threshold": THRESHOLD,
        "config": {
            "max_len": MAX_LEN,
            "embed_dim": EMBED_DIM,
            "hidden_dim": HIDDEN_DIM,
            "dropout": DROPOUT,
            "vocab_size": len(vocab),
            "num_classes": len(labels)
        }
    }

    save_json(str(METADATA_PATH), metadata)
    print(f"[SAVED METADATA] {METADATA_PATH}")

    print(f"MODEL EXISTS    : {MODEL_PATH.exists()}")
    print(f"METADATA EXISTS : {METADATA_PATH.exists()}")
    print("=== TRAINING DONE ===")


if __name__ == "__main__":
    main()