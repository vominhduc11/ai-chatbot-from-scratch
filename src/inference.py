import random
from pathlib import Path

import torch

from src.model import ChatbotNet
from src.nlp import encode_text, load_json, PAD_TOKEN


class ChatbotEngine:
    def __init__(
        self,
        model_path: str = "artifacts/model.pt",
        metadata_path: str = "artifacts/metadata.json"
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not Path(model_path).exists():
            raise FileNotFoundError("Không tìm thấy model. Hãy chạy train.py trước.")

        if not Path(metadata_path).exists():
            raise FileNotFoundError("Không tìm thấy metadata. Hãy chạy train.py trước.")

        self.metadata = load_json(metadata_path)
        self.vocab = self.metadata["vocab"]
        self.label_to_idx = self.metadata["label_to_idx"]
        self.idx_to_label = {int(k): v for k, v in self.metadata["idx_to_label"].items()}
        self.responses_by_intent = self.metadata["responses_by_intent"]
        self.max_len = self.metadata["config"]["max_len"]
        self.fallback_response = self.metadata["fallback_response"]
        self.threshold = self.metadata["threshold"]

        self.model = ChatbotNet(
            vocab_size=self.metadata["config"]["vocab_size"],
            embed_dim=self.metadata["config"]["embed_dim"],
            hidden_dim=self.metadata["config"]["hidden_dim"],
            num_classes=self.metadata["config"]["num_classes"],
            pad_idx=self.vocab[PAD_TOKEN],
            dropout=self.metadata["config"]["dropout"]
        ).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def predict_intent(self, message: str):
        input_ids = encode_text(message, self.vocab, self.max_len)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)

        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = torch.softmax(logits, dim=-1)
            confidence, pred_idx = torch.max(probs, dim=-1)

        intent = self.idx_to_label[pred_idx.item()]
        return intent, confidence.item()

    def reply(self, message: str):
        intent, confidence = self.predict_intent(message)

        if confidence < self.threshold:
            return {
                "intent": "fallback",
                "confidence": round(confidence, 4),
                "answer": self.fallback_response
            }

        answer = random.choice(self.responses_by_intent[intent])
        return {
            "intent": intent,
            "confidence": round(confidence, 4),
            "answer": answer
        }