from pathlib import Path

import torch

from src.generative_model import Seq2SeqChatbot
from src.generative_nlp import (
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    decode_ids,
    encode_source,
    load_json,
    normalize_text,
    token_overlap_score,
)


class GenerativeChatbotEngine:
    def __init__(
        self,
        model_path: str = "artifacts/level2_model.pt",
        metadata_path: str = "artifacts/level2_metadata.json",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not Path(model_path).exists():
            raise FileNotFoundError("Không tìm thấy model cấp 2. Hãy chạy train_level2.py trước.")
        if not Path(metadata_path).exists():
            raise FileNotFoundError("Không tìm thấy metadata cấp 2. Hãy chạy train_level2.py trước.")

        self.metadata = load_json(metadata_path)
        self.vocab = self.metadata["vocab"]
        self.idx_to_token = {int(idx): token for idx, token in self.metadata["idx_to_token"].items()}
        self.pairs = self.metadata["pairs"]
        self.fallback_response = self.metadata["fallback_response"]
        self.min_overlap_score = self.metadata["min_overlap_score"]
        self.max_source_len = self.metadata["config"]["max_source_len"]
        self.max_target_len = self.metadata["config"]["max_target_len"]

        self.model = Seq2SeqChatbot(
            vocab_size=self.metadata["config"]["vocab_size"],
            embed_dim=self.metadata["config"]["embed_dim"],
            hidden_dim=self.metadata["config"]["hidden_dim"],
            pad_idx=self.vocab[PAD_TOKEN],
            dropout=self.metadata["config"]["dropout"],
        ).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def best_matching_pair(self, message: str):
        best_pair = None
        best_score = 0.0

        for pair in self.pairs:
            score = token_overlap_score(message, pair["input"])
            if score > best_score:
                best_score = score
                best_pair = pair

        return best_pair, best_score

    def reply(self, message: str):
        normalized = normalize_text(message)
        if not normalized:
            return {
                "mode": "fallback",
                "overlap_score": 0.0,
                "answer": self.fallback_response,
            }

        best_pair, overlap_score = self.best_matching_pair(message)
        if overlap_score < self.min_overlap_score:
            return {
                "mode": "fallback",
                "overlap_score": round(overlap_score, 4),
                "answer": self.fallback_response,
            }

        source_ids = torch.tensor(
            [encode_source(message, self.vocab, self.max_source_len)],
            dtype=torch.long,
            device=self.device,
        )

        with torch.no_grad():
            generated_ids = self.model.generate(
                source_ids,
                bos_idx=self.vocab[BOS_TOKEN],
                eos_idx=self.vocab[EOS_TOKEN],
                max_new_tokens=self.max_target_len - 1,
            )

        generated_text = decode_ids(generated_ids, self.idx_to_token)
        if len(generated_text.split()) < 3:
            return {
                "mode": "retrieved_fallback",
                "overlap_score": round(overlap_score, 4),
                "matched_input": best_pair["input"] if best_pair else None,
                "answer": best_pair["output"] if best_pair else self.fallback_response,
            }

        return {
            "mode": "generated",
            "overlap_score": round(overlap_score, 4),
            "matched_input": best_pair["input"] if best_pair else None,
            "answer": generated_text,
        }
