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


class GenerativeChatbotEngineV2:
    def __init__(
        self,
        model_path: str = "artifacts/level2_stable_model.pt",
        metadata_path: str = "artifacts/level2_stable_metadata.json",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not Path(model_path).exists():
            raise FileNotFoundError("Không tìm thấy model cấp 2 ổn định. Hãy chạy train_level2_stable.py trước.")
        if not Path(metadata_path).exists():
            raise FileNotFoundError("Không tìm thấy metadata cấp 2 ổn định. Hãy chạy train_level2_stable.py trước.")

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
        normalized_message = normalize_text(message)

        for pair in self.pairs:
            if normalize_text(pair["input"]) == normalized_message:
                return pair, 1.0, True

        best_pair = None
        best_score = 0.0

        for pair in self.pairs:
            score = token_overlap_score(message, pair["input"])
            if score > best_score:
                best_score = score
                best_pair = pair

        return best_pair, best_score, False

    def _generate_text(self, message: str) -> str:
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

        return decode_ids(generated_ids, self.idx_to_token).strip()

    def reply(self, message: str):
        normalized = normalize_text(message)
        if not normalized:
            return {
                "mode": "fallback",
                "overlap_score": 0.0,
                "exact_match": False,
                "answer": self.fallback_response,
            }

        best_pair, overlap_score, exact_match = self.best_matching_pair(message)

        if best_pair is None or overlap_score < self.min_overlap_score:
            return {
                "mode": "fallback",
                "overlap_score": round(overlap_score, 4),
                "exact_match": exact_match,
                "answer": self.fallback_response,
            }

        generated_text = self._generate_text(message)
        generated_word_count = len(generated_text.split())
        reference_overlap = token_overlap_score(generated_text, best_pair["output"])
        generic_signatures = {
            "mình là chatbot hỏi đáp theo",
            "mình là chatbot ai tự train",
            "mình là chatbot hỏi đáp đơn giản",
        }
        is_generic_bad = any(generated_text.startswith(signature) for signature in generic_signatures)

        if exact_match:
            return {
                "mode": "retrieved_exact",
                "overlap_score": round(overlap_score, 4),
                "exact_match": exact_match,
                "matched_input": best_pair["input"],
                "answer": best_pair["output"],
            }

        if (
            generated_word_count < 4
            or reference_overlap < 0.2
            or generated_text == normalized
            or is_generic_bad
        ):
            return {
                "mode": "retrieved_fallback",
                "overlap_score": round(overlap_score, 4),
                "exact_match": exact_match,
                "matched_input": best_pair["input"],
                "answer": best_pair["output"],
            }

        return {
            "mode": "generated",
            "overlap_score": round(overlap_score, 4),
            "exact_match": exact_match,
            "matched_input": best_pair["input"],
            "answer": generated_text,
        }
