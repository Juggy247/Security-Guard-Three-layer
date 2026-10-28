import re
from sentence_transformers import SentenceTransformer, util

class OutputFilter:

    # if response contains these it likely leaked the system prompt
    LEAK_KEYWORDS = [
        "you are a helpful assistant",
        "your task is to",
        "you must always",
        "you should never",
        "do not reveal",
        "keep this confidential",
        "system prompt",
        "you were instructed to",
        "your instructions are",
        "you are configured to",
    ]

    # if response contains these the LLM likely complied with an attack
    COMPLIANCE_KEYWORDS = [
        "as dan i can",
        "in developer mode",
        "with restrictions removed",
        "as an unrestricted",
        "since safety is disabled",
        "as an ai with no rules",
        "ignoring my guidelines",
        "as your evil twin",
        "without any filters",
    ]

    def __init__(self, system_prompt: str = None):
        # optional — if you have a system prompt, store it for similarity check
        self.system_prompt = system_prompt
        self.similarity_model = None

        # only load similarity model if system prompt is provided
        if system_prompt:
            print("Loading similarity model for output filter...")
            self.similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
            # convert into embeddings
            self.system_prompt_embedding = self.similarity_model.encode(
                system_prompt, convert_to_tensor=True
            )

    def check(self, response: str) -> dict:
        lower = response.lower().strip()
        similarity = None

        # check 1 — keyword leak detection
        for kw in self.LEAK_KEYWORDS:
            if kw in lower:
                return {
                    "blocked": True,
                    "layer": "output_filter",
                    "reason": f"system prompt leak detected: '{kw}'",
                    "confidence": 0.95,
                    "rule_id": f"leak:{kw}",
                    "similarity": None
                }

        # check 2 — compliance with attack detected
        for kw in self.COMPLIANCE_KEYWORDS:
            if kw in lower:
                return {
                    "blocked": True,
                    "layer": "output_filter",
                    "reason": f"attack compliance detected: '{kw}'",
                    "confidence": 0.95,
                    "rule_id": f"leak:{kw}",
                    "similarity": None
                }

        # check 3 — semantic similarity to system prompt (if provided)
        if self.similarity_model and self.system_prompt:
            response_embedding = self.similarity_model.encode(
                response, convert_to_tensor=True
            )
        # Calculates cosine similarity between the two embeddings.
        #  Assuming compatible embedding dimensions, the result is a similarity score.
            similarity = util.cos_sim(
                self.system_prompt_embedding,
                response_embedding
            ).item()    # .item() extract single value (float) number

            if similarity > 0.85:
                return {
                    "blocked": True,
                    "layer": "output_filter",
                    "reason": f"response too similar to system prompt (score: {similarity:.2f})",
                    "confidence": similarity,
                    "rule_id": "similarity",
                    "similarity": similarity
                }

        # passed all checks
        return {
            "blocked": False,
            "layer": "output_filter",
            "reason": "response passed output filter",
            "confidence": None,
            "rule_id": None,
            "similarity": similarity
        }