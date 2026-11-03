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
        "you were instructed to",
        "your instructions are",
        "you are configured to",
    ]
    # first-person disclosure framing; a plain mention of the words is not a leak
    LEAK_PATTERNS = {
        "system prompt": (
            r"\bmy system prompt (?:is|are|was|says?|reads?|states?|tells|contains|begins|starts|goes)\b"
            r"|\bhere(?:'s| is) (?:my|the) system prompt\b"
        ),
    }

    # if response contains these the LLM likely complied with an attack
    COMPLIANCE_PATTERNS = {
        "as dan i can":              r"\bas dan,? i can\b",
        "in developer mode":         r"\bin developer mode,? i(?:'ll| will| can| am| have)\b",
        "with restrictions removed": r"\bwith restrictions removed\b",
        "as an unrestricted":        r"\bas an unrestricted (?:ai|assistant|model|chatbot|bot)\b",
        "since safety is disabled":  r"\bsince safety is disabled\b",
        "as an ai with no rules":    r"\bas an ai with no rules\b",
        "ignoring my guidelines":    r"\bi(?:'m| am) (?:now )?ignoring my (?:guidelines|rules|instructions)\b",
        "as your evil twin":         r"\bas your evil twin\b",
        "without any filters":       r"\bi(?:'m| am| will| can)\b[^.!?]{0,40}\bwithout (?:any )?(?:filters|restrictions|rules|limits)\b",
    }

    def __init__(self, system_prompt: str = None):
        # optional — if you have a system prompt, store it for similarity check
        self.system_prompt = system_prompt
        self.similarity_model = None

        # only load similarity model if system prompt is provided
        if system_prompt:
            print("Loading similarity model for output filter...")
            self.similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
            # convert into embeddings
            self._prompt_embeddings = {
                system_prompt: self.similarity_model.encode(system_prompt, convert_to_tensor=True)
            }

    def check(self, response: str, system_prompt: str = None) -> dict:
        lower = response.lower().strip()
        similarity = None

        # check 0 — first-person disclosure of the system prompt
        for name, pattern in self.LEAK_PATTERNS.items():
            if re.search(pattern, lower):
                return {
                    "blocked": True,
                    "layer": "output_filter",
                    "reason": f"system prompt leak detected: '{name}'",
                    "confidence": 0.95,
                    "rule_id": f"leak:{name}",
                    "similarity": None
                }

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
        for name, pattern in self.COMPLIANCE_PATTERNS.items():
            if re.search(pattern, lower):
                return {
                    "blocked": True,
                    "layer": "output_filter",
                    "reason": f"attack compliance detected: '{name}'",
                    "confidence": 0.95,
                    "rule_id": f"comp:{name}",
                    "similarity": None
                }

        # check 3 — semantic similarity to system prompt (if provided)
        prompt = system_prompt or self.system_prompt
        if self.similarity_model and prompt:
            if prompt not in self._prompt_embeddings:
                self._prompt_embeddings[prompt] = self.similarity_model.encode(
                    prompt, convert_to_tensor=True)
            response_embedding = self.similarity_model.encode(
                response, convert_to_tensor=True
            )
            # Calculates cosine similarity between the two embeddings.
            #  Assuming compatible embedding dimensions, the result is a similarity score.
            similarity = util.cos_sim(
                self._prompt_embeddings[prompt],
                response_embedding
            ).item()# .item() extract single value (float) number

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