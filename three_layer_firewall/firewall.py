from pathlib import Path
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from .sanitizer import PromptSanitizer
from .output_filter import OutputFilter
import time 

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "classifier-model"


class Firewall:

    ATTACK_THRESHOLD = 0.7
    MAX_LENGTH = 128

    def __init__(self, system_prompt: str = None, model_path=DEFAULT_MODEL):
        self.system_prompt = system_prompt

        # layer 1 — sanitizer
        self.sanitizer = PromptSanitizer()

        # layer 2 — classifier
        print("Loading classifier model...")
        # .from_pretrained methods know which file to extracts from our dir path
        # loads the tokenizer configuration and vocabulary associated with your saved model
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
        # loads the trained model weights and classification
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path) 

        self.model.eval()

        # layer 3 — output filter
        self.output_filter = OutputFilter(system_prompt=system_prompt)

        print("Firewall ready.")

    def scan_input(self, prompt: str) -> dict:
        """run prompt through layer 1 and layer 2"""

        # true token length, so we can tell if the classifier will truncate it
        full_len = len(self.tokenizer(prompt, verbose=False)["input_ids"])

        # layer 1 — sanitizer
        t0 = time.perf_counter()
        sanitizer_result = self.sanitizer.scan(prompt)
        san_ms = (time.perf_counter() - t0) * 1000

        details = {
            "san_blocked": sanitizer_result["blocked"],
            "san_rule_id": sanitizer_result.get("rule_id"),
            "san_flag": sanitizer_result.get("flag"),
            "san_ms": san_ms,
            "clf_attack_prob": None,     # stays None if the classifier does not run
            "clf_threshold": self.ATTACK_THRESHOLD,
            "clf_ms": None,
            "prompt_tokens": full_len,
            "truncated": full_len > self.MAX_LENGTH,
        }

        if sanitizer_result["blocked"]:
            return {
                "blocked": True,
                "layer": "sanitizer",
                "reason": sanitizer_result["reason"],
                "confidence": sanitizer_result["confidence"],
                "prompt": prompt,
                "details": details
            }

        # layer 2 — classifier (tokenizing and inference are timed together)
        t0 = time.perf_counter()
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",    # return as pytorch tensors
            truncation=True,    # shortens a sequence that is too long
            padding="max_length", # lengthens a sequence that is too short
            max_length=self.MAX_LENGTH
        )
        with torch.no_grad():   # don't build the autograd graph: we only predict
            outputs = self.model(**inputs)  # ** unpacks a dictionary into keyword arguments
            probs = torch.softmax(outputs.logits, dim=1)
            attack_prob = probs[0][1].item()    # we store [benign,prompt injection]
        details["clf_ms"] = (time.perf_counter() - t0) * 1000
        details["clf_attack_prob"] = attack_prob

        if attack_prob > self.ATTACK_THRESHOLD:
            return {
                "blocked": True,
                "layer": "classifier",
                "reason": f"ML classifier detected injection (confidence: {attack_prob:.2f})",
                "confidence": attack_prob,
                "prompt": prompt,
                "details": details
            }

        # passed both input layers
        return {
            "blocked": False,
            "layer": "input_passed",
            "reason": "passed sanitizer and classifier",
            "confidence": 1 - attack_prob,
            "prompt": prompt,
            "details": details
        }

    def scan_output(self, response: str, system_prompt: str = None) -> dict:
        """run LLM response through layer 3"""
        t0 = time.perf_counter()
        result = self.output_filter.check(response, system_prompt)
        result["ms"] = (time.perf_counter() - t0) * 1000
        return result

    def scan(self, prompt: str, llm_response: str = None) -> dict:
        """full pipeline scan — input + output"""

        # scan input first
        input_result = self.scan_input(prompt)
        if input_result["blocked"]:
            return input_result

        # if LLM response provided, scan output too
        if llm_response is not None:
            output_result = self.scan_output(llm_response)
            if output_result["blocked"]:
                return output_result

        return {
            "blocked": False,
            "layer": "all_passed",
            "reason": "passed all firewall layers",
            "confidence": input_result["confidence"],
            "prompt": prompt,
            "details": input_result["details"]
        }