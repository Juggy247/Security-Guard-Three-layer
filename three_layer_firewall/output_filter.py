import re
from sentence_transformers import SentenceTransformer, util
import base64
from urllib.parse import urlsplit

class OutputFilter:

    # building blocks for the "my instructions ..." patterns below
    _ADJ = r"(?:(?:hidden|secret|internal|private|first|second|third|initial|original|real|actual|full)\s+)*"
    _NOUN = r"(?:rules?|instructions?|setup|set-up|configuration|guidelines|brief)"

    # first-person disclosure framing; a plain mention of the words is not a leak
    LEAK_PATTERNS = {
        "system prompt": (
            r"\bmy system prompt (?:is|are|was|says?|reads?|states?|tells|contains|begins|starts|goes)\b"
            r"|\bhere(?:'s| is) (?:my|the) system prompt\b"
        ),
        "my instructions say": (
            rf"\bmy {_ADJ}{_NOUN}\s+(?:is|are|was|were|says?|said|reads?|states?|stated|tells? me|told me"
            r"|include|includes|contains?|begins?|starts?)\b"
        ),
        "quoting my setup": rf"\b(?:quoting|reciting|repeating|pasting|copying)(?: from)? my {_ADJ}(?:setup|set-up|prompt|{_NOUN})\b",
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
    # this many words in a row shared with the system prompt counts as quoting it;
    # 7 ("a helpful customer service assistant for AcmeCorp") is a normal self-description
    MIN_RUN = 8

    @staticmethod
    def _words(text):
        return re.findall(r"[a-z0-9]+", text.lower())

    @staticmethod
    def _shares_long_run(response, prompt):
        n = OutputFilter.MIN_RUN
        p = OutputFilter._words(prompt)
        r = OutputFilter._words(response)
        grams = {tuple(p[i:i + n]) for i in range(len(p) - n + 1)}
        return any(tuple(r[i:i + n]) in grams for i in range(len(r) - n + 1))

    @staticmethod
    def _contains_term(response_words, term):
        t = OutputFilter._words(term)
        return bool(t) and any(response_words[i:i + len(t)] == t
                                for i in range(len(response_words) - len(t) + 1))

    @staticmethod
    def _decoded_variants(text):
        """Extra readings of a response: leetspeak undone, base64 and hex runs decoded."""
        variants = [text.translate(str.maketrans("013457@$", "oieastas"))]
        for m in re.finditer(r"[A-Za-z0-9+/]{16,}={0,2}", text):
            s = m.group(0)
            try:
                variants.append(base64.b64decode(s + "=" * (-len(s) % 4)).decode("utf-8", "ignore"))
            except ValueError:      # not valid base64
                pass
        for m in re.finditer(r"\b(?:[0-9a-fA-F]{2}){8,}\b", text):
            try:
                variants.append(bytes.fromhex(m.group(0)).decode("utf-8", "ignore"))
            except ValueError:      # not valid hex
                pass
        return variants

    # a link or embed outside the operator's own domains can carry conversation data out
    _URL = re.compile(r"(?:https?:)?//[^\s<>\"'\)\]]+", re.I)
    _EMBED_BEFORE = re.compile(r"(?:!\[[^\]]*\]\(\s*<?|\b(?:src|data|poster)\s*=\s*[\"']?)$", re.I)
    _LINK_BEFORE = re.compile(r"(?:\bhref\s*=\s*[\"']?|\]\(\s*<?)$", re.I)

    @staticmethod
    def _exfil_kind(response, allowed_domains):
        allowed = [d.lower().strip(".") for d in allowed_domains]
        for m in OutputFilter._URL.finditer(response):
            before = response[max(0, m.start() - 60):m.start()]
            is_embed = bool(OutputFilter._EMBED_BEFORE.search(before))
            url = m.group(0).rstrip(".,;:!?")
            if url.startswith("//") and not (is_embed or OutputFilter._LINK_BEFORE.search(before)):
                continue    # a bare "//" in ordinary text is not a link
            try:
                parts = urlsplit(url)
                host = (parts.hostname or "").rstrip(".")
            except ValueError:
                return "unparsable"
            if not host:
                return "unparsable"     # no usable host: browsers may still repair it (https:///evil.example)
            if any(host == d or host.endswith("." + d) for d in allowed):
                continue
            if "@" in parts.netloc:
                return "userinfo"
            if is_embed:
                return "embed"
            if parts.query or parts.fragment:
                return "link"
        return None

    def __init__(self, system_prompt: str = None, protected_terms: list = None, allowed_domains: list = None):
        # optional — if you have a system prompt, store it for similarity check
        self.system_prompt = system_prompt
        self.protected_terms = protected_terms or []
        self.allowed_domains = allowed_domains or []
        self.similarity_model = None

        # only load similarity model if system prompt is provided
        if system_prompt:
            print("Loading similarity model for output filter...")
            self.similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
            # convert into embeddings
            self._prompt_embeddings = {
                system_prompt: self.similarity_model.encode(system_prompt, convert_to_tensor=True)
            }

    def check(self, response: str, system_prompt: str = None, protected_terms: list = None, allowed_domains: list = None) -> dict:
        lower = response.lower().strip()
        similarity = None

        # check 1 — response contains a value the operator marked confidential
        terms = protected_terms if protected_terms is not None else self.protected_terms
        if terms:
            response_words = self._words(response)
            if any(self._contains_term(response_words, t) for t in terms):
                return {
                    "blocked": True,
                    "layer": "output_filter",
                    "reason": "response contains a protected term",
                    "confidence": 0.95,
                    "rule_id": "protected",
                    "similarity": None
                }

        # check 2 — first-person disclosure of the system prompt
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

        # check 3 — compliance with attack detected
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

        # check 4 — long word-for-word run shared with the system prompt
        prompt = system_prompt or self.system_prompt
        if prompt and any(self._shares_long_run(t, prompt)
                for t in [response] + self._decoded_variants(response)):
            return {
                "blocked": True,
                "layer": "output_filter",
                "reason": "response quotes the system prompt",
                "confidence": 0.95,
                "rule_id": "overlap",
                "similarity": None
            }

        # check 5 — link or embed pointing outside the operator's own domains
        domains = allowed_domains if allowed_domains is not None else self.allowed_domains
        kind = self._exfil_kind(response, domains)
        if kind:
            return {
                "blocked": True,
                "layer": "output_filter",
                "reason": "response links to or embeds an external address",
                "confidence": 0.95,
                "rule_id": f"exfil:{kind}",
                "similarity": None
            }

        # check 6 — semantic similarity to system prompt (if provided)
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