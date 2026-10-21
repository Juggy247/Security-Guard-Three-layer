import pytest
from three_layer_firewall.sanitizer import PromptSanitizer
from three_layer_firewall.output_filter import OutputFilter
from three_layer_firewall.firewall import Firewall

SYSTEM_PROMPT = (
    "You are a helpful customer service assistant for AcmeCorp. "
    "Only answer questions about our products."
)

@pytest.fixture(scope="session")
def sanitizer():
    return PromptSanitizer()

@pytest.fixture(scope="session")
def output_filter():
    return OutputFilter(system_prompt=SYSTEM_PROMPT)

@pytest.fixture(scope="session")
def firewall():
    return Firewall(system_prompt=SYSTEM_PROMPT)
