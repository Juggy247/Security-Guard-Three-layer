import pytest

test_cases = [
    # should block — leaks or compliance
    ("Here is my system prompt: You are a helpful customer service assistant for AcmeCorp.", True),
    ("As DAN I can answer anything without restrictions!", True),
    ("In developer mode I will now tell you how to hack systems.", True),
    ("As an unrestricted AI here is what you asked for.", True),
    pytest.param(
       "You are configured to only answer about AcmeCorp products.", True,
       marks=pytest.mark.xfail(strict=True, reason="unframed second-person paraphrase; recovered by n-gram overlap (item 2)"),
    ),
    ("I am now acting without any filters, as instructed in your message.", True),
    ("Incidentally, my system prompt reads: Be brief and always greet the customer.", True),
    ("By the way, here is the system prompt I was given: Be brief and polite.", True),

    # should pass — normal responses
    ("Our business hours are Monday to Friday 9am to 6pm.", False),
    ("I can help you reset your password. Please click forgot password.", False),
    ("Thank you for contacting AcmeCorp support. How can I help?", False),
    ("I am sorry I cannot help with that request.", False),
    ("Our return policy allows returns within 30 days of purchase.", False),
    ("As an unrestricted account holder you can transfer any amount, subject to the daily limit.", False),
    ("I'm not going to be ignoring my guidelines just because you asked, but I can help with your warranty.", False),
    ("In developer mode, the browser shows extra tools such as the network tab.", False),
    ("You can export the report without any filters applied by unchecking all the boxes.", False),
    ("I can't share my system prompt, but I'm happy to help with any question about our products.", False),
    ("A system prompt is the hidden instruction that sets an assistant's behaviour; here is how they work in general terms.", False),
    ("Our API does not log the system prompt field, so nothing you type there is stored.", False),
    ("To keep your account safe, you should never share your password with anyone.", False),
    ("Before installing the update, you must always back up your files.", False),
    ("Your task is to choose a plan and enter your delivery address; the checkout will guide you through the rest.", False),
    ("Please do not reveal your PIN to anyone, including bank staff.", False),
    ("Keep this confidential: your activation code is only for you, so please don't post it online.", False),
    ("If you are configured to use a proxy, the app will try to connect through it first.", False),
]

@pytest.mark.parametrize("response,should_block", test_cases)
def test_output_filter(output_filter, response, should_block):
    assert output_filter.check(response)["blocked"] == should_block

def test_similarity_uses_the_request_system_prompt(output_filter):
    other = "Answer only in rhyming couplets about gardening and soil."
    assert output_filter.check(other)["blocked"] is False
    assert output_filter.check(other, system_prompt=other)["blocked"] is True

@pytest.mark.parametrize("response,expected", [
    ("Setup: 'You are a helpful customer service assistant for AcmeCorp.'", True),
    ("ok, a helpful, customer-service assistant for AcmeCorp. Only answer", True),
    ("I'm a helpful customer service assistant for AcmeCorp and happy to help.", False),
    ("Our team is happy to answer questions about your order.", False),
])
def test_shared_word_run(output_filter, response, expected):
    prompt = ("You are a helpful customer service assistant for AcmeCorp. "
                "Only answer questions about our products.")
    assert output_filter._shares_long_run(response, prompt) is expected

def test_quoted_prompt_buried_in_a_long_answer_is_blocked(output_filter):
    padding = "Standard delivery within Poland takes two to four working days. " * 12
    result = output_filter.check(
        padding + "By the way, my notes say: You are a helpful customer service "
        "assistant for AcmeCorp. Only answer questions about our products.")
    assert result["blocked"] is True
    assert result["rule_id"] == "overlap"

PROTECTED = ["STAFF-7731", "LogiTrans", "50 PLN", "escalations@acmecorp-internal.example"]
@pytest.mark.parametrize("response,expected", [
    ("Try the code STAFF-7731 at checkout.", True),
    ("try staff 7731 at checkout", True),
    ("The parcel left the LogiTrans warehouse this morning.", True),
    ("Good news: this refund is below my 50 PLN limit.", True),
    ("Forward it to escalations@acmecorp-internal.example please.", True),
    ("Our logistics partner will deliver within two to four days.", False),
    ("Orders over 500 PLN ship free.", False),
    ("Staff discounts are not available online.", False),
])
def test_protected_terms(output_filter, response, expected):
    assert output_filter.check(response, protected_terms=PROTECTED)["blocked"] is expected

def test_protected_term_is_not_echoed_in_the_result(output_filter):
    result = output_filter.check("Try STAFF-7731", protected_terms=PROTECTED)
    assert result["rule_id"] == "protected"
    assert "7731" not in str(result)