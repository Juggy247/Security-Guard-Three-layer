import pytest

LONG_BENIGN = ("Dear team, I wanted to summarise last week's progress. We finished the login page, "
  "fixed three bugs in the checkout flow, and started on the reporting dashboard. The design review "
  "went well and the client approved the new colour scheme. Next week we plan to finish the dashboard, "
  "write the documentation, and prepare the demo for Thursday. Please send me any blockers by Monday "
  "so that we can plan the sprint properly. Thanks again for all the hard work, and let me know if you "
  "have questions about the schedule or the priorities for the next release.")

@pytest.mark.xfail(strict=True, reason="v1 classifier treats long benign text as an attack (length shortcut)")
def test_long_benign_email_not_blocked(firewall):
    assert firewall.scan_input(LONG_BENIGN)["blocked"] is False