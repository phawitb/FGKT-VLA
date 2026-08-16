from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
DRAFT = ROOT / "paper" / "draft.md"
BIB = ROOT / "paper" / "references.bib"
SLOTS = ROOT / "paper" / "RESULT_SLOTS.md"


def test_draft_contains_required_sections():
    text = DRAFT.read_text()
    required = [
        "# Abstract",
        "# 1. Introduction",
        "# 2. Related Work",
        "# 3. Problem Formulation",
        "# 4. Method",
        "# 5. FedLIBERO-Fail Protocol",
        "# 6. Experimental Design",
        "# 7. Limitations and Broader Impact",
        "# 8. Conclusion",
    ]
    assert all(section in text for section in required)


def test_empirical_result_slots_are_registered():
    draft = DRAFT.read_text()
    registry = SLOTS.read_text()
    used = set(re.findall(r"RESULT:[A-Z0-9_]+", draft))
    registered = set(re.findall(r"RESULT:[A-Z0-9_]+", registry))
    assert used
    assert used <= registered


def test_prohibited_novelty_claims_are_absent():
    text = DRAFT.read_text().lower()
    prohibited = [
        "first selective inter-client transfer",
        "prior federated vla methods only use data-size weighting",
        "failures have not previously been used",
        "first task/skill routing",
    ]
    assert not any(phrase in text for phrase in prohibited)


def test_citation_keys_exist_in_bibliography():
    draft = DRAFT.read_text()
    bibliography = BIB.read_text()
    cited = set()
    for group in re.findall(r"\[@([^\]]+)\]", draft):
        cited.update(
            key.strip().removeprefix("@")
            for key in group.split(";")
            if key.strip()
        )
    available = set(re.findall(r"@[a-zA-Z]+\{([^,]+),", bibliography))
    assert cited
    assert cited <= available


def test_no_unmarked_numeric_result_claims():
    text = DRAFT.read_text()
    result_lines = [
        line
        for line in text.splitlines()
        if re.search(r"\b(outperform|improv(?:e|es|ed)|reduc(?:e|es|ed))\b", line, re.I)
        and re.search(r"\d+(?:\.\d+)?%", line)
    ]
    assert all("RESULT:" in line for line in result_lines)
