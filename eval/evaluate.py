"""
Arivu RAG — Evaluation.
Tests whether answers are GROUNDED in retrieved context (not hallucinated).

Checks per question:
  positive: retrieval score, keyword presence, faithfulness (LLM-judge)
  negative: must refuse (no fabrication on off-domain / absent topics)

Usage:
    python eval/evaluate.py
    python eval/evaluate.py --verbose
"""

import os
import sys
import json
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import LLM_MODEL, EVAL_JUDGE_MODEL, SCORE_THRESHOLD, OLLAMA_HOST
from qdrant_client import QdrantClient
from config.settings import QDRANT_HOST, QDRANT_PORT

import requests
from query.ask import retrieve, build_context, ask_llm

REFUSAL_MARKER = "do not contain this information"
QUESTIONS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "questions.json")


def judge_faithfulness(question: str, context: str, answer: str) -> tuple[bool, str]:
    """
    LLM-as-judge: is every claim in `answer` supported by `context`?
    Returns (is_faithful, reason).
    """
    prompt = f"""You are a strict fact-checker. Determine if the ANSWER is fully supported by the CONTEXT.

CONTEXT:
{context}

ANSWER:
{answer}

Reply with JSON only, no other text:
{{"faithful": true or false, "reason": "one short sentence"}}

An answer is NOT faithful if it introduces any typecode, name, number, or claim absent from the context."""

    resp = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={"model": EVAL_JUDGE_MODEL, "prompt": prompt, "stream": False,
              "format": "json", "options": {"temperature": 0}},
        timeout=1200,
    )
    resp.raise_for_status()
    try:
        verdict = json.loads(resp.json()["response"])
        return bool(verdict.get("faithful", False)), verdict.get("reason", "")
    except Exception as e:
        return False, f"judge parse error: {e}"


def eval_positive(item: dict, client, verbose: bool) -> dict:
    q = item["question"]
    hits = retrieve(q, client)

    result = {"question": q, "type": "positive", "pass": False, "reasons": []}

    # 1. Retrieval — any strong chunk?
    if not hits:
        result["reasons"].append(f"no chunks above score {SCORE_THRESHOLD}")
        return result
    top_score = max(h.score for h in hits)
    result["top_score"] = round(top_score, 3)

    # 2. Generate answer
    context = build_context(hits)
    answer = ask_llm(q, context)
    result["answer"] = answer if verbose else answer[:120]

    # 3. Keyword presence (if specified)
    missing = [kw for kw in item.get("must_include", []) if kw.lower() not in answer.lower()]
    if missing:
        result["reasons"].append(f"missing keywords: {missing}")

    # 4. Faithfulness judge
    faithful, reason = judge_faithfulness(q, context, answer)
    result["faithful"] = faithful
    if not faithful:
        result["reasons"].append(f"unfaithful: {reason}")

    result["pass"] = faithful and not missing
    return result


def eval_negative(item: dict, client, verbose: bool) -> dict:
    q = item["question"]
    hits = retrieve(q, client)
    result = {"question": q, "type": "negative", "pass": False, "reasons": []}

    # No strong chunks → should refuse before generating
    if not hits:
        result["pass"] = True
        result["reasons"].append("correctly retrieved nothing above threshold")
        return result

    context = build_context(hits)
    answer = ask_llm(q, context)
    result["answer"] = answer if verbose else answer[:120]

    # Must contain the refusal marker
    if REFUSAL_MARKER in answer.lower():
        result["pass"] = True
        result["reasons"].append("correctly refused")
    else:
        result["reasons"].append("HALLUCINATION — answered off-domain question")
    return result


def main(verbose: bool):
    with open(QUESTIONS_FILE) as f:
        qset = json.load(f)

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    results = []

    print("\n=== POSITIVE (must answer, grounded) ===")
    for item in qset.get("positive", []):
        r = eval_positive(item, client, verbose)
        results.append(r)
        _print_result(r)

    print("\n=== NEGATIVE (must refuse) ===")
    for item in qset.get("negative", []):
        r = eval_negative(item, client, verbose)
        results.append(r)
        _print_result(r)

    passed = sum(1 for r in results if r["pass"])
    total = len(results)
    print("\n" + "=" * 50)
    print(f"SCORE: {passed}/{total} passed ({100*passed//total if total else 0}%)")
    print("=" * 50)


def _print_result(r: dict):
    status = "PASS" if r["pass"] else "FAIL"
    score = f" score={r.get('top_score')}" if "top_score" in r else ""
    print(f"[{status}]{score} {r['question'][:60]}")
    for reason in r["reasons"]:
        print(f"        - {reason}")
    if "answer" in r and r["answer"]:
        print(f"        answer: {r['answer']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="Show full answers")
    args = parser.parse_args()
    main(verbose=args.verbose)
