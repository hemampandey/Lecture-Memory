"""
Measures retrieval quality

Metrics:
  - Retrieval Precision: are the retrieved chunks actually relevant?
  - Temporal Accuracy: do time-filtered queries respect session constraints?
  - Contradiction Recall: does contradiction detection fire when it should?
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from retrieval.retriever import TemporalRetriever, RetrievalConfig
from config import config
from logger import logger
from retrieval.contradiction import check_contradiction

retriever = TemporalRetriever()

# Format: query, expected_session_range, expected_topic_keywords
RETRIEVAL_EVAL_SET = [
    {
        "query": "what is self-attention",
        "expected_sessions": [2, 3, 4],
        "expected_keywords": ["attention", "transformer", "query", "key", "value"],
    },
    {
        "query": "how does gradient descent work",
        "expected_sessions": [1, 2],
        "expected_keywords": ["gradient", "loss", "learning rate", "optimize"],
    },
    {
        "query": "what is a convolutional neural network",
        "expected_sessions": [3, 4, 5],
        "expected_keywords": ["convolution", "filter", "pooling", "CNN"],
    },
]

TEMPORAL_EVAL_SET = [
    {"query": "transformers", "session_from": 5, "should_not_contain_session_lt": 5},
    {"query": "neural networks", "session_to": 3, "should_not_contain_session_gt": 3},
]

CONTRADICTION_EVAL_SET = [
    ("recommended optimizer for training", True),
    ("what activation function to use", True),
    ("what is a tensor", False),
]


def eval_retrieval_precision() -> float:
    scores = []
    print("\nRetrieval Precision")

    for item in RETRIEVAL_EVAL_SET:
        results = retriever.retrieve(item["query"], RetrievalConfig(top_k=5))
        hits = 0
        for r in results:
            text = r["payload"]["text"].lower()
            if any(kw.lower() in text for kw in item["expected_keywords"]):
                hits += 1
        precision = hits / len(results) if results else 0
        scores.append(precision)
        print(f"  Query: '{item['query']}' → precision={precision:.2f} ({hits}/{len(results)} chunks relevant)")

    avg = sum(scores) / len(scores) if scores else 0
    print(f"  Average Retrieval Precision: {avg:.2f}")
    return avg


def eval_temporal_filtering() -> float:
    correct = 0
    total = 0
    print("\nTemporal Filtering Accuracy")

    for item in TEMPORAL_EVAL_SET:
        config = RetrievalConfig(
            top_k=5,
            session_from=item.get("session_from"),
            session_to=item.get("session_to"),
        )
        results = retriever.retrieve(item["query"], config)

        for r in results:
            session = r["payload"]["session_number"]
            total += 1
            violation = False

            if "should_not_contain_session_lt" in item and session < item["should_not_contain_session_lt"]:
                violation = True
            if "should_not_contain_session_gt" in item and session > item["should_not_contain_session_gt"]:
                violation = True

            if not violation:
                correct += 1
            else:
                print(f"  ✗ Filter violation: query='{item['query']}' returned session {session}")

    accuracy = correct / total if total else 1.0
    print(f"  Temporal Filter Accuracy: {accuracy:.2f} ({correct}/{total})")
    return accuracy


def eval_contradiction_detection() -> dict:
    print("\nContradiction Detection")
    results = []

    for query, expect_contradiction in CONTRADICTION_EVAL_SET:
        result = check_contradiction(query)
        found = result["found"]
        correct = found == expect_contradiction
        results.append(correct)
        status = "✓" if correct else "✗"
        print(f"  {status} '{query}' → found={found}, expected={expect_contradiction}")
        if found:
            print(f"     → {result.get('explanation', '')}")

    accuracy = sum(results) / len(results) if results else 0
    print(f"  Contradiction Detection Accuracy: {accuracy:.2f}")
    return accuracy


def run_all():
    print("Running evaluation suite...\n")
    retrieval_precision = eval_retrieval_precision()
    temporal_accuracy = eval_temporal_filtering()
    contradiction_accuracy = eval_contradiction_detection()

    print("\n=== SUMMARY ===")
    print(f"  Retrieval Precision:          {retrieval_precision:.2f}")
    print(f"  Temporal Filtering Accuracy:  {temporal_accuracy:.2f}")
    print(f"  Contradiction Detection:       {contradiction_accuracy:.2f}")

    with open("evaluation/results.json", "w") as f:
        json.dump({
            "retrieval_precision": retrieval_precision,
            "temporal_accuracy": temporal_accuracy,
            "contradiction_accuracy": contradiction_accuracy,
        }, f, indent=2)

    print("\n✓ Results saved to evaluation/results.json")

if __name__ == "__main__":
    run_all()
