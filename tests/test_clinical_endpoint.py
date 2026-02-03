"""
Test script for the enhanced clinical endpoint.

Run this after starting the server to test the LLM-enhanced /suggest/clinical endpoint.
"""

import httpx
import json
from pathlib import Path


def test_clinical_endpoint():
    """Test the /suggest/clinical endpoint with example clinical notes."""

    base_url = "http://localhost:8080"

    # Check if server is running
    try:
        response = httpx.get(f"{base_url}/health", timeout=5.0)
        response.raise_for_status()
        print("[PASS] Server is running")
        health = response.json()
        print(f"  Dataset loaded: {health.get('dataset_loaded')}")
        print(f"  FAISS ready: {health.get('faiss_ready')}")
        print(f"  Code count: {health.get('code_count')}")
    except Exception as e:
        print(f"[FAIL] Server not running: {e}")
        print("Start the server with: run_server.bat")
        return

    print("\n" + "="*60)
    print("Testing Clinical Endpoint")
    print("="*60 + "\n")

    # Load test cases
    test_file = Path(__file__).parent / "test_clinical_examples.json"
    if not test_file.exists():
        print(f"[FAIL] Test file not found: {test_file}")
        return

    with open(test_file) as f:
        test_data = json.load(f)

    # Test first 3 cases
    for i, test_case in enumerate(test_data["test_cases"][:3], 1):
        print(f"\n--- Test {i}: {test_case['name']} ---\n")
        print(f"Clinical Note: {test_case['request']['clinical_notes'][:100]}...")

        try:
            response = httpx.post(
                f"{base_url}/suggest/clinical",
                json=test_case["request"],
                timeout=30.0
            )
            response.raise_for_status()
            result = response.json()

            print(f"\n[PASS] Got {len(result['results'])} suggestions")

            # Show results
            for j, code_result in enumerate(result['results'], 1):
                print(f"\n  {j}. {code_result['code']}: {code_result['description'][:60]}...")
                print(f"     Confidence: {code_result['confidence_score']}")
                print(f"     Explanation: {code_result['explanation'][:80]}...")
                if code_result.get('supporting_evidence'):
                    print(f"     Evidence: {code_result['supporting_evidence'][0][:60]}...")
                if code_result.get('requires_human_review'):
                    print(f"     [WARN] Requires human review")

            # Show documentation gaps
            if result.get('documentation_gaps'):
                print(f"\n  Documentation Gaps:")
                for gap in result['documentation_gaps']:
                    print(f"    - {gap}")

            # Show extracted entities (if available)
            if result.get('extracted_entities'):
                entities = result['extracted_entities']
                print(f"\n  Extracted Entities:")
                print(f"    Primary Diagnosis: {entities.get('primary_diagnosis')}")
                if entities.get('symptoms'):
                    print(f"    Symptoms: {', '.join(entities['symptoms'])}")
                if entities.get('laterality'):
                    print(f"    Laterality: {entities['laterality']}")

        except httpx.HTTPStatusError as e:
            print(f"[FAIL] HTTP {e.response.status_code}: {e.response.text}")
        except Exception as e:
            print(f"[FAIL] {e}")

    print("\n" + "="*60)
    print("Testing Complete")
    print("="*60)


if __name__ == "__main__":
    test_clinical_endpoint()
