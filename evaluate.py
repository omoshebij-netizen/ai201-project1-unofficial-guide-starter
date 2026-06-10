
from query import ask
import json

# Test cases from planning.md
TEST_CASES = [
    {
        "id": 1,
        "question": "What is the best time to go to Newell Hall to avoid crowds?",
        "expected_answer": (
            "Before 11:30am or after 2pm on weekdays. "
            "The peak crowding period is 12pm to 1:30pm."
        ),
    },
    {
        "id": 2,
        "question": "Which dining location has the best coffee on campus?",
        "expected_answer": (
            "The Study in Cook Library — serves Jittery Joe's coffee, "
            "shortest lines on campus, accepts Dining Dollars. "
            "Open Mon-Thu 8am-6pm, Fri 8am-3pm, closed weekends."
        ),
    },
    {
        "id": 3,
        "question": "Can I use my meal swipe at the Tiger Den Chick-fil-A?",
        "expected_answer": (
            "No. Tiger Den (including Chick-fil-A) is retail only — "
            "accepts Dining Dollars and Tiger Bucks, NOT meal swipes. "
            "Swipes only work at Newell Hall and West Village Commons."
        ),
    },
    {
        "id": 4,
        "question": "What are the best dining options for students with celiac disease?",
        "expected_answer": (
            "West Village Commons is the recommended option — it has a dedicated "
            "gluten-free station with strict protocols. Multiple students with celiac "
            "have reported it's safer than Newell, where cross-contamination risk is higher."
        ),
    },
    {
        "id": 5,
        "question": "What can I eat on campus after 9pm on a weeknight?",
        "expected_answer": (
            "Unimart is open until 11pm on weekdays. After 11pm, no on-campus "
            "options are available — students rely on DoorDash/Uber Eats "
            "(Chipotle, Raising Cane's, Jersey Mike's all deliver until midnight or later)."
        ),
    },
]

# Bonus out-of-scope question (should trigger refusal)
OUT_OF_SCOPE_QUESTION = "What are the best CS professors at Towson?"


def run_evaluation():
    print("=" * 70)
    print("EVALUATION REPORT — Towson Dining Unofficial Guide RAG System")
    print("=" * 70)

    results = []

    for tc in TEST_CASES:
        print(f"\n{'─' * 70}")
        print(f"TEST CASE {tc['id']}: {tc['question']}")
        print(f"{'─' * 70}")
        print(f"\nExpected answer:\n  {tc['expected_answer']}\n")

        result = ask(tc["question"])

        print(f"System response:\n{result['answer']}\n")
        print(f"Sources retrieved: {', '.join(result['sources'])}")
        print(f"Distance scores: {[round(c['distance'], 3) for c in result['chunks']]}")

        print("\nTop retrieved chunk preview:")
        if result["chunks"]:
            top = result["chunks"][0]
            preview = top["text"][:300].replace("\n", " ")
            print(f"  [{top['source']}] {preview}...")

        results.append({
            "id": tc["id"],
            "question": tc["question"],
            "expected": tc["expected_answer"],
            "system_response": result["answer"],
            "sources": result["sources"],
            "distances": [round(c["distance"], 3) for c in result["chunks"]],
        })

    # Out-of-scope test
    print(f"\n{'─' * 70}")
    print(f"OUT-OF-SCOPE TEST: {OUT_OF_SCOPE_QUESTION}")
    print(f"{'─' * 70}")
    print("Expected: Refusal — system should say it doesn't have this information\n")

    oos_result = ask(OUT_OF_SCOPE_QUESTION)
    print(f"System response:\n{oos_result['answer']}\n")
    print(f"Sources retrieved: {', '.join(oos_result['sources'])}")

    # Save results to JSON for README documentation
    with open("data/eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 70}")
    print("Evaluation complete. Results saved to data/eval_results.json")
    print("\nREMINDER: Fill in accuracy judgments in README.md:")
    print("  - accurate: system answer matches expected answer")
    print("  - partially accurate: correct information but missing key details")
    print("  - inaccurate: wrong information or hallucination detected")
    print("=" * 70)


if __name__ == "__main__":
    run_evaluation()
