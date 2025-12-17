import json
import pandas as pd
import re
from pathlib import Path

def parse_evaluation_score(response_text: str) -> dict:
    """Parse the expert annotator response"""
    try:
        response_text = response_text.strip()

        meaning_match = re.search(r"Meaning:\s*(\d+)/5", response_text)
        slang_match = re.search(r"Slang:\s*(\d+)/5", response_text)
        reddit_match = re.search(r"Reddit:\s*(\d+)/5", response_text)

        meaning_score = int(meaning_match.group(1)) if meaning_match else 0
        slang_score = int(slang_match.group(1)) if slang_match else 0
        reddit_score = int(reddit_match.group(1)) if reddit_match else 0

        notes_match = re.search(r"Notes:\s*(.*)", response_text)
        notes = notes_match.group(1).strip() if notes_match else "No notes"

        return {
            "meaning": meaning_score,
            "slang": slang_score,
            "reddit": reddit_score,
            "overall": meaning_score + slang_score + reddit_score,
            "notes": notes,
        }
    except Exception:
        return {
            "meaning": 0,
            "slang": 0,
            "reddit": 0,
            "overall": 0,
            "notes": "Parse error",
        }


def parse_evaluation_results(jsonl_file: str,
                             n_t5: int = 3000,
                             n_bart: int = 3000) -> pd.DataFrame:
    """
    Parse single Bedrock batch results file.

    Assumes:
      - Lines 0 .. n_t5-1  are T5
      - Lines n_t5 .. n_t5+n_bart-1 are BART
      - recordId is present but not needed to infer model.
    """

    print(f"Loading {jsonl_file}...")
    results = []

    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            try:
                result = json.loads(line)

                record_id = result.get("recordId")
                if not record_id:
                    continue

                # Derive model from position:
                # 0..n_t5-1 → t5, n_t5..n_t5+n_bart-1 → bart
                if line_num < n_t5:
                    model = "t5"
                else:
                    model = "bart"

                # Extract response text from modelOutput.choices[0].message.content
                response_text = ""
                mo = result.get("modelOutput", {})
                choices = mo.get("choices", [])
                if choices:
                    msg = choices[0].get("message", {})
                    response_text = msg.get("content", "") or ""

                scores = parse_evaluation_score(response_text)

                results.append(
                    {
                        "model": model,
                        "record_id": record_id,
                        "meaning": scores["meaning"],
                        "slang": scores["slang"],
                        "reddit": scores["reddit"],
                        "overall": scores["overall"],
                        "notes": scores["notes"],
                    }
                )

            except Exception as e:
                print(f"Line {line_num+1}: {e}")
                continue

    print(f"Parsed {len(results)} rows")
    return pd.DataFrame(results)
