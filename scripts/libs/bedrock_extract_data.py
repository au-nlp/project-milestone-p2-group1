import json
import re
import csv
from typing import Dict, List, Tuple, Optional


def extract_answer(model_output: Dict) -> Optional[str]:
    """Extract answer text from nested modelOutput structure."""
    try:
        return model_output.get("choices", [{}])[0].get("message", {}).get("content")
    except (KeyError, IndexError, TypeError):
        return None


def parse_row_pipe(content: str) -> Optional[Tuple[int, str]]:
    """Parse ROW_ID|text format."""
    match = re.match(r'ROW_\s*(\d+)\s*\|\s*(.+?)$', content, re.DOTALL)
    if match:
        try:
            return (int(match.group(1).strip()), match.group(2).strip())
        except ValueError:
            pass
    return None


def process_jsonl(
    input_file: str,
    csv_file: str
) -> List[Dict]:
    """
    Process JSONL file and extract AI answers.
    
    Args:
        input_file: Path to input JSONL file
        csv_file: Optional path to save results as CSV
    
    Returns:
        List of processed results with structure:
        {
            'record_id': str,
            'model_output': str,
            'parsed_row_id': int,
            'parsed_text': str,
            'success': bool
        }
    """
    results = []
    
    # Read and process JSONL
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            try:
                obj = json.loads(line)
                model_output_text = extract_answer(obj.get("modelOutput", {}))
                
                result = {
                    "record_id": obj.get("recordId"),
                    "model_output": model_output_text,
                    "parsed_row_id": None,
                    "parsed_text": None,
                    "success": False
                }
                
                if model_output_text:
                    parsed = parse_row_pipe(model_output_text)
                    if parsed:
                        result["parsed_row_id"], result["parsed_text"] = parsed
                        result["success"] = True
                
                results.append(result)
            except json.JSONDecodeError:
                continue
       
    # Save as CSV
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['record_id', 'parsed_row_id', 'parsed_text', 'success']
        )
        writer.writeheader()
        for result in results:
            writer.writerow({
                'record_id': result['record_id'],
                'parsed_row_id': result['parsed_row_id'],
                'parsed_text': result['parsed_text'],
                'success': result['success']
            })
    
    return results
