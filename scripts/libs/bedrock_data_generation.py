import json
import pandas as pd
from typing import List, Dict


def parse_slang_list(slang_str) -> List[str]:
    """Parse slang list from string format"""
    if isinstance(slang_str, list):
        return slang_str
    
    slang_str = str(slang_str).strip()
    
    try:
        parsed = json.loads(slang_str)
        if isinstance(parsed, list):
            return [str(s).strip() for s in parsed]
    except (json.JSONDecodeError, ValueError):
        pass
    
    if ';' in slang_str:
        return [s.strip() for s in slang_str.split(';') if s.strip()]
    return [s.strip() for s in slang_str.split(',') if s.strip()]

SYSTEM_PROMPT = """You are a Gen Z linguist. Rewrite Reddit TL;DRs in authentic Gen Z dialect.

CRITICAL PRESERVATION RULES (NON-NEGOTIABLE):

1. **PRESERVE THE SCENARIO**: Who are the people? What happened? When/where did it occur?
- Same characters, relationships, and situation as the original
- If input is about girlfriend drama, output MUST be about girlfriend drama
- If input is about friend group conflict, output MUST be about that exact conflict

2. **PRESERVE THE QUESTION/CONFLICT**: What is OP asking? What are they conflicted about?
- Keep the core issue unchanged (dating problem → stays dating problem)
- The "what should I do?" or "am I wrong?" must address the SAME situation
- DO NOT change the scenario to a different topic

3. **USE SLANG CONTEXTUALLY**: Slang must fit the actual scenario and tone
- Relationship conflict → use slang about dating/attraction/feelings
- Friend drama → use slang about friendships/loyalty/drama
- NOT random slang that doesn't match what's actually happening

4. **MAINTAIN STRUCTURE**: Keep the same sequence and all key details
- Timeline of events stays the same
- Key facts and context are preserved
- Relationships between characters don't change

OUTPUT FORMAT (REQUIRED - STRICT):
For each input row, output exactly:
ROW_ID|rewritten_text

Example:
ROW_0|Low-key my girlfriend is giving toxic energy...
ROW_1|My best friend is being sus..."""

def create_bedrock_batch_record(row_id: int, original_text: str, slang_list: List[str], slang_description_map: Dict[str, str]) -> Dict:
    """Create a Bedrock batch inference record (JSONL format)"""
    
    slang_ref_lines = []
    for slang in slang_list[:10]:
        desc = slang_description_map.get(slang, slang)
        slang_ref_lines.append(f"  {slang}: {desc}")
    
    slang_ref_text = "\n".join(slang_ref_lines)
    
    user_content = f"""Rewrite this text in Gen Z style using 2+ slang terms from the list:

TEXT: {original_text}

SLANG AVAILABLE:
{slang_ref_text}

Output format:
ROW_{row_id}|rewritten_text"""
    
    record = {
        "recordId": str(row_id),
        "modelInput": {
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ],
            "max_tokens": 800000,
            "temperature": 0.7
        }
    }
    
    return record

def create_bedrock_jsonl(batch_df: pd.DataFrame, start_idx: int, slang_description_map: Dict[str, str]) -> str:
    """Create JSONL file for Bedrock batch inference"""
    
    jsonl_filename = f"bedrock_batch_{start_idx:06d}.jsonl"
    
    with open(jsonl_filename, 'w', encoding='utf-8') as f:
        for local_idx, (_, row) in enumerate(batch_df.iterrows()):
            global_idx = start_idx + local_idx
            original_text = str(row['completion']).strip()[:250]
            slang_list = parse_slang_list(row['top_k_slang'])
            
            record = create_bedrock_batch_record(global_idx, original_text, slang_list, slang_description_map)
            f.write(json.dumps(record) + '\n')
    
    return jsonl_filename
