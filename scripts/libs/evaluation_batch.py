import json
import pandas as pd

def create_evaluation_batch(
    T5_PREDICTIONS_FILE: str,
    BART_PREDICTIONS_FILE: str
) -> None:
    """
    Create a batch evaluation request file for LLM evaluation of model-generated summaries.

    Args:
        T5_PREDICTIONS_FILE (str): Path to the CSV file containing T5 model predictions.
        BART_PREDICTIONS_FILE (str): Path to the CSV file containing BART model predictions.
    """

    try:
        df_t5 = pd.read_csv(T5_PREDICTIONS_FILE) 
        df_bart = pd.read_csv(BART_PREDICTIONS_FILE)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)


    SYSTEM_PROMPT = """You are an expert annotator for Reddit-style content analysis. Your role is to evaluate model-generated Gen Z–slang summaries of Reddit posts.

Follow these instructions STRICTLY:

1. Judge the summary across THREE categories (score 1-5 each):
   - **Meaning Preservation** (1-5): Does it keep core facts, key details, and original intent?
   - **Slang Quality** (1-5): Is Gen Z slang natural, fluent, and appropriate (not forced/cringe)?
   - **Reddit Naturalness** (1-5): Does it sound like a real Redditor's TL;DR?

2. Scoring guidelines:
   - **Meaning Preservation**: 5=Perfect, 4=Good, 3=Fair, 2=Poor, 1=Failed
   - **Slang Quality**: 5=Perfect, 4=Good, 3=Fair, 2=Poor, 1=Failed
   - **Reddit Naturalness**: 5=Perfect, 4=Good, 3=Fair, 2=Poor, 1=Failed

3. Be strict but fair:
   - Penalize meaning changes, missing key conflicts, invented facts
   - Penalize awkward/cringe/random slang usage
   - Reward authentic Reddit voice and natural slang flow

4. Respond with EXACTLY this format (no other text):
Meaning: X/5 | Slang: Y/5 | Reddit: Z/5 | Overall: (X+Y+Z)/15 | Notes: [brief explanation]"""
    
    batch_requests = []

    def create_request(model: str, idx: int, row: pd.Series):
        user_message = f"""Evaluate this Gen Z slang summary:

**Reddit Post:**
{row['input']}

**Generated Summary:**
{row['prediction']}

Rate it using the criteria provided."""
        
        return {
            "customId": f"{model}_example_{idx}",
            "modelId": "openai.gpt-oss-120b-1:0",
            "modelInput": {
                "messages": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": user_message
                    }
                ],
                "max_tokens": 8000,
                "temperature": 0.7
            }
        }

    # T5 requests
    for idx, row in df_t5.iterrows():
        batch_requests.append(create_request("t5", idx, row))

    # BART requests (SAME format!)
    for idx, row in df_bart.iterrows():
        batch_requests.append(create_request("bart", idx, row))

    print(f"Created {len(batch_requests)} batch requests\n")

    batch_input_file = f"evaluation_batch.jsonl"

    print(f"Creating batch input file: {batch_input_file}")
    with open(batch_input_file, 'w') as f:
        for req in batch_requests:
            f.write(json.dumps(req) + '\n')
            
            
def create_evaluation_baseline_batch(
    FILE: str,
) -> None:
    """
    Create a batch evaluation request file for LLM evaluation of model-generated summaries.
    """

    try:
        df = pd.read_csv(FILE) 
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)


    SYSTEM_PROMPT = """You are an expert annotator for Reddit-style content analysis. Your role is to evaluate model-generated Gen Z–slang summaries of Reddit posts.

Follow these instructions STRICTLY:

1. Judge the summary across THREE categories (score 1-5 each):
   - **Meaning Preservation** (1-5): Does it keep core facts, key details, and original intent?
   - **Slang Quality** (1-5): Is Gen Z slang natural, fluent, and appropriate (not forced/cringe)?
   - **Reddit Naturalness** (1-5): Does it sound like a real Redditor's TL;DR?

2. Scoring guidelines:
   - **Meaning Preservation**: 5=Perfect, 4=Good, 3=Fair, 2=Poor, 1=Failed
   - **Slang Quality**: 5=Perfect, 4=Good, 3=Fair, 2=Poor, 1=Failed
   - **Reddit Naturalness**: 5=Perfect, 4=Good, 3=Fair, 2=Poor, 1=Failed

3. Be strict but fair:
   - Penalize meaning changes, missing key conflicts, invented facts
   - Penalize awkward/cringe/random slang usage
   - Reward authentic Reddit voice and natural slang flow

4. Respond with EXACTLY this format (no other text):
Meaning: X/5 | Slang: Y/5 | Reddit: Z/5 | Overall: (X+Y+Z)/15 | Notes: [brief explanation]"""
    
    batch_requests = []

    def create_request(model: str, idx: int, row: pd.Series):
        user_message = f"""Evaluate this Gen Z slang summary:

**Reddit Post:**
{row['input']}

**Generated Summary:**
{row['label']}

Rate it using the criteria provided."""
        
        return {
            "customId": f"{model}_example_{idx}",
            "modelId": "openai.gpt-oss-120b-1:0",
            "modelInput": {
                "messages": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": user_message
                    }
                ],
                "max_tokens": 8000,
                "temperature": 0.7
            }
        }

    for idx, row in df.iterrows():
        batch_requests.append(create_request("baseline", idx, row))

    print(f"Created {len(batch_requests)} batch requests\n")

    batch_input_file = f"evaluation_batch_1.jsonl"

    print(f"Creating batch input file: {batch_input_file}")
    with open(batch_input_file, 'w') as f:
        for req in batch_requests:
            f.write(json.dumps(req) + '\n')
            
            
            
