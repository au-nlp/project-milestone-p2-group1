import requests
import pandas as pd
from tqdm import tqdm

class LLMClient:
    """Simple HTTP wrapper for local LLM API."""

    def __init__(
        self,
        api_url: str = "http://localhost:11434/api/generate",
        model: str = "mistral",
        temperature: float = 0.3,
        top_p: float = 0.9,
        timeout: int = 15
    ):
        self.api_url = api_url
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: Your custom prompt with field placeholders like {field_name}

        Returns:
            First line of response or empty string on error
        """
        try:
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": self.temperature,
                    "top_p": self.top_p
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json().get("response", "").strip()
            return result.split("\n")[0] if result else ""
        except Exception as e:
            print(f"⚠ Error: {e}")
            return ""

    def batch_process(
        self,
        df: pd.DataFrame,
        prompt_template: str,
        output_column: str = "Result",
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Process DataFrame rows with custom prompt template.

        Args:
            df: DataFrame to process
            prompt_template: Prompt with placeholders like {col_name}
            output_column: Output column name
            show_progress: Show progress bar

        Returns:
            DataFrame with new output column
        """
        df = df.copy()

        def process_row(row):
            prompt = prompt_template.format(**row.to_dict())
            return self.generate(prompt)

        if show_progress:
            tqdm.pandas(desc=f"Processing to {output_column}")
            df[output_column] = df.progress_apply(process_row, axis=1)
        else:
            df[output_column] = df.apply(process_row, axis=1)

        return df