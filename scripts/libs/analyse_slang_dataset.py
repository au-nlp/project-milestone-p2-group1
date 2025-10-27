import pandas as pd
from typing import Dict
from IPython.display import display


class SlangAnalyzer:
    """Analyze slang datasets for duplicates and case sensitivity."""

    def __init__(self, show_header: bool = True):
        """
        Initialize the analyzer.

        Args:
            show_header: Whether to show headers in display methods
        """
        self.show_header = show_header

    @staticmethod
    def analyze(
            df: pd.DataFrame,
            column: str,
            dataset_name: str = "Dataset"
    ) -> Dict:
        """
        Analyze slang dataset for duplicates and case sensitivity.

        Args:
            df: DataFrame containing the dataset
            column: Column name with slang terms
            dataset_name: Name for display purposes

        Returns:
            Dictionary with all statistics
        """
        total_rows = len(df)
        unique_case_sensitive = df[column].nunique()
        unique_lowercase = df[column].str.lower().nunique()

        duplicates = df[column].str.lower().value_counts()
        duplicate_rate = (total_rows - unique_lowercase) / total_rows * 100

        most_common_term = duplicates.head(1).index[0]
        most_common_count = duplicates.iloc[0]
        original_cases = df[df[column].str.lower() == most_common_term][column].unique()

        results = {
            "dataset_name": dataset_name,
            "total_rows": total_rows,
            "unique_case_sensitive": unique_case_sensitive,
            "unique_lowercase": unique_lowercase,
            "duplicate_rate": duplicate_rate,
            "most_common_term": most_common_term,
            "most_common_count": most_common_count,
            "most_common_cases": original_cases,
            "top_20_duplicates": duplicates.head(20),
            "sample_unique_terms": sorted(set(df[column].str.lower()))[:20]
        }

        return results

    def _print_section_header(self, dataset_name: str) -> None:
        """Print dataset header."""
        if self.show_header:
            print(f"\n{'=' * 70}")
            print(f"DATASET ANALYSIS: {dataset_name}")
            print(f"{'=' * 70}\n")

    def _print_section_footer(self) -> None:
        """Print section footer."""
        if self.show_header:
            print(f"\n{'=' * 70}\n")

    def show_total_statistics(self, results: Dict) -> None:
        """Display total statistics."""
        self._print_section_header(results['dataset_name'])

        print("1. Total Statistics:")
        stats_df = pd.DataFrame({
            'Metric': ['Total Rows', 'Unique (Case-Sensitive)', 'Unique (Lowercase)'],
            'Count': [
                f"{results['total_rows']:,}",
                f"{results['unique_case_sensitive']:,}",
                f"{results['unique_lowercase']:,}"
            ]
        })
        display(stats_df.head())

    @staticmethod
    def show_duplication_analysis(results: Dict) -> None:
        """Display duplication analysis."""
        print("2. Duplication Analysis:")
        duplication_df = pd.DataFrame({
            'Metric': [
                'Duplication Rate (%)',
                'Most Common Term',
                'Most Common Count',
                'Original Cases'
            ],
            'Value': [
                f"{results['duplicate_rate']:.1f}%",
                f"'{results['most_common_term']}'",
                str(results['most_common_count']),
                str(list(results['most_common_cases']))
            ]
        })
        display(duplication_df.head())

    @staticmethod
    def show_top_duplicates(results: Dict, n: int = 5) -> None:
        """Display top N duplicates."""
        print(f"\n3. Top {n} Duplicates (lowercase):")
        top_duplicates_df = pd.DataFrame({
            'Term': results['top_20_duplicates'].head(n).index,
            'Count': results['top_20_duplicates'].head(n).values
        }).reset_index(drop=True)
        top_duplicates_df.index = top_duplicates_df.index + 1
        display(top_duplicates_df.head())

    @staticmethod
    def show_sample_terms(results: Dict, n: int = 5) -> None:
        """Display sample unique terms."""
        print(f"\n4. Sample of Unique Terms (first {n}):")
        sample_df = pd.DataFrame({
            'Index': range(1, n + 1),
            'Term': results['sample_unique_terms'][:n]
        })
        display(sample_df.head())

    def show_all(self, results: Dict, n_duplicates: int = 5, n_samples: int = 5) -> None:
        """
        Display all sections.

        Args:
            results: Analysis results dictionary
            n_duplicates: Number of top duplicates to show
            n_samples: Number of sample terms to show
        """
        self.show_total_statistics(results)
        self.show_duplication_analysis(results)
        self.show_top_duplicates(results, n=n_duplicates)
        self.show_sample_terms(results, n=n_samples)
        self._print_section_footer()
