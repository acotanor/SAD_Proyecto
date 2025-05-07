#!/usr/bin/env python3
"""
Flatten a CSV file's `review_scores` column into separate columns.

Usage:
    python break_review_scores.py --input_csv input.csv --output_csv output.csv

This script reads `input.csv`, parses the `review_scores` column (which should contain
string representations of Python lists or dicts), and expands each element into its own
column. The resulting flattened dataframe is written to `output.csv`.
"""
import argparse
import ast
import sys
import pandas as pd

def parse_review_scores(series: pd.Series) -> pd.DataFrame:
    """
    Parse and expand the `review_scores` series into a DataFrame of separate columns.
    Detects whether entries are lists or dicts to choose the expansion method.
    """
    # Convert strings to Python objects
    parsed = series.apply(ast.literal_eval)

    # Drop any nulls before type inspection
    sample = parsed.dropna()
    if sample.empty:
        raise ValueError("No non-null entries found in `review_scores` column.")

    first = sample.iloc[0]

    if isinstance(first, dict):
        # Dicts: use json_normalize to get keys as column names
        expanded = pd.json_normalize(parsed)
    elif isinstance(first, (list, tuple)):
        # Lists/Tuples: expand by position
        expanded = parsed.apply(pd.Series)
        expanded.columns = [f"review_score_{i+1}" for i in expanded.columns]
    else:
        raise ValueError(
            f"Unsupported type for review_scores: {type(first)}. "
            "Expected list/tuple or dict."
        )
    return expanded


def main():
    parser = argparse.ArgumentParser(
        description="Flatten the review_scores column in a CSV file."
    )
    parser.add_argument(
        "--input_csv", help="Path to the input CSV file containing `review_scores` column."
    )
    parser.add_argument(
        "--output_csv", help="Path where the flattened CSV file will be saved."
    )
    args = parser.parse_args()

    # Read the CSV
    df = pd.read_csv(args.input_csv)

    if 'review_scores' not in df.columns:
        print("Error: `review_scores` column not found in input CSV.", file=sys.stderr)
        sys.exit(1)

    # Expand the review_scores
    try:
        scores_df = parse_review_scores(df['review_scores'])
    except Exception as e:
        print(f"Error parsing `review_scores`: {e}", file=sys.stderr)
        sys.exit(1)

    # Drop original column and combine
    df_flat = pd.concat(
        [df.drop(['review_scores'], axis=1), scores_df], axis=1
    )

    # Save to output CSV
    df_flat.to_csv(args.output_csv, index=False)
    print(f"Flattened CSV saved to {args.output_csv}")

if __name__ == "__main__":
    main()
