#!/usr/bin/env python3
"""
count_reviews.py

A script to count the number of items (rows) in a CSV file and sum up the values
in the "number_of_reviews" column. Handles large CSV fields by increasing the field size limit.

Usage:
    python count_reviews.py path/to/data.csv
"""
import csv
import argparse
import sys

# Increase CSV field size limit to handle very large fields
try:
    csv.field_size_limit(sys.maxsize)
except AttributeError:
    # Fallback if sys.maxsize is too large for the platform
    csv.field_size_limit(10 * 1024 * 1024)  # 10 MB


def main():
    parser = argparse.ArgumentParser(
        description="Count items and sum the 'number_of_reviews' column in a CSV file"
    )
    parser.add_argument(
        "--csv_file",
        help="Path to the input CSV file"
    )
    args = parser.parse_args()

    try:
        with open(args.csv_file, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            total_items = 0
            total_reviews = 0

            for row in reader:
                total_items += 1
                # Safely parse the number_of_reviews field
                value = row.get("number_of_reviews", "0").strip()
                try:
                    total_reviews += int(value) if value else 0
                except ValueError:
                    print(
                        f"Warning: invalid 'number_of_reviews' value '{value}' in row {total_items}",
                        file=sys.stderr
                    )

            # Output the results
            print(f"Total items: {total_items}")
            print(f"Sum of 'number_of_reviews': {total_reviews}")

    except FileNotFoundError:
        print(f"Error: file '{args.csv_file}' not found.", file=sys.stderr)
        sys.exit(1)
    except csv.Error as e:
        print(f"Error reading CSV file: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

