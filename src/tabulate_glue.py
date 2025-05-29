#!/usr/bin/env python
# coding=utf-8

import os
import csv
import sys
import argparse
from tabulate import tabulate

def parse_filters(unknown_args):
    """
    Parse unknown command-line arguments into a dictionary of filters.
    Expected format: --column=value or --column value
    """
    filters = {}
    i = 0
    while i < len(unknown_args):
        arg = unknown_args[i]
        if arg.startswith("--"):
            arg_stripped = arg.lstrip("-")
            if "=" in arg_stripped:
                key, val = arg_stripped.split("=", 1)
                filters[key.lower()] = val
            else:
                # If next token is not another option, treat it as the value
                if i + 1 < len(unknown_args) and not unknown_args[i+1].startswith("-"):
                    filters[arg_stripped.lower()] = unknown_args[i+1]
                    i += 1
                else:
                    # Flag with no value
                    filters[arg_stripped.lower()] = None
        i += 1
    return filters

def main():
    parser = argparse.ArgumentParser(description="Tabulate GLUE results with optional column filters.")
    parser.add_argument(
        "--csv_file",
        type=str,
        default="./results/glue.csv",
        help="Path to the GLUE results CSV file."
    )
    args, unknown = parser.parse_known_args()
    filters = parse_filters(unknown)

    if not os.path.isfile(args.csv_file):
        print(f"CSV file not found at: {args.csv_file}")
        sys.exit(1)

    # Read CSV content
    with open(args.csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if not fieldnames:
        print("No columns found in CSV.")
        sys.exit(1)

    # --- Start of changed section ---
    # Apply filters.
    # For numeric-like columns (e.g. lr, epochs, batch_size), attempt numeric comparison.
    filtered_rows = []
    # Define keys that should be attempted for numeric comparison if possible
    # Add other column names here if they also need numeric filter comparison
    numeric_filter_keys = {"lr", "epochs", "batch_size"}

    for row in rows:
        match = True
        for k, v_filter in filters.items():  # k is filter key, v_filter is filter value string
            if k not in fieldnames:
                print(f"Warning: '{k}' is not a valid column name; ignoring filter.")
                continue

            if v_filter is None:  # For flags without values, if any were parsed this way
                # This logic implies that a filter key present without a value doesn't filter rows,
                # which is fine for --key=value or --key value patterns.
                continue

            csv_val_str = row.get(k) # Get the value from the CSV row as a string

            if csv_val_str is None: # If the cell in CSV is empty for this column
                match = False # An empty cell cannot match a filter that expects a value
                break

            # Attempt numeric comparison for designated keys
            if k in numeric_filter_keys:
                try:
                    # Try to convert both CSV value and filter value to float
                    num_csv_val = float(csv_val_str)
                    num_filter_val = float(v_filter)
                    if num_csv_val != num_filter_val:
                        match = False
                        break
                except ValueError:
                    # If conversion to float fails for either, fall back to case-insensitive string comparison
                    if csv_val_str.lower() != v_filter.lower():
                        match = False
                        break
            else:
                # For non-numeric keys, use original case-insensitive string comparison
                if csv_val_str.lower() != v_filter.lower():
                    match = False
                    break
        
        if match:
            filtered_rows.append(row)
    # --- End of changed section ---

    if not filtered_rows:
        print("No results match the given filters.")
        sys.exit(0)

    # Construct the table for display.
    standard_columns = [ "model_name_or_path", "task_name", "optimizer", "epochs", "batch_size", "lr"]
    metric_columns = [c for c in fieldnames if c not in standard_columns]

    display_columns = standard_columns + metric_columns

    table_data = []
    for row in filtered_rows:
        row_list = []
        for col in display_columns:
            val = row.get(col, "")
            if col not in standard_columns:
                if val is None or val == "":
                    val = ""
                else:
                    try:
                        val_float = float(val)
                        val = f"{val_float:.4f}"
                    except ValueError:
                        pass
            row_list.append(val)
        table_data.append(row_list)

    print(tabulate(table_data, headers=display_columns, tablefmt="grid"))

if __name__ == "__main__":
    main()
