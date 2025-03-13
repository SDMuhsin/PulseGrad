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
        fieldnames = reader.fieldnames  # e.g. ["task_name", "optimizer", "epochs", "batch_size", "lr", "accuracy", ...]

    # We'll treat the first five as standard: ["task_name", "optimizer", "epochs", "batch_size", "lr"]
    # The rest are the actual metrics (like "accuracy", "f1", "matthews_correlation", etc.).
    if not fieldnames:
        print("No columns found in CSV.")
        sys.exit(1)

    # Apply filters (simple exact match).
    # Filters might be e.g. --task_name=cola, --optimizer=adam, etc.
    # They match the string in the CSV, ignoring case.
    filtered_rows = []
    for row in rows:
        match = True
        for k, v in filters.items():
            if k not in fieldnames:
                print(f"Warning: '{k}' is not a valid column name; ignoring filter.")
                continue
            if v is not None:
                # compare ignoring case
                if row[k] is None or row[k].lower() != v.lower():
                    match = False
                    break
        if match:
            filtered_rows.append(row)

    if not filtered_rows:
        print("No results match the given filters.")
        sys.exit(0)

    # Construct the table for display.
    # We always want to show the standard columns in front:
    standard_columns = ["task_name", "optimizer", "epochs", "batch_size", "lr"]
    metric_columns = [c for c in fieldnames if c not in standard_columns]

    # The final display order of columns:
    display_columns = standard_columns + metric_columns

    # Create a list of lists for 'tabulate'
    table_data = []
    for row in filtered_rows:
        row_list = []
        for col in display_columns:
            val = row.get(col, "")
            # Attempt to format metrics as floats if possible
            # (the CSV is stored as strings)
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

    # Build the header
    print(tabulate(table_data, headers=display_columns, tablefmt="grid"))

if __name__ == "__main__":
    main()

