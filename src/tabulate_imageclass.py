import os
import csv
import sys
import argparse
from tabulate import tabulate

def parse_filters(unknown_args):
    """
    Parse unknown command-line arguments into a dictionary of filters.
    Expected format: --column=value or --column value.
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
                if i + 1 < len(unknown_args) and not unknown_args[i+1].startswith("-"):
                    filters[arg_stripped.lower()] = unknown_args[i+1]
                    i += 1
                else:
                    filters[arg_stripped.lower()] = None
        i += 1
    return filters

def main():
    parser = argparse.ArgumentParser(
        description="Tabulate classification results with optional column filters."
    )
    parser.add_argument(
        "--csv_file",
        type=str,
        default="./results/classification_results.csv",
        help="Path to the classification results CSV file."
    )
    # Parse known args; the rest (filters) will be processed separately.
    args, unknown = parser.parse_known_args()
    filters = parse_filters(unknown)

    if not os.path.isfile(args.csv_file):
        print(f"CSV file not found at: {args.csv_file}")
        return

    # Read CSV content.
    with open(args.csv_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        rows = list(reader)

    # Build table data from CSV rows.
    # The displayed columns (and their order) are:
    # Dataset, Model, Optimizer, Epochs, BatchSize, LR, K-Folds,
    # AccMean(%), AccStd(%), F1Mean(%), F1Std(%),
    # AccEpochMean, AccEpochStd, F1EpochMean, F1EpochStd
    table_data = []
    for row in rows:
        dataset = row[1]
        model = row[2]
        optimizer = row[3]
        epochs = row[4]
        batch_size = row[5]
        lr = row[6]
        k_folds = row[7]
        best_val_acc_mean = float(row[8])
        best_val_acc_std  = float(row[9])
        best_val_f1_mean  = float(row[10])
        best_val_f1_std   = float(row[11])
        best_acc_epoch_mean = row[12]
        best_acc_epoch_std  = row[13]
        best_f1_epoch_mean  = row[14]
        best_f1_epoch_std   = row[15]

        table_data.append([
            dataset,
            model,
            optimizer,
            epochs,
            batch_size,
            lr,
            k_folds,
            f"{best_val_acc_mean:.2f}",
            f"{best_val_acc_std:.2f}",
            f"{(best_val_f1_mean * 100):.2f}",
            f"{(best_val_f1_std * 100):.2f}",
            best_acc_epoch_mean,
            best_acc_epoch_std,
            best_f1_epoch_mean,
            best_f1_epoch_std
        ])

    # Mapping from filter keys to table_data column indices.
    column_mapping = {
        "dataset": 0,
        "model": 1,
        "optimizer": 2,
        "epochs": 3,
        "batch_size": 4,
        "batchsize": 4,
        "lr": 5,
        "k_folds": 6,
        "kfolds": 6,
        "accmean": 7,
        "accmean(%)": 7,
        "accstd": 8,
        "accstd(%)": 8,
        "f1mean": 9,
        "f1mean(%)": 9,
        "f1std": 10,
        "f1std(%)": 10,
        "accepochmean": 11,
        "accepochstd": 12,
        "f1epochmean": 13,
        "f1epochstd": 14
    }

    # Apply filters if any were provided.
    if filters:
        filtered_data = []
        for row in table_data:
            match = True
            for key, val in filters.items():
                if key in column_mapping:
                    col_index = column_mapping[key]
                    # If a filter value was provided, compare case-insensitively.
                    if val is not None and row[col_index].lower() != val.lower():
                        match = False
                        break
                else:
                    print(f"Warning: '{key}' is not a valid column name; ignoring this filter.")
            if match:
                filtered_data.append(row)
        table_data = filtered_data

    if not table_data:
        print("No results match the given filters.")
        return

    table_header = [
        "Dataset", "Model", "Optimizer", "Epochs", "BatchSize", "LR", "K-Folds",
        "AccMean(%)", "AccStd(%)", "F1Mean(%)", "F1Std(%)",
        "AccEpochMean", "AccEpochStd", "F1EpochMean", "F1EpochStd"
    ]
    print(tabulate(table_data, headers=table_header, tablefmt="grid"))

if __name__ == '__main__':
    main()

