import os
import csv
from tabulate import tabulate

def main():
    results_dir = "./results"
    csv_file = os.path.join(results_dir, "classification_results.csv")

    if not os.path.isfile(csv_file):
        print(f"CSV file not found at: {csv_file}")
        return

    # Read all rows from the CSV
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader, None)  # The CSV header
        rows = list(reader)

    # The CSV header in your main script is:
    # 0: timestamp
    # 1: dataset
    # 2: model
    # 3: optimizer
    # 4: epochs
    # 5: batch_size
    # 6: lr
    # 7: k_folds
    # 8: best_val_acc_mean  (already in %)
    # 9: best_val_acc_std   (already in %)
    # 10: best_val_f1_mean  (range [0..1])
    # 11: best_val_f1_std   (range [0..1])
    # 12: best_acc_epoch_mean
    # 13: best_acc_epoch_std
    # 14: best_f1_epoch_mean
    # 15: best_f1_epoch_std

    # Define the table headers we want to display (skipping timestamp for clarity)
    table_header = [
        "Dataset", "Model", "Optimizer", "Epochs", "BatchSize", "LR", "K-Folds",
        "AccMean(%)", "AccStd(%)", "F1Mean(%)", "F1Std(%)",
        "AccEpochMean", "AccEpochStd", "F1EpochMean", "F1EpochStd"
    ]

    table_data = []
    for row in rows:
        # Convert relevant columns to float and handle percentage formatting
        dataset = row[1]
        model = row[2]
        optimizer = row[3]
        epochs = row[4]
        batch_size = row[5]
        lr = row[6]
        k_folds = row[7]

        best_val_acc_mean = float(row[8])   # already in percentage form
        best_val_acc_std  = float(row[9])   # already in percentage form
        best_val_f1_mean  = float(row[10])  # in [0..1], convert to %
        best_val_f1_std   = float(row[11])  # in [0..1], convert to %

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
            f"{best_val_acc_mean:.2f}",      # e.g. 78.43
            f"{best_val_acc_std:.2f}",       # e.g.  2.50
            f"{(best_val_f1_mean * 100):.2f}",  # e.g. 0.87 -> 87.00
            f"{(best_val_f1_std  * 100):.2f}",  # e.g. 0.02 -> 2.00
            best_acc_epoch_mean,
            best_acc_epoch_std,
            best_f1_epoch_mean,
            best_f1_epoch_std
        ])

    # Print the nicely tabulated data
    print(tabulate(table_data, headers=table_header, tablefmt="grid"))

if __name__ == '__main__':
    main()

