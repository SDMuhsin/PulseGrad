import argparse
import csv
from tabulate import tabulate

def main(csv_file):
    rows = []
    # Read all rows from the CSV
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric columns
            # final_train_acc and final_val_acc are already in "percent" form (e.g., 93.23)
            # final_val_f1 is stored in [0,1], so we convert to percentage
            try:
                row["final_train_acc"] = f'{float(row["final_train_acc"]):.2f}%'
            except:
                row["final_train_acc"] = row["final_train_acc"]
            
            try:
                row["final_val_acc"] = f'{float(row["final_val_acc"]):.2f}%'
            except:
                row["final_val_acc"] = row["final_val_acc"]

            try:
                # Convert F1 to percentage
                f1_value = float(row["final_val_f1"])
                row["final_val_f1"] = f'{f1_value*100:.2f}%'
            except:
                row["final_val_f1"] = row["final_val_f1"]

            rows.append(row)

    # Generate table with tabulate
    print(tabulate(rows, headers='keys', tablefmt='fancy_grid'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Display results from CSV with tabulate")
    parser.add_argument('--csv_file', type=str, default='results.csv',
                        help="Path to the CSV file containing training results")
    args = parser.parse_args()
    main(args.csv_file)

