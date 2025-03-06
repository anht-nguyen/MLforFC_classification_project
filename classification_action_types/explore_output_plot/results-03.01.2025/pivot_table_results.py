import pandas as pd
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the CSV data
file_path = os.path.join(script_dir, "performance_errors.csv")

def generate_pivot_table(measure, input_file="performance_errors.csv", output_file=None):
    """
    Generates a pivot table for a given measure from the performance errors dataset.

    Parameters:
    - measure (str): The measure to filter (e.g., "AUC_errors", "Acc_errors").
    - input_file (str): Path to the input CSV file.
    - output_file (str, optional): Path to save the output CSV file. If None, it defaults to '{measure}_pivot_table.csv'.

    Returns:
    - pd.DataFrame: The pivot table DataFrame.
    """
    # Load the CSV data
    df = pd.read_csv(input_file)

    # Filter the data for the specified measure
    df_filtered = df[df["Measure"] == measure].copy()

    # Format as Mean ± CI
    df_filtered["Formatted Value"] = df_filtered.apply(lambda row: f"{row['Mean']:.3f} ± {row['95% CI Error']:.3f}", axis=1)

    # Pivot Table: Rows as Feature Categories (FC), Columns as Models
    pivot_df = df_filtered.pivot(index="FC_metrics", columns="Model", values="Formatted Value")

    # Define the output file path
    if output_file is None:
        output_file = f"{measure}_pivot_table.csv"

    # Save the pivot table to a CSV file
    pivot_df.to_csv(os.path.join(script_dir,output_file))

    return pivot_df

# Example usage
# generate_pivot_table("AUC_errors", input_file="performance_errors.csv", output_file="AUC_errors_pivot_table.csv")

generate_pivot_table("Acc_errors", input_file=file_path, output_file="Acc_errors_pivot_table.csv")
