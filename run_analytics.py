from google.cloud import bigquery
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def run_analytics(table_id, output_dir):
    client = bigquery.Client()
    query = f"""
        SELECT disease_label, COUNT(*) as image_count
        FROM `{table_id}`
        GROUP BY disease_label
    """
    df = client.query(query).to_dataframe()
    plt.figure(figsize=(8, 6))
    df.plot(kind="bar", x="disease_label", y="image_count")
    plt.title("Synthetic Image Distribution by Disease")
    plt.xlabel("Disease Label")
    plt.ylabel("Number of Images")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/disease_distribution.png")
    plt.close()
    print(f"Saved plot to {output_dir}/disease_distribution.png")
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", required=True)
    parser.add_argument("--output", default="analytics/plots")
    args = parser.parse_args()
    run_analytics(args.table, args.output)