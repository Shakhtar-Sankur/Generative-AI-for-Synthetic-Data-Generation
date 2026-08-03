from google.cloud import bigquery
import os
import glob
import argparse

def upload_to_bigquery(image_dir, table_id, disease_label="pulmonary_fibrosis"):
    client = bigquery.Client()
    images = glob.glob(f"{image_dir}/*.png")
    rows_to_insert = [
        {"image_id": i, "image_data": open(img, "rb").read().hex(), "disease_label": disease_label}
        for i, img in enumerate(images)
    ]
    errors = client.insert_rows_json(table_id, rows_to_insert)
    if not errors:
        print(f"Uploaded {len(rows_to_insert)} images to {table_id}")
    else:
        print(f"Errors: {errors}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="data/synthetic")
    parser.add_argument("--table", required=True)
    parser.add_argument("--disease-label", default="pulmonary_fibrosis")
    args = parser.parse_args()
    upload_to_bigquery(args.images, args.table, args.disease_label)