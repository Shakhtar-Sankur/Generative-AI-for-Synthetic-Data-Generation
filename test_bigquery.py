import unittest
from google.cloud import bigquery
from bigquery_upload import upload_to_bigquery
import os
import glob

class TestBigQuery(unittest.TestCase):
    def setUp(self):
        self.client = bigquery.Client()
        self.table_id = "test-project.dataset.test_images"
        self.image_dir = "data/test_synthetic"
        os.makedirs(self.image_dir, exist_ok=True)
        with open(f"{self.image_dir}/test.png", "wb") as f:
            f.write(b"test_image_data")

    def test_upload(self):
        upload_to_bigquery(self.image_dir, self.table_id)
        query = f"SELECT COUNT(*) as count FROM `{self.table_id}`"
        result = self.client.query(query).result()
        count = list(result)[0]["count"]
        self.assertEqual(count, 1)
        for f in glob.glob(f"{self.image_dir}/*.png"):
            os.remove(f)

if __name__ == "__main__":
    unittest.main()