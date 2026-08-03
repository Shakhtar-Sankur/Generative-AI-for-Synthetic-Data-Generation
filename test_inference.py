import unittest
import os
import glob
from inference import generate_images
from argparse import Namespace

class TestInference(unittest.TestCase):
    def setUp(self):
        self.args = Namespace(
            checkpoint="models/checkpoints/epoch_49.pth",
            output="data/test_synthetic",
            num_images=2
        )
        os.makedirs(self.args.output, exist_ok=True)

    def test_generate_images(self):
        generate_images(self.args)
        generated_files = glob.glob(f"{self.args.output}/synthetic_*.png")
        self.assertEqual(len(generated_files), 2)
        for f in generated_files:
            os.remove(f)

if __name__ == "__main__":
    unittest.main()