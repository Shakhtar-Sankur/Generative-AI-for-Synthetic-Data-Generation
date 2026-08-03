import unittest
import torch
from ddpm import DDPM, get_noise_schedule

class TestDDPM(unittest.TestCase):
    def setUp(self):
        self.model = DDPM(channels=1, hidden_dim=32, timesteps=1000)
        self.batch_size = 4
        self.input_shape = (self.batch_size, 1, 64, 64)

    def test_forward_pass(self):
        x = torch.randn(self.input_shape)
        t = torch.randint(0, self.model.timesteps, (self.batch_size,))
        output = self.model(x, t)
        self.assertEqual(output.shape, self.input_shape)

    def test_noise_schedule(self):
        betas, alphas, alphas_cumprod = get_noise_schedule(timesteps=1000)
        self.assertEqual(len(betas), 1000)
        self.assertTrue((betas > 0).all())
        self.assertTrue((alphas_cumprod <= 1).all())

if __name__ == "__main__":
    unittest.main()