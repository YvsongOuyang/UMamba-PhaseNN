"""Contract checks for split, input units and fail-fast behavior (no dataset needed)."""
import unittest
import numpy as np
import torch

from data import generator_functions, diffraction, split_ids
from train import Samples, check_optimizer, finite_tensors, run_epoch


class Contracts(unittest.TestCase):
    def test_paper_split(self):
        splits = split_ids()
        self.assertEqual([len(splits[k]) for k in ("train", "validation", "test")], [12000, 1500, 1500])
        self.assertTrue(np.array_equal(np.sort(np.concatenate(list(splits.values()))), np.arange(15000)))
        self.assertTrue(np.array_equal(splits["test"], np.arange(13500, 15000)))

    def test_generator_forward_matches_labels(self):
        fn = generator_functions()
        rng = np.random.default_rng(0)
        field = rng.normal(size=(128, 128)) + 1j * rng.normal(size=(128, 128))
        phases, amps, measured = fn["Generate_synthetic_data"](field, np.ones((64, 64)), 128, 64)
        obj = np.float32(amps) * np.exp(1j * np.float32(phases))
        np.testing.assert_allclose(diffraction(obj), measured, rtol=1e-5, atol=1e-5)

    def test_amplitude_is_not_square_rooted(self):
        arrays = dict(modulus_detector=np.full((1, 64, 64), 4, dtype=np.float32),
                      object_modulus=np.ones((1, 64, 64), dtype=np.float32),
                      object_phase=np.full((1, 64, 64), np.pi / 2, dtype=np.float32))
        x, y = Samples(arrays, [0], 8)[0]
        self.assertEqual(tuple(x.shape), (1, 64, 64, 2))
        self.assertTrue(torch.all(x == 0.5))
        self.assertTrue(torch.allclose(y[..., 1], torch.ones_like(y[..., 1])))

    def test_rejects_bad_gradients(self):
        with self.assertRaises(FloatingPointError):
            finite_tensors([torch.tensor(float("nan"))], "gradient")

    def test_rejects_bad_adam_state_with_finite_parameters(self):
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.Adam(model.parameters())
        model(torch.ones(1, 1)).sum().backward()
        optimizer.step()
        next(iter(optimizer.state.values()))["exp_avg_sq"].fill_(float("inf"))
        with self.assertRaises(FloatingPointError):
            check_optimizer(model, optimizer)

    def test_single_and_unequal_batches_are_sample_weighted(self):
        model = torch.nn.Identity()
        x = torch.tensor([1., 1., 4.]).reshape(3, 1, 1, 1, 1).expand(-1, -1, -1, -1, 2)
        y = torch.zeros_like(x)
        for batch in (2, 3):
            loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=batch)
            result = run_epoch(model, loader, torch.device("cpu"))
            self.assertAlmostEqual(result["mae"], 4.0)


if __name__ == "__main__":
    unittest.main()
