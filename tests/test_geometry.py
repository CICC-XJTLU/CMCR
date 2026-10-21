"""Small-tensor checks for CMCR geometry; no pretrained model is required."""

import importlib.util
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from cmcr.model import CommutatorRotator, cayley, generators, skew


@unittest.skipUnless(HAS_TORCH, "PyTorch is required for geometry tests")
class GeometryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(33)
        torch.set_num_threads(1)

    def test_supported_variants(self):
        for variant in ('cmcr', 'static', 'no_comm'):
            model = CommutatorRotator(32, 8, 2, 3, 4, variant=variant)
            self.assertEqual(model.variant, variant)
        with self.assertRaises(ValueError):
            CommutatorRotator(32, 8, 2, 3, 4, variant='unsupported')

    def test_cayley_is_orthogonal(self):
        rotation = cayley(skew(torch.randn(3, 4, 4)))
        self.assertTrue(torch.allclose(rotation @ rotation.transpose(-1, -2), torch.eye(4), atol=1e-5))

    def test_commutator_is_skew_symmetric(self):
        mt, mv = torch.randn(2, 4), torch.randn(2, 4)
        c, d = torch.randn(2, 4, 4), torch.randn(2, 4, 4)
        c, d = c @ c.transpose(-1, -2), d @ d.transpose(-1, -2)
        first, second = generators(mt, c, mv, d)
        self.assertTrue(torch.allclose(first, -first.transpose(-1, -2)))
        self.assertTrue(torch.allclose(second, -second.transpose(-1, -2)))
        self.assertEqual(float(generators(mt, c, mt, c)[1].abs().max()), 0.0)

    def test_rotation_preserves_norms(self):
        model = CommutatorRotator(32, 8, 2, 3, 4)
        text = torch.tensor([[0, 0, 0, 1, 1, 1, 0, 0]] * 2, dtype=torch.bool)
        visual = torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0]] * 2, dtype=torch.bool)
        model.masks = (text, visual)
        hidden, query, key = torch.randn(2, 8, 32), torch.randn(2, 4, 8, 8), torch.randn(2, 2, 8, 8)
        q, k = model(hidden, query, key, 0)
        self.assertTrue(torch.allclose(query.norm(dim=-1), q.norm(dim=-1), atol=2e-5))
        self.assertTrue(torch.allclose(key.norm(dim=-1), k.norm(dim=-1), atol=2e-5))
        model.intervention = "bypass"
        q, k = model(hidden, query, key, 0)
        self.assertTrue(torch.equal(query, q) and torch.equal(key, k))


if __name__ == "__main__":
    unittest.main()
