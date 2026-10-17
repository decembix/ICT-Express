import unittest
import torch
import torch.nn as nn

from lora_comparison.modeling import install_variant


class Dense(nn.Module):
    def __init__(self, hidden=6, width=12):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, width, bias=False)
        self.up_proj = nn.Linear(hidden, width, bias=False)
        self.down_proj = nn.Linear(width, hidden, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class SparseShim(nn.Module):
    def __init__(self, dense):
        super().__init__()
        self.experts = nn.ModuleList([dense])
        self.gate = nn.Linear(dense.gate_proj.in_features, 1, bias=False)


class Layer(nn.Module):
    def __init__(self, dense):
        super().__init__()
        self.mlp = SparseShim(dense)


class Model(nn.Module):
    def __init__(self, dense):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([Layer(dense)])


def make_model(seed=0):
    torch.manual_seed(seed)
    return Model(Dense())


class ModelingTest(unittest.TestCase):
    def test_lowrank_router_starts_as_exact_uniform_spread(self):
        model = make_model()
        x = torch.randn(2, 3, 6)
        dense = model.model.layers[0].mlp.experts[0]
        expected = dense(x)
        install_variant(
            model,
            "lowrank_router",
            num_groups=3,
            rank=2,
            alpha=2,
            router_init=None,
        )
        actual, logits = model.model.layers[0].mlp(x)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(logits, torch.zeros_like(logits))

    def test_concat_lora_zero_B_starts_as_exact_spread(self):
        model = make_model()
        x = torch.randn(2, 3, 6)
        dense = model.model.layers[0].mlp.experts[0]
        expected = dense(x)
        install_variant(
            model,
            "uniform_expert_lora",
            num_groups=3,
            rank=2,
            alpha=2,
            router_init=None,
        )
        actual, logits = model.model.layers[0].mlp(x)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        self.assertIsNone(logits)

    def test_router_factorial_has_only_requested_trainable_parameters(self):
        router_init = torch.randn(1, 3, 6)
        model = make_model()
        install_variant(
            model,
            "router_lora",
            num_groups=3,
            rank=2,
            alpha=2,
            router_init=router_init,
        )
        names = {name for name, value in model.named_parameters() if value.requires_grad}
        self.assertEqual(
            names,
            {
                "model.layers.0.mlp.router_lora_A",
                "model.layers.0.mlp.router_lora_B",
            },
        )

        model = make_model()
        install_variant(
            model,
            "router_frozen_expert_lora",
            num_groups=3,
            rank=2,
            alpha=2,
            router_init=router_init,
        )
        names = {name for name, value in model.named_parameters() if value.requires_grad}
        self.assertTrue(names)
        self.assertTrue(all("expert_" in name for name in names))
