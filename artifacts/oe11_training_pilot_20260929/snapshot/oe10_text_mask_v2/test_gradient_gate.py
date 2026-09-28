"""CPU torch injection tests; no actual reader/EO, GPU or performance claim."""
import unittest
import torch
from torch import nn
from gradient_gate import require_finite_loss, validate_gradient_contract


class GradientGateTests(unittest.TestCase):
    def modules(self, arm='B2'):
        modules = {name: nn.Linear(1, 2, bias=False)
                   for name in ('encoder', 'head', 'qwen', 'unused_connector')}
        modules['head'].weight.grad = torch.tensor([[1.0], [-0.5]])
        if arm == 'B2':
            modules['encoder'].weight.grad = torch.tensor([[0.0], [0.25]])
        return modules

    def test_loss_finite_zero_and_nonzero_accepted(self):
        for value in (0.0, 0.5, -1.0):
            require_finite_loss(torch.tensor(value))

    def test_loss_nan_and_infinities_rejected_before_backward(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'Nonfinite mask loss'):
                require_finite_loss(torch.tensor(value, requires_grad=True))

    def test_loss_nontensor_and_vector_rejected(self):
        for value in (0.5, None, torch.ones(2)):
            with self.assertRaisesRegex(ValueError, 'scalar tensor'):
                require_finite_loss(value)

    def test_finite_b2_and_b0_gradient_boundaries(self):
        for arm in ('B0', 'B2'):
            modules = self.modules(arm)
            modules['qwen'].weight.grad = torch.zeros_like(modules['qwen'].weight)
            result = validate_gradient_contract(modules, arm)
            self.assertEqual(result, {'encoder': int(arm == 'B2'), 'head': 1,
                                      'qwen': 0, 'unused_connector': 0})

    def test_nan_inf_any_present_gradient_rejected(self):
        for role in ('encoder', 'head', 'qwen', 'unused_connector'):
            for value in (float('nan'), float('inf'), -float('inf')):
                with self.subTest(role=role, value=value):
                    modules = self.modules()
                    modules[role].weight.grad = torch.tensor([[value], [1.0]])
                    with self.assertRaisesRegex(ValueError, 'Nonfinite gradient: ' + role):
                        validate_gradient_contract(modules, 'B2')

    def test_zero_or_missing_required_gradients_fail(self):
        for role in ('encoder', 'head'):
            for value in (None, torch.zeros(2, 1)):
                modules = self.modules()
                modules[role].weight.grad = value
                with self.subTest(role=role, zero=value is not None), self.assertRaises(ValueError):
                    validate_gradient_contract(modules, 'B2')

    def test_nonzero_frozen_gradients_fail(self):
        for role in ('qwen', 'unused_connector'):
            modules = self.modules()
            modules[role].weight.grad = torch.ones(2, 1)
            with self.assertRaisesRegex(ValueError, 'Gradient boundaries'):
                validate_gradient_contract(modules, 'B2')
        with self.assertRaisesRegex(ValueError, 'EO arm gradient contract'):
            validate_gradient_contract(self.modules('B2'), 'B0')

    def test_actual_cpu_backward_matches_contract(self):
        modules = self.modules('B0')
        for module in modules.values():
            module.zero_grad(set_to_none=True)
        loss = modules['head'](modules['encoder'](torch.ones(1, 1)).mean().reshape(1, 1)).square().mean()
        require_finite_loss(loss)
        loss.backward()
        result = validate_gradient_contract(modules, 'B2')
        self.assertEqual(result['encoder'], 1)
        self.assertEqual(result['head'], 1)


if __name__ == '__main__':
    unittest.main()
