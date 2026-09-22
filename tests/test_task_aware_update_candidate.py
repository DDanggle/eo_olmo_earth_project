import importlib.util
from pathlib import Path
import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    spec = importlib.util.spec_from_file_location("task_update", Path(__file__).parents[1] / "code/task_aware_update_candidate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


@unittest.skipIf(torch is None, "PyTorch not installed; run isolated CPU tests in existing server environment")
class CandidateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.model = module.TaskAwareResidualUpdate(channels=4, hidden=8)
        self.state = torch.randn(2, 4, 5, 5)
        self.obs = torch.randn_like(self.state)
        self.days = torch.tensor([3.0, 12.0])
        self.quality = torch.ones(2, 1, 5, 5)
        self.valid = torch.ones(2, 1, 5, 5, dtype=torch.bool)

    def test_exact_identity_at_initialization(self):
        self.assertTrue(torch.equal(self.model(self.state, self.obs, self.days, self.quality, self.valid), self.state))

    def test_invalid_nan_observation_holds_state(self):
        torch.nn.init.normal_(self.model.delta.weight)
        self.valid[:] = False; self.obs[:] = float("nan")
        self.assertTrue(torch.equal(self.model(self.state, self.obs, self.days, self.quality, self.valid), self.state))

    def test_quality_zero_holds_state(self):
        torch.nn.init.normal_(self.model.delta.weight)
        self.quality[:] = 0
        self.assertTrue(torch.equal(self.model(self.state, self.obs, self.days, self.quality, self.valid), self.state))

    def test_first_step_receives_gradient(self):
        output = self.model(self.state, self.obs, self.days, self.quality, self.valid)
        (output - self.obs).square().mean().backward()
        self.assertGreater(float(self.model.delta.weight.grad.abs().sum()), 0)

    def test_time_and_quality_inputs_are_checked(self):
        for days, quality in ((torch.tensor([-1.0, 2.0]), self.quality), (self.days, self.quality * 2)):
            with self.assertRaises(ValueError):
                self.model(self.state, self.obs, days, quality, self.valid)

    def test_valid_nan_rejected(self):
        self.obs[0, 0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            self.model(self.state, self.obs, self.days, self.quality, self.valid)

    def head(self):
        head = torch.nn.Conv2d(4, 1, 1).eval()
        head.requires_grad_(False)
        return head

    def test_task_loss_reaches_student_not_teacher_or_head(self):
        head = self.head()
        student = self.state.clone().requires_grad_(True)
        teacher = self.obs.clone().requires_grad_(True)
        report = module.preservation_loss(student, teacher, [head], feature_weight=0)
        report["loss"].backward()
        self.assertGreater(float(student.grad.abs().sum()), 0)
        self.assertIsNone(teacher.grad)
        self.assertTrue(all(p.grad is None for p in head.parameters()))

    def test_equal_features_zero_kl(self):
        report = module.preservation_loss(self.state, self.state, [self.head()])
        self.assertAlmostEqual(float(report["loss"]), 0.0, places=6)

    def test_unfrozen_or_training_head_rejected(self):
        head = self.head().train()
        with self.assertRaises(ValueError):
            module.preservation_loss(self.state, self.obs, [head])


if __name__ == "__main__":
    if torch is not None:
        torch.set_num_threads(1)
    unittest.main()
