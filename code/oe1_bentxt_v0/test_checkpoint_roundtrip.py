import io
import unittest

import torch
from torch import nn

from train_pilot import Projector, cpu_state, last_token_logits, validate_checkpoint_probe


class TinyReader(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(19, 8)
        self.readout = nn.Linear(8, 19)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, inputs_embeds, attention_mask, use_cache):
        cumulative = (inputs_embeds * attention_mask.unsqueeze(-1)).cumsum(dim=1)
        return type("ReaderOutput", (), {"logits": self.readout(cumulative)})()


class CheckpointRoundtripTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.reader = TinyReader().eval().requires_grad_(False)
        self.encoder = nn.Linear(3, 4)
        self.projector = Projector(4, 8, .25)
        self.images = torch.randn(2, 16, 3)
        self.prompts = [{"prefix_ids": [1, 2], "suffix_ids": [3, 4]},
                        {"prefix_ids": [1], "suffix_ids": [5, 6, 7]}]
        self.question_ids = ["train-question-0", "train-question-1"]
        optimizer = torch.optim.AdamW(list(self.encoder.parameters()) + list(self.projector.parameters()), lr=.01)
        logits = self.logits()
        loss = nn.functional.cross_entropy(logits, torch.tensor([8, 9]))
        loss.backward()
        self.assertGreater(float(self.encoder.weight.grad.abs().sum()), 0)
        self.assertTrue(all(parameter.grad is None for parameter in self.reader.parameters()))
        optimizer.step()
        self.encoder.eval(); self.projector.eval()
        with torch.no_grad():
            self.reference = {"question_ids": self.question_ids, "logits": self.logits().detach().clone()}
        buffer = io.BytesIO()
        torch.save({"encoder": cpu_state(self.encoder), "projector": cpu_state(self.projector),
                    "pre_save_train_probe": self.reference}, buffer)
        buffer.seek(0)
        self.saved = torch.load(buffer, weights_only=True)

    def logits(self):
        return last_token_logits(self.reader, self.projector, self.prompts, [0, 1],
                                 self.encoder(self.images), torch.device("cpu"))

    def test_saved_state_reproduces_trained_state(self):
        with torch.no_grad():
            self.encoder.weight.zero_()
            self.projector.gain.zero_()
        self.encoder.load_state_dict(self.saved["encoder"])
        self.projector.load_state_dict(self.saved["projector"])
        with torch.no_grad():
            actual = self.logits()
        report = validate_checkpoint_probe(self.saved["pre_save_train_probe"], self.question_ids, actual)
        self.assertTrue(report["logits_bit_exact"])
        self.assertEqual(report["split"], "train")
        self.assertEqual(report["n_questions"], 2)

    def test_mutated_encoder_is_rejected_even_if_repeated_load_is_stable(self):
        self.saved["encoder"]["weight"][0, 0] += 2.
        self.encoder.load_state_dict(self.saved["encoder"])
        self.projector.load_state_dict(self.saved["projector"])
        with torch.no_grad():
            corrupted = self.logits()
        self.encoder.load_state_dict(self.saved["encoder"])
        self.projector.load_state_dict(self.saved["projector"])
        with torch.no_grad():
            repeated = self.logits()
        self.assertTrue(torch.equal(corrupted, repeated))
        with self.assertRaisesRegex(ValueError, "original trained-state logits"):
            validate_checkpoint_probe(self.saved["pre_save_train_probe"], self.question_ids, corrupted)

    def test_different_probe_questions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "questions differ"):
            validate_checkpoint_probe(self.reference, list(reversed(self.question_ids)), self.reference["logits"])


if __name__ == "__main__":
    unittest.main()
