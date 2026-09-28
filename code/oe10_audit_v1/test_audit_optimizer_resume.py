import copy
import unittest
import numpy as np
from audit_optimizer_resume import validate_groups, moment_stats, stats


def groups():
    return [{'role': role, 'params': ids, 'eps': 1e-6, 'foreach': False, 'fused': False,
             'betas': (0.9, 0.999), 'lr': 1e-5}
            for role, ids in [('encoder', [0, 1]), ('native_decoder', [2]), ('readouts', [3])]]


class AuditContractTests(unittest.TestCase):
    def test_actual_order_mapping_and_b0_no_encoder(self):
        self.assertEqual(validate_groups(groups(), 'B2', ['encoder.z', 'encoder.a']), {'encoder.z': 0, 'encoder.a': 1})
        self.assertEqual(validate_groups(groups()[2:], 'B0', ['encoder.z', 'encoder.a']), {})
        with self.assertRaises(ValueError):
            validate_groups(groups(), 'B0', ['encoder.z', 'encoder.a'])

    def test_saved_epsilon_order_and_duplicate_rejected(self):
        for change in ('epsilon', 'count', 'duplicate', 'fused'):
            g = copy.deepcopy(groups())
            if change == 'epsilon':
                g[0]['eps'] = 1e-8
            if change == 'count':
                g[0]['params'] = [0]
            if change == 'duplicate':
                g[1]['params'] = [1]
            if change == 'fused':
                g[0]['fused'] = True
            with self.assertRaises(ValueError):
                validate_groups(g, 'B2', ['encoder.z', 'encoder.a'])

    def test_first_step_bias_correction_and_epsilon_floor(self):
        grad = np.array([1e-8, -2e-8, 0.])
        result = moment_stats({'step': 1, 'exp_avg': grad * .1, 'exp_avg_sq': grad**2 * .001}, groups()[0])
        self.assertAlmostEqual(result['bias_corrected_exp_avg']['abs_max'], 2e-8, places=20)
        self.assertAlmostEqual(result['bias_corrected_sqrt_exp_avg_sq']['abs_max'], 2e-8, places=20)
        self.assertEqual(result['bias_corrected_rms_below_eps_fraction'], 1.)
        self.assertEqual(stats(np.zeros(4))['nonzero_count'], 0)
        with self.assertRaises(ValueError):
            moment_stats({'step': 1, 'exp_avg': grad, 'exp_avg_sq': np.array([-1., 0., 0.])}, groups()[0])
        with self.assertRaises(ValueError):
            stats([float('nan')])


if __name__ == '__main__':
    unittest.main()
