import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"code"))
from audit_streaming_external_reports import ap_value, date_stats, recovery, summarize_kuro


def reports():
    out=[]
    for d in (1,2,3):
        arms={}
        values={"teacher3":.7,"stale2":.1,"singles_mean":.2}
        values.update({f"{mod}_seed{s}":value for mod,value in (("gru",.76),("gru_noobs",.11),("ema",.15)) for s in (1,2,3)})
        for name,value in values.items():
            arms[name]={"flood_ap":value,"recovery":recovery(value,.7,.1),
                "activation_macro_ap":value,"per_activation_ap":{"a":value,"b":value},
                "false_alarm_no_water":.01}
        out.append({"decoder_seed":d,"n_test":1999,"arms":arms})
    return out


class AuditTests(unittest.TestCase):
    def test_complete_arithmetic_is_not_provenance(self):
        r=summarize_kuro(reports())
        self.assertTrue(r["all_arithmetic_gates_pass"])
        self.assertFalse(r["provenance_certified"])

    def test_over_100_is_not_clipped(self):
        self.assertAlmostEqual(recovery(.76,.7,.1),1.1)

    def test_bad_denominator_undefined(self):
        self.assertIsNone(recovery(.3,.1,.1))
        self.assertIsNone(recovery(.3,.1,.2))

    def test_nonfinite_and_bool_rejected(self):
        for v in (float("nan"),float("inf"),True,-.1,1.1):
            with self.assertRaises(ValueError): ap_value(v)

    def test_missing_arm_cannot_pass(self):
        rs=reports(); del rs[0]["arms"]["ema_seed3"]
        self.assertFalse(summarize_kuro(rs)["all_arithmetic_gates_pass"])

    def test_duplicate_decoder_cannot_pass(self):
        rs=reports(); rs[2]=copy.deepcopy(rs[1])
        self.assertFalse(summarize_kuro(rs)["complete"])

    def test_reported_recovery_checked(self):
        rs=reports(); rs[0]["arms"]["gru_seed1"]["recovery"]=2
        self.assertFalse(summarize_kuro(rs)["complete"])

    def test_event_macro_checked(self):
        rs=reports(); rs[0]["arms"]["teacher3"]["activation_macro_ap"]+=.1
        self.assertFalse(summarize_kuro(rs)["complete"])

    def test_seeds_not_counted_as_events(self):
        r=summarize_kuro(reports())
        self.assertEqual(len(r["event_delta_gru_minus_teacher_seed_mean"]),2)

    def test_date_reuse_and_later_reference_are_separate(self):
        r=date_stats([{"s1_dates":["2026-01-02","2026-01-02"],"gap_days":[1,-1]}])
        self.assertEqual(r["tiles_reusing_same_s1_date"],1)
        self.assertEqual(r["after_target_s2_date"],1)
        self.assertEqual(r["assignments"],2)


if __name__=="__main__": unittest.main()
