import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "skills/code-band-bgm/scripts"))
from instrument_readiness import assess, review_score  # noqa: E402


def zone(low, high, root, sample="s"):
    return {"key_range": [low, high], "root_key": root, "velocity_range": [1, 127], "available_in_file": True,
            "sample_id": f"{sample}{root}", "source_sample_name": f"{sample}{root}", "sample_modes": 0}


def resource(name="x", low=0, high=127, **extra):
    return {"instrument_id": name, "kind": "sampler", "engine_resource_id": f"sf:0:0:{'a' * 64}",
            "pitch_range": {"min_midi": low, "max_midi": high}, **extra}


def inventory(zones):
    return {"presets": [{"bank": 0, "program": 0, "mappings": zones}]}


def score(notes, function="melody"):
    return {"parts": [{"part_id": "p", "musical_function": function, "resource_binding": {"instrument_id": "x"},
                       "notes": [{"pitch": {"kind": "pitched", "midi": m}} for m in notes]}]}


DENSE = [zone(0, 26, 24)] + [zone(r - 1, r + 1, r) for r in range(28, 98, 3)] + [zone(98, 127, 99)]


class ReadinessTest(unittest.TestCase):
    def test_edge_zone_stretched_to_full_range_does_not_decide_tier(self):
        result = assess(resource(), inventory(DENSE))
        self.assertEqual(result["tier"], "coverage_ok")
        self.assertTrue(any("声明键" in note for note in result["notes"]))

    def test_six_roots_with_holes_is_probe(self):
        # 《江暮归舟》v2 的古筝：6 个根音各只映射到自身键
        zones = [zone(r, r, r) for r in (41, 48, 57, 62, 65, 72)]
        self.assertEqual(assess(resource(low=41, high=72), inventory(zones))["tier"], "probe")

    def test_two_roots_is_probe_even_when_mapped_everywhere(self):
        zones = [zone(40, 60, 50), zone(61, 90, 75)]
        result = assess(resource(), inventory(zones))
        self.assertEqual(result["tier"], "probe")
        self.assertIn("只有2个真实根音", result["gaps"])

    def test_score_using_far_transposed_key_is_blocked_for_foreground_only(self):
        cap = {"resources": [resource()]}
        inv = {"a" * 64: inventory(DENSE)}
        self.assertEqual(review_score(score([60, 64]), cap, inv)["blocked"], [])
        blocked = review_score(score([5]), cap, inv)["blocked"]
        self.assertEqual(blocked[0]["used_key_gaps"], ["MIDI 5 离真实根音19半音"])
        self.assertEqual(review_score(score([5], "harmony"), cap, inv)["blocked"], [])

    def test_unmapped_used_key_is_reported(self):
        cap = {"resources": [resource()]}
        inv = {"a" * 64: inventory([zone(r - 1, r + 1, r) for r in range(40, 80, 3)])}
        self.assertIn("MIDI 100 没有可发声区域", review_score(score([100]), cap, inv)["blocked"][0]["used_key_gaps"])


if __name__ == "__main__":
    unittest.main()
