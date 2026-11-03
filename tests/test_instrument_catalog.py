"""真实目录引用与查询边界；不把结构检查当成演奏或听感验收。"""

import copy
import hashlib
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "skills/code-band-bgm/scripts"))
from instrument_catalog import CATALOG_PATH, query_catalog, validate_catalog
from score_tools import digest

# 覆盖公开注册表新资源（古筝 v3、古琴 v2）的实测能力报告
PUBLIC_CAPABILITY = ROOT / "tests/fixtures/engine-capabilities-20261103-v22.json"


class InstrumentCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        candidates = [ROOT / "tests/fixtures" / name for name in (
            "engine-capabilities-expansion-20261031-v9.json", "engine-capabilities-expansion-20261031-v8.json", "engine-capabilities-expansion-20261031-v7.json", "engine-capabilities-expansion-20261031-v6.json", "engine-capabilities-expansion-20261031-v5.json", "engine-capabilities-expansion-20261031-v4.json", "engine-capabilities-expansion-20261031-v3.json", "engine-capabilities-expansion-20261031-v2.json", "engine-capabilities-final.json", "engine-capabilities-local-v4.json",
            "engine-capabilities-local-v3.json")]
        cls.capability_path = next(path for path in candidates if path.is_file())

    def test_real_catalog_references_and_scope(self):
        report = validate_catalog()
        self.assertEqual(report["status"], "通过")
        self.assertGreaterEqual(report["catalog_entries"], 200)
        self.assertGreaterEqual(report["registered_resources"], 150)
        self.assertFalse(report["runtime_verified"])

    def test_optional_meta_piano_preserves_primary_and_registry_summary(self):
        item = query_catalog("立式钢琴")["matches"][0]
        self.assertEqual(item["instrument_id"], "upright_piano")
        self.assertEqual(item["availability"], "partial")
        self.assertEqual(item["full_instrument_coverage"], "missing")
        self.assertEqual(item["resource_id"], "vcsl_upright_piano_rr1")
        self.assertEqual(item["prototype_resource_id"], "vcsl_upright_piano_rr1")
        self.assertEqual(item["available_resource_ids"], ["vcsl_upright_piano_rr1", "vcsl_upright_piano_rr2", "meta_piano_upright_cc0"])
        self.assertEqual(item["resource_playable_midi_keys"]["meta_piano_upright_cc0"], list(range(21, 109)))
        self.assertEqual(item["resource_pitch_ranges"]["meta_piano_upright_cc0"], {"min_midi": 21, "max_midi": 108})
        self.assertIn("显式", item["notes_zh"])
        self.assertFalse(item["runtime"]["runtime_verified"])
        registry_path = ROOT / "skills/code-band-bgm" / self.catalog["registry"]["supplementary_path"]
        registry = json.loads(registry_path.read_text())
        default = json.loads((ROOT / "skills/code-band-bgm/assets/soundfont-packs.json").read_text())
        self.assertEqual(registry["packs"][:len(default["packs"])], default["packs"])
        all_supp = {r["instrument_id"] for p in registry["packs"] for r in p["resources"]}
        self.assertIn("meta_piano_upright_cc0", all_supp)
        all_main = {r["instrument_id"] for r in json.loads((ROOT / "skills/code-band-bgm/assets/instruments.json").read_text())}
        plugin_path = self.catalog["registry"].get("plugin_resources_path")
        all_plugins = ({r["instrument_id"] for r in json.loads((ROOT / "skills/code-band-bgm" / plugin_path).read_text())["resources"]}
                       if plugin_path else set())
        mapped = {r for entry in self.catalog["instruments"] for r in entry["available_resource_ids"]}
        self.assertEqual(set(self.catalog["registry"]["mapped_unique_resource_ids"]), mapped)
        self.assertEqual(set(self.catalog["registry"]["unmapped_registered_resource_ids"]), (all_main | all_supp | all_plugins) - mapped)
        self.assertEqual(self.catalog["summary"]["mapped_unique_resource_count"], len(mapped))
        self.assertEqual(self.catalog["summary"]["registered_resource_count"], len(all_main))
        self.assertEqual(self.catalog["summary"]["supplementary_resource_count"], len(all_supp))
        self.assertEqual(self.catalog["registry"]["supplementary_file_sha256"],
                         hashlib.sha256(registry_path.read_bytes()).hexdigest())

    def test_old_capabilities_never_implicitly_verify_meta_piano(self):
        item = query_catalog("立式钢琴", self.capability_path)["matches"][0]
        resources = {r["resource_id"]: r for r in item["runtime"]["resources"]}
        self.assertTrue(resources["vcsl_upright_piano_rr1"]["runtime_verified"])
        self.assertFalse(resources["meta_piano_upright_cc0"]["runtime_verified"])
        self.assertEqual(resources["meta_piano_upright_cc0"]["playable_midi_keys"], list(range(21, 109)))

    def test_meta_nylon_keeps_default_and_requires_matching_capability(self):
        report = ROOT / "tests/fixtures/engine-capabilities-128-progress-v12-20261102.json"
        item = query_catalog("古典吉他", report)["matches"][0]
        self.assertEqual(item["resource_id"], "nylon_guitar")
        self.assertIn("meta_nylon_guitar_cc0", item["available_resource_ids"])
        resources = {r["resource_id"]: r for r in item["runtime"]["resources"]}
        self.assertTrue(resources["meta_nylon_guitar_cc0"]["runtime_verified"])
        self.assertEqual(resources["meta_nylon_guitar_cc0"]["playable_midi_keys"], list(range(40, 84)))
        old = query_catalog("古典吉他", self.capability_path)["matches"][0]
        states = {r["resource_id"]: r["runtime_verified"] for r in old["runtime"]["resources"]}
        self.assertFalse(states["meta_nylon_guitar_cc0"])

    @unittest.skipUnless((ROOT / "tests/fixtures/Meta钢琴合并音色能力.json").is_file()
                         and (ROOT / "skills/code-band-bgm/assets/meta-piano/MetaPiano-v1-48k16.sf2").is_file(),
                         "未提供可选Meta钢琴音源与已探测能力报告")
    def test_meta_piano_requires_matching_optional_capability_and_key_map(self):
        report = ROOT / "tests/fixtures/Meta钢琴合并音色能力.json"
        item = query_catalog("立式钢琴", report)["matches"][0]
        states = {r["resource_id"]: r["runtime_verified"] for r in item["runtime"]["resources"]}
        self.assertTrue(states["meta_piano_upright_cc0"])
        self.assertTrue(states["vcsl_upright_piano_rr1"])
        self.assertEqual(item["availability"], "partial")
        cap = json.loads(report.read_text())
        resource = next(r for r in cap["resources"] if r["instrument_id"] == "meta_piano_upright_cc0")
        resource["playable_midi_keys"].remove(60)
        cap["resource_manifest_sha256"] = digest(cap["resources"])
        with tempfile.TemporaryDirectory() as temporary:
            altered = Path(temporary) / "wrong-keys.json"
            altered.write_text(json.dumps(cap), encoding="utf-8")
            item = query_catalog("立式钢琴", altered)["matches"][0]
            states = {r["resource_id"]: r["runtime_verified"] for r in item["runtime"]["resources"]}
            self.assertFalse(states["meta_piano_upright_cc0"])
            self.assertTrue(states["vcsl_upright_piano_rr1"])

    def test_guzheng_partial_has_real_range_without_full_claim(self):
        item = query_catalog("古筝")["matches"][0]
        self.assertEqual(item["availability"], "partial")
        self.assertEqual(item["available_resource_ids"], ["guzheng_pufermufin_sliced_v3"])
        self.assertEqual(item["resource_playable_midi_keys"]["guzheng_pufermufin_sliced_v3"], list(range(42, 78)))
        self.assertEqual(item["pitch_range"], {"min_midi": 42, "max_midi": 77})
        self.assertEqual(item["full_instrument_coverage"], "missing")
        self.assertFalse(item["runtime"]["runtime_verified"])

    def test_sparse_chinese_keys_are_visible_and_not_filled_with_pitch_range(self):
        pipa = query_catalog("琵琶", self.capability_path)["matches"][0]
        self.assertTrue(pipa["runtime"]["runtime_verified"])
        keys = pipa["resource_playable_midi_keys"]["pipa_normal_ccby4"]
        self.assertIn(88, keys)
        self.assertNotIn(87, keys)
        sheng = next(e for e in query_catalog("笙", self.capability_path)["matches"] if e["instrument_id"] == "sheng")
        self.assertEqual(len(sheng["resource_playable_midi_keys"]["sheng_old_ccby4"]), 20)
        self.assertEqual(sheng["full_instrument_coverage"], "missing")

    def test_new_prototypes_expose_only_verified_keys(self):
        report = ROOT / "tests/fixtures/engine-capabilities-128-progress-v14-20261102.json"
        cases = (("木鱼", "muyu_ccby4_four_hits", [60, 61, 62, 63]),
                 ("科拉琴", "sp_kora_cc0", [46, 47, 48, 49, 50]),
                 ("乌德琴", "aurora_oud_cc0", [45, 46, 48, 50, 52, 53, 54, 55, 58, 59, 60, 62, 63, 64, 65]),
                 ("奈伊笛", "aurora_ney_cc0", [60, 65, 78]),
                 ("布祖基琴", "justmog_bouzouki_cc0", [43, 50, 62, 76]),
                 ("超高音竖笛", "piermic_sopranino_cc0", [86, 88, 89, 91]),
                 ("击弦古钢琴", "edinburgh_clavichord_415_by", [57, 67, 69, 72, 75, 76, 77, 79, 81, 85, 86]),
                 ("北美原住民笛", "clintgoss_native_flute", list(range(64, 81))),
                 ("拨杆竖琴", "etherealwinds_lever_harp_by", list(range(36, 94))),
                 ("古大提琴", "orchidea_bass_viol_by", [47, 48, 51, 52, 54, 62, 64, 67, 68, 69, 72]),
                 ("交响大锣", "pjcohen_tamtam_bya", [60, 61]),
                 ("萨朗吉", "fsd_sarangi_by3_single", [63]))
        for name, resource, keys in cases:
            with self.subTest(name=name):
                item = query_catalog(name, report)["matches"][0]
                self.assertEqual(item["availability"], "partial")
                self.assertEqual(item["full_instrument_coverage"], "missing")
                self.assertEqual(item["resource_playable_midi_keys"][resource], keys)
                self.assertTrue(item["runtime"]["runtime_verified"])
                self.assertFalse(query_catalog(name, self.capability_path)["matches"][0]["runtime"]["runtime_verified"])

    def test_unpublished_packs_leave_their_concepts_unbound(self):
        # 公开版不收 BY-SA、权利链不清和播放器预览流的包；对应概念必须降为缺失，不能回退到 GM 音色
        for name in ("柳琴", "三弦", "中阮", "唢呐", "六角手风琴", "富鲁格号", "安第斯排箫", "姆里丹加鼓", "中胡"):
            with self.subTest(name=name):
                item = query_catalog(name)["matches"][0]
                self.assertEqual(item["availability"], "missing")
                self.assertEqual(item["available_resource_ids"], [])
                self.assertIsNone(item["resource_id"])

    def test_cultural_instruments_are_not_gm_substitutes(self):
        dizi = query_catalog("笛子")["matches"][0]
        self.assertEqual(dizi["resource_id"], "dizi_sutd_g_membrane")
        self.assertEqual(dizi["available_resource_ids"], ["dizi_sutd_g_membrane", "dizi_sutd_g_alt1", "dizi_sutd_g_alt2"])
        self.assertEqual(len(dizi["resource_playable_midi_keys"]["dizi_sutd_g_membrane"]), 21)
        self.assertNotIn(75, dizi["resource_playable_midi_keys"]["dizi_sutd_g_membrane"])
        self.assertNotEqual(dizi["resource_id"], "flute")
        self.assertNotIn("dizi", [item["instrument_id"] for item in query_catalog("长笛")["matches"]])
        erhu = query_catalog("二胡")["matches"][0]
        self.assertEqual(erhu["availability"], "partial")
        self.assertIn("erhu_long_cc0", erhu["available_resource_ids"])
        self.assertNotIn("gm_110", erhu["available_resource_ids"])

    def test_real_capability_verifies_only_finite_prototype(self):
        item = query_catalog("古筝", PUBLIC_CAPABILITY)["matches"][0]
        self.assertTrue(item["runtime"]["runtime_verified"])
        self.assertEqual(item["availability"], "partial")
        self.assertEqual(item["full_instrument_coverage"], "missing")
        self.assertEqual(item["runtime"]["resources"][0]["pitch_range"], item["pitch_range"])

    def test_unknown_capability_cannot_verify(self):
        capability = json.loads(self.capability_path.read_text())
        capability["probe_state"] = "unprobed"
        capability.update(engine=None, observed_at=None, resource_manifest_sha256=None,
                          evidence=[], resources=[])
        for group in ("execution", "timing", "performance", "mixing", "export", "reproducibility"):
            for feature in capability[group].values():
                feature["state"] = "unknown"
                feature["evidence_ids"] = []
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "unprobed.json"
            path.write_text(json.dumps(capability))
            self.assertFalse(query_catalog("古筝", path)["matches"][0]["runtime"]["runtime_verified"])

    def test_wrong_asset_binding_cannot_verify_that_resource(self):
        capability = json.loads(PUBLIC_CAPABILITY.read_text())
        for resource in capability["resources"]:
            if resource["instrument_id"] == "guqin_songest_v2":
                resource["engine_resource_id"] = "sf:0:0:" + "0" * 64
        capability["resource_manifest_sha256"] = digest(capability["resources"])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "wrong.json"
            path.write_text(json.dumps(capability))
            item = query_catalog("古琴", path)["matches"][0]
            states = {r["resource_id"]: r["runtime_verified"] for r in item["runtime"]["resources"]}
            self.assertFalse(states["guqin_songest_v2"])
            self.assertTrue(states["guqin_open_strings_ccby4"])

    def test_drum_hit_requires_actual_adapter_key(self):
        capability = json.loads(self.capability_path.read_text())
        for resource in capability["resources"]:
            if resource["instrument_id"] == "drums":
                resource["unpitched_hits"].pop("kick")
        capability["resource_manifest_sha256"] = digest(capability["resources"])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "missing-hit.json"
            path.write_text(json.dumps(capability))
            self.assertFalse(query_catalog("底鼓", path)["matches"][0]["runtime"]["runtime_verified"])

    def test_drum_hit_name_must_match_actual_midi_number(self):
        capability = json.loads(self.capability_path.read_text())
        for resource in capability["resources"]:
            if resource["instrument_id"] == "drums":
                resource["unpitched_hits"]["kick"] = "35"
        capability["resource_manifest_sha256"] = digest(capability["resources"])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "wrong-hit-note.json"
            path.write_text(json.dumps(capability))
            self.assertFalse(query_catalog("底鼓", path)["matches"][0]["runtime"]["runtime_verified"])

    def test_verified_gm_percussion_bindings_have_exact_notes(self):
        expected = {
            "triangle": {"triangle_mute": "80", "triangle_open": "81"},
            "tambourine": {"tambourine": "54"},
            "conga": {"conga_mute_high": "62", "conga_open_high": "63", "conga_low": "64"},
            "bongo": {"bongo_high": "60", "bongo_low": "61"},
            "cowbell": {"cowbell": "56"}, "cabasa": {"cabasa": "69"},
            "claves": {"claves": "75"}, "guiro": {"guiro_short": "73", "guiro_long": "74"},
            "timbales": {"timbale_high": "65", "timbale_low": "66"},
            "vibraslap": {"vibraslap": "58"},
            "cuica": {"cuica_mute": "78", "cuica_open": "79"}}
        for concept, notes in expected.items():
            with self.subTest(concept=concept):
                item = next(entry for entry in query_catalog(concept, self.capability_path)["matches"]
                            if entry["instrument_id"] == concept)
                self.assertEqual(item["availability"], "available")
                self.assertEqual(item["available_resource_ids"], ["drums"])
                self.assertEqual(item["required_hit_notes"], notes)
                self.assertTrue(item["runtime"]["runtime_verified"])

    def test_dedicated_cymbals_do_not_bind_standard_kit_candidates(self):
        entries = {entry["instrument_id"]: entry for entry in query_catalog()["matches"]}
        for concept in ("chinese_cymbals", "chinese_daluo"):
            with self.subTest(concept=concept):
                self.assertEqual(entries[concept]["availability"], "partial")
                self.assertTrue(all(r.startswith("jingju_") for r in entries[concept]["available_resource_ids"]))
                self.assertNotIn("drums", entries[concept]["available_resource_ids"])
        self.assertEqual(entries["suspended_cymbal"]["candidate_hit_notes"], {"crash": "49", "ride": "51"})
        for concept in ("pair_cymbals", "suspended_cymbal"):
            self.assertEqual(entries[concept]["availability"], "partial")
            self.assertTrue(all(r.startswith("vcsl_") for r in entries[concept]["available_resource_ids"]))

    def test_invalid_capabilities_return_cli_json_without_traceback(self):
        capability = json.loads(self.capability_path.read_text())
        wrong_digest = copy.deepcopy(capability)
        wrong_digest["resource_manifest_sha256"] = "0" * 64
        inconsistent_unprobed = copy.deepcopy(capability)
        inconsistent_unprobed["probe_state"] = "unprobed"
        for label, value in (("数组", []), ("错误摘要", wrong_digest),
                             ("未探测却宣称支持", inconsistent_unprobed)):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "invalid.json"
                path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    query_catalog("古筝", path)
                result = subprocess.run([sys.executable, str(ROOT / "skills/code-band-bgm/scripts/band.py"),
                                         "catalog", "--query", "古筝", "--capabilities", str(path)],
                                        capture_output=True, text=True, check=False)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(json.loads(result.stdout)["status"], "失败")
                self.assertNotIn("Traceback", result.stderr)

    def test_catalog_and_registry_container_types_rejected(self):
        with self.assertRaises(ValueError):
            validate_catalog([])
        for field, value in (("registry", []), ("instruments", [None]),
                             ("families", {}), ("sources", [None]), ("summary", [])):
            catalog = copy.deepcopy(self.catalog)
            catalog[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_catalog(catalog)
        with patch("instrument_catalog._read", return_value={}), self.assertRaises(ValueError):
            validate_catalog(self.catalog)
        base = json.loads((ROOT / "skills/code-band-bgm/assets/instruments.json").read_text())
        with patch("instrument_catalog._read", side_effect=[base, {"packs": [None]}]), self.assertRaises(ValueError):
            validate_catalog(self.catalog)

    def test_missing_resource_family_source_and_duplicate_rejected(self):
        for change in ("resource", "family", "source", "duplicate"):
            catalog = copy.deepcopy(self.catalog)
            entry = catalog["instruments"][0]
            if change == "resource":
                entry["available_resource_ids"] = ["nonexistent"]
                entry["resource_id"] = "nonexistent"
            elif change == "family":
                entry["family_id"] = "nonexistent"
            elif change == "source":
                entry["source_refs"] = ["nonexistent"]
            else:
                catalog["instruments"].append(copy.deepcopy(entry))
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_catalog(catalog)

    def test_overview_and_shell_characters_are_data(self):
        overview = query_catalog()
        self.assertEqual(len(overview["matches"]), overview["summary"]["catalog_entries"])
        self.assertTrue(all(not entry["runtime"]["runtime_verified"] for entry in overview["matches"]))
        self.assertEqual(len(overview["families"]), 14)
        self.assertEqual(query_catalog("$(touch /tmp/catalog_bad);古筝")["matches"], [])


if __name__ == "__main__":
    unittest.main()
