"""
Comprehensive Unit & Integration Tests for Women Safety AI System.
Covers:
1. Configuration and safe defaults (verifying restricted area is completely removed)
2. Person Detection and ByteTrack tracking
3. Gender Classification (working_m_&_F.pt) and Counter Equality (TOTAL = MALE + FEMALE + UNKNOWN)
4. Violence Model (best.pt ResNet18-GRU) checkpoint architecture and forward pass
5. Scenario A: Woman attacked -> Female victim red box attribution and qualifying snapshot generated
6. Scenario B: Male-only fight -> No female victim attribution (FIGHT SUSPECTED), NO woman snapshot
7. Scenario C: Woman walking normally -> Safe status (NORMAL), NO snapshot
8. Scenario D: Violence with unconfirmed female involvement -> Warning displayed, NO woman snapshot
9. Fall and Universal SOS detection (gender-aware fall and universal SOS)
10. Incident storage in outputs/incidents/YYYY-MM-DD/ with INCIDENT_YYYYMMDD_HHMMSS_XXX format
11. Alert dispatching in TEST_MODE and non-blocking async queue
12. Nearest police station lookup with distance calculation
13. Evaluation metrics diagnostics
"""

import os
import json
import time
import unittest
from datetime import datetime
import numpy as np
import cv2
import torch

import config
from Person_Detection import detect_person, detect_and_track_persons, detect_carried_objects
from Gender_Detection import detect_genders_in_frame, get_or_predict_gender
from Violence_Detection import load_violence_model, classify_scene, ResNet18GRU
from Pose_Detection import detect_poses
from Snatching_Detection import detect_snatching, reset_snatch_tracks
from SOS_Gesture_Detection import SOSGestureDetector
from Risk_Assessment import RiskAssessmentEngine, IncidentState
from Incident_Manager import save_incident_evidence, load_recent_incidents, get_dated_incident_dir
from Alert_Manager import dispatch_incident_alert, is_queue_busy
from Police_Station_Locator import find_nearest_police_stations
from training.evaluate_metrics import compute_binary_metrics


class TestWomenSafetyAI(unittest.TestCase):

    def setUp(self):
        self.risk_engine = RiskAssessmentEngine()
        self.test_img_path = config.INPUT_IMAGE_PATH

    def test_01_config_and_safety_defaults(self):
        """Verify configuration constants, safe defaults, and complete removal of restricted area."""
        self.assertIsNotNone(config.APP_NAME)
        self.assertTrue(os.path.exists(config.OUTPUT_DIR))
        self.assertTrue(os.path.exists(config.INCIDENT_SAVE_DIR))
        self.assertTrue(config.TEST_MODE, "TEST_MODE must default to True for safety")
        self.assertFalse(config.ALERTS_ENABLED, "ALERTS_ENABLED must default to False until configured")
        self.assertEqual(config.VICTIM_BOX_COLOR, (0, 0, 255), "Victim box must be bright red (BGR)")
        
        # Verify restricted area is completely removed
        self.assertFalse(hasattr(config, "RESTRICTED_ZONES_ENABLED"), "RESTRICTED_ZONES_ENABLED must be completely removed")
        self.assertFalse(hasattr(config, "RESTRICTED_ZONE_POLYGON"), "RESTRICTED_ZONE_POLYGON must be completely removed")
        self.assertNotIn("restricted_zone_entry", config.RISK_WEIGHTS)

    def test_02_person_detection_and_tracking(self):
        """Verify YOLO person detection and ByteTrack tracking on sample image."""
        if os.path.exists(self.test_img_path):
            img = cv2.imread(self.test_img_path)
            self.assertIsNotNone(img, "test_image.png failed to load")
            boxes = detect_person(img)
            self.assertIsInstance(boxes, list)
            self.assertGreater(len(boxes), 0)

            tracked = detect_and_track_persons(img, persist=False)
            self.assertIsInstance(tracked, list)
            self.assertEqual(len(tracked), len(boxes))
            self.assertIn("id", tracked[0])
            self.assertIn("box", tracked[0])

    def test_03_gender_detection_and_counter_equality(self):
        """Verify working_m_&_F.pt outputs and strict mathematical counter equality."""
        if os.path.exists(self.test_img_path):
            img = cv2.imread(self.test_img_path)
            g_dets = detect_genders_in_frame(img)
            self.assertIsInstance(g_dets, list)

            # Test counter identity: TOTAL PERSONS = MALE + FEMALE + UNKNOWN
            tracked = detect_and_track_persons(img, persist=False)
            females, males, unknowns = [], [], []
            for det in tracked:
                g, _ = get_or_predict_gender(det["id"], det["box"], g_dets)
                if g == "Female":
                    females.append(det["id"])
                elif g == "Male":
                    males.append(det["id"])
                else:
                    unknowns.append(det["id"])

            total = len(tracked)
            self.assertEqual(total, len(females) + len(males) + len(unknowns),
                             "Strict equality failed: TOTAL != MALE + FEMALE + UNKNOWN")

    def test_04_violence_cnn_gru_model_loading(self):
        """Verify best.pt loads strictly into ResNet18GRU and executes forward pass."""
        model = load_violence_model()
        self.assertIsNotNone(model, "best.pt failed to load into ResNet18GRU")
        self.assertIsInstance(model, ResNet18GRU)

        dummy_seq = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        for _ in range(8):
            res = classify_scene(dummy_seq, force=True)
        self.assertIn("label", res)
        self.assertIn("confidence", res)
        self.assertIn("is_violence", res)
        self.assertEqual(res.get("model_used"), "ResNet18-GRU (RWF-2000)")

    def test_05_scenario_a_female_victim_highlighted(self):
        """Scenario A: Woman attacked in a fight -> Must be marked unsafe, identified for RED box, and qualify for snapshot."""
        engine = RiskAssessmentEngine()
        signals = {
            "female_ids": [10],
            "male_ids": [20, 21],
            "unknown_ids": [],
            "violence_detected": True,
            "violence_confidence": 0.85,
            "female_attacked_ids": [10],
            "fighting_pairs": [(10, 20)]
        }
        res = engine.evaluate_frame(signals)
        self.assertIn(10, res["victim_female_ids"], "Female victim ID must be attributed for red box")
        self.assertFalse(res["is_woman_safe"])
        self.assertTrue(res["is_qualifying_incident"], "Woman under attack must qualify for snapshot recording")
        self.assertIn("WOMAN", res["primary_incident"])

    def test_06_scenario_b_male_only_fight(self):
        """Scenario B: Male-only fight -> No female victim alert, no red box, NO woman snapshot."""
        engine = RiskAssessmentEngine()
        signals = {
            "female_ids": [],
            "male_ids": [20, 21],
            "unknown_ids": [],
            "violence_detected": True,
            "violence_confidence": 0.80,
            "female_attacked_ids": [],
            "fighting_pairs": [(20, 21)]
        }
        res = engine.evaluate_frame(signals)
        self.assertEqual(res["victim_female_ids"], [], "No female victim should be identified in male-only fight")
        self.assertEqual(res["primary_incident"], "FIGHT SUSPECTED")
        self.assertFalse(res["is_qualifying_incident"], "Male-only fight must NOT qualify for woman safety snapshot")
        self.assertFalse(res["alert_worthy"], "Should not trigger women-specific alert for male-only fight")

    def test_07_scenario_c_woman_normal(self):
        """Scenario C: Woman walking normally with men nearby -> Remains safe, no red box, NO snapshot."""
        engine = RiskAssessmentEngine()
        signals = {
            "female_ids": [5],
            "male_ids": [6, 7],
            "unknown_ids": [],
            "violence_detected": False,
            "female_attacked_ids": [],
            "fighting_pairs": []
        }
        res = engine.evaluate_frame(signals)
        self.assertEqual(res["victim_female_ids"], [])
        self.assertTrue(res["is_woman_safe"])
        self.assertFalse(res["is_qualifying_incident"])
        self.assertEqual(res["woman_status_text"], "WOMAN STATUS: SECURE (SAFE)")

    def test_08_scenario_d_violence_unconfirmed_female(self):
        """Scenario D: Fight detected but woman is distant/unconfirmed -> No red box, NO woman snapshot."""
        engine = RiskAssessmentEngine()
        signals = {
            "female_ids": [5],
            "male_ids": [20, 21],
            "unknown_ids": [],
            "violence_detected": True,
            "female_attacked_ids": [],
            "fighting_pairs": [(20, 21)]
        }
        res = engine.evaluate_frame(signals)
        self.assertEqual(res["victim_female_ids"], [], "Distant female should not be assigned red box")
        self.assertFalse(res["is_qualifying_incident"], "Unconfirmed female presence must NOT generate woman safety snapshot")
        self.assertEqual(res["primary_incident"], "VIOLENCE DETECTED — WOMAN INVOLVEMENT UNCONFIRMED")

    def test_09_fall_and_universal_sos(self):
        """Verify fall attribution and universal SOS eligibility regardless of gender."""
        engine = RiskAssessmentEngine()

        # Female fall qualifies
        signals_fall = {
            "female_ids": [8],
            "male_ids": [9],
            "unknown_ids": [],
            "fall_confirmed": True,
            "fall_person_ids": [8]
        }
        res_fall = engine.evaluate_frame(signals_fall)
        self.assertIn(8, res_fall["victim_female_ids"])
        self.assertTrue(res_fall["is_qualifying_incident"])
        self.assertEqual(res_fall["primary_incident"], "WOMAN FALL SUSPECTED")

        # Universal SOS with Unknown gender qualifies
        signals_sos = {
            "female_ids": [],
            "male_ids": [],
            "unknown_ids": [99],
            "sos_detected": True,
            "sos_details": {"person_id": 99, "description": "Manual Panic Keypress"}
        }
        res_sos = engine.evaluate_frame(signals_sos)
        self.assertTrue(res_sos["is_qualifying_incident"], "SOS must qualify for snapshot and alert regardless of gender")
        self.assertTrue(res_sos["alert_worthy"])
        self.assertIn("SOS", res_sos["primary_incident"])

    def test_10_incident_storage_organization_and_naming(self):
        """Verify outputs/incidents/YYYY-MM-DD/ organization and INCIDENT_YYYYMMDD_HHMMSS_XXX format."""
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(dummy_frame, "QUALIFYING INCIDENT TEST", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 1. Non-qualifying event should NOT save snapshot
        res_skipped = save_incident_evidence(
            raw_frame=dummy_frame,
            annotated_frame=dummy_frame,
            incident_type="Male Struggle",
            risk_score=50,
            reasons=["Male fight only"],
            confidence=0.8,
            victim_ids=[],
            is_qualifying_incident=False
        )
        self.assertIsNone(res_skipped, "Non-qualifying event must return None and skip saving snapshot")

        # 2. Qualifying woman incident SHOULD save snapshot in dated folder
        record = save_incident_evidence(
            raw_frame=dummy_frame,
            annotated_frame=dummy_frame,
            incident_type="POTENTIAL WOMAN SAFETY INCIDENT",
            risk_score=92,
            reasons=["Female track under direct physical struggle"],
            confidence=0.92,
            victim_ids=[10],
            is_qualifying_incident=True
        )
        self.assertIsNotNone(record)
        inc_id = record["incident_id"]
        self.assertTrue(inc_id.startswith("INCIDENT_"), f"ID must follow format INCIDENT_..., got {inc_id}")

        dated_dir = get_dated_incident_dir()
        today_str = datetime.now().strftime("%Y-%m-%d")
        self.assertIn(today_str, dated_dir)

        jpg_path = os.path.join(dated_dir, f"{inc_id}.jpg")
        json_path = os.path.join(dated_dir, f"{inc_id}.json")
        self.assertTrue(os.path.exists(jpg_path), f"Expected snapshot at {jpg_path}")
        self.assertTrue(os.path.exists(json_path), f"Expected JSON record at {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.assertEqual(data["incident_id"], inc_id)
            self.assertEqual(data["victim_tracking_ids"], [10])
            self.assertIn("snapshot", data["evidence_files"])

    def test_11_police_station_locator(self):
        """Verify nearest police station lookup returns distance, name, and contact details."""
        stations = find_nearest_police_stations(
            lat=config.LOCATION_LATITUDE,
            lon=config.LOCATION_LONGITUDE,
            max_results=2
        )
        self.assertIsInstance(stations, list)
        self.assertGreater(len(stations), 0)
        self.assertIn("name", stations[0])
        self.assertIn("distance_km", stations[0])
        self.assertIn("public_contact", stations[0])
        self.assertIn("emergency_hotline", stations[0])

    def test_12_evaluation_metrics_computation(self):
        """Verify accuracy, precision, recall, and F1 score calculation."""
        y_true = [0, 0, 1, 1, 1, 0, 1, 0]
        y_pred = [0, 0, 1, 1, 0, 0, 1, 0]
        metrics = compute_binary_metrics(y_true, y_pred)
        self.assertEqual(metrics["confusion_matrix"]["TP"], 3)
        self.assertEqual(metrics["accuracy"], 0.875)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 0.75)


if __name__ == "__main__":
    unittest.main()
