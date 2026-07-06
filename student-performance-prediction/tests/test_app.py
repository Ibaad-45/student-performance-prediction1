"""
test_app.py
------------
Automated tests for the Flask application using Python's built-in
`unittest` framework (no extra dependency needed to run these; pytest
can also discover and run them if installed, since pytest is fully
compatible with unittest-style TestCase classes).

Run:
    python -m unittest tests/test_app.py -v
    (or, if pytest is installed)
    pytest tests/test_app.py -v
"""

import os
import sys
import unittest
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as flask_app_module  # noqa: E402
import database  # noqa: E402
from config import Config  # noqa: E402


class StudentPerformanceAppTests(unittest.TestCase):
    """Covers page rendering, the prediction API (valid + invalid input),
    history/stats endpoints, and error handling."""

    @classmethod
    def setUpClass(cls):
        # Use a temporary, isolated SQLite database for the test run so
        # tests never pollute (or depend on) real prediction history.
        cls.temp_db_fd, cls.temp_db_path = tempfile.mkstemp(suffix=".db")
        Config.DATABASE_PATH = cls.temp_db_path

        flask_app_module.app.config["TESTING"] = True
        cls.client = flask_app_module.app.test_client()

        with flask_app_module.app.app_context():
            database.init_db()

        cls.valid_student = {
            "gender": "Female",
            "age": 19,
            "study_hours_per_week": 18,
            "attendance_percentage": 91,
            "previous_exam_score": 78,
            "parental_education": "Masters",
            "family_income_level": "Medium",
            "internet_access": "Yes",
            "extracurricular_activities": "Yes",
            "sleep_hours": 7.5,
            "tutoring": "Yes",
            "part_time_job": "No",
            "screen_time_hours": 2.5,
            "parental_involvement": "High",
        }

    @classmethod
    def tearDownClass(cls):
        os.close(cls.temp_db_fd)
        if os.path.exists(cls.temp_db_path):
            os.remove(cls.temp_db_path)

    # ---- Page rendering ----

    def test_home_page_loads(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Predict Performance", response.data)

    def test_dashboard_page_loads(self):
        response = self.client.get("/dashboard")
        self.assertEqual(response.status_code, 200)

    def test_unknown_page_returns_404(self):
        response = self.client.get("/this-page-does-not-exist")
        self.assertEqual(response.status_code, 404)

    # ---- Health check ----

    def test_health_check(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")

    # ---- Prediction API: happy path ----

    def test_predict_with_valid_input_returns_200(self):
        response = self.client.post("/api/predict", json=self.valid_student)
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertTrue(body["success"])
        self.assertIn("predicted_score", body["data"])
        self.assertIn("predicted_grade", body["data"])
        self.assertIn("pass_fail", body["data"])
        self.assertGreaterEqual(body["data"]["predicted_score"], 0)
        self.assertLessEqual(body["data"]["predicted_score"], 100)
        self.assertIn(body["data"]["pass_fail"], ["Pass", "Fail"])

    # ---- Prediction API: validation errors ----

    def test_predict_with_empty_body_returns_400(self):
        response = self.client.post("/api/predict", json={})
        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.get_json()["success"])

    def test_predict_with_missing_fields_returns_400(self):
        incomplete = {"gender": "Female", "age": 19}
        response = self.client.post("/api/predict", json=incomplete)
        self.assertEqual(response.status_code, 400)
        body = response.get_json()
        self.assertFalse(body["success"])
        self.assertIn("required", body["error"])

    def test_predict_with_out_of_range_age_returns_400(self):
        bad_student = dict(self.valid_student)
        bad_student["age"] = 999
        response = self.client.post("/api/predict", json=bad_student)
        self.assertEqual(response.status_code, 400)
        self.assertIn("age", response.get_json()["error"])

    def test_predict_with_invalid_category_returns_400(self):
        bad_student = dict(self.valid_student)
        bad_student["gender"] = "Alien"
        response = self.client.post("/api/predict", json=bad_student)
        self.assertEqual(response.status_code, 400)
        self.assertIn("gender", response.get_json()["error"])

    def test_predict_with_non_numeric_value_returns_400(self):
        bad_student = dict(self.valid_student)
        bad_student["study_hours_per_week"] = "not-a-number"
        response = self.client.post("/api/predict", json=bad_student)
        self.assertEqual(response.status_code, 400)

    # ---- History & stats ----

    def test_history_endpoint_reflects_new_prediction(self):
        self.client.post("/api/predict", json=self.valid_student)
        response = self.client.get("/api/history?limit=10")
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertTrue(body["success"])
        self.assertGreaterEqual(len(body["data"]), 1)

    def test_stats_endpoint_returns_expected_keys(self):
        response = self.client.get("/api/stats")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()["data"]
        self.assertIn("total_predictions", data)
        self.assertIn("average_score", data)
        self.assertIn("pass_rate_pct", data)

    def test_clear_history_endpoint(self):
        self.client.post("/api/predict", json=self.valid_student)
        response = self.client.post("/api/history/clear")
        self.assertEqual(response.status_code, 200)
        history = self.client.get("/api/history").get_json()["data"]
        self.assertEqual(len(history), 0)

    # ---- Metrics ----

    def test_metrics_endpoint_returns_model_scores(self):
        response = self.client.get("/api/metrics")
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertTrue(body["success"])
        self.assertIn("regression", body["data"])
        self.assertIn("classification", body["data"])


if __name__ == "__main__":
    unittest.main()
