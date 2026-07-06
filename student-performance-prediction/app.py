"""
app.py
------
Flask backend for the Student Performance Prediction web app.

Routes:
    GET  /                  -> Home page with the prediction form
    GET  /dashboard         -> Analytics dashboard (model metrics + charts + history)
    POST /api/predict       -> REST API: predict a student's performance
    GET  /api/history       -> REST API: recent prediction history
    GET  /api/metrics       -> REST API: saved model evaluation metrics
    GET  /api/stats         -> REST API: aggregate stats over stored predictions
    POST /api/history/clear -> REST API: clears prediction history (demo utility)
    GET  /api/health        -> REST API: health check

Run (development):
    python app.py
Run (production):
    gunicorn app:app
"""

import os
import sys
import logging
from flask import Flask, render_template, request, jsonify

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from config import get_config  # noqa: E402
import database  # noqa: E402
from src.predict import predict_student_performance, VALID_CATEGORIES, NUMERIC_RANGES  # noqa: E402
from src.utils import load_metrics  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(get_config())

# Initialize the SQLite database (creates the table if missing) at startup.
with app.app_context():
    database.init_db()


# --------------------------------------------------------------------------
# Page routes
# --------------------------------------------------------------------------

@app.route("/")
def home():
    """Renders the main prediction form page."""
    return render_template(
        "index.html",
        categories=VALID_CATEGORIES,
        ranges=NUMERIC_RANGES,
    )


@app.route("/dashboard")
def dashboard():
    """Renders the analytics dashboard: model metrics, charts, and history."""
    metrics = load_metrics()
    return render_template("dashboard.html", metrics=metrics)


# --------------------------------------------------------------------------
# REST API routes
# --------------------------------------------------------------------------

@app.route("/api/health", methods=["GET"])
def api_health():
    """Simple health check endpoint -- useful for uptime monitors and
    deployment platforms (Render/Railway) that ping this to confirm
    the service is alive."""
    return jsonify({"status": "ok", "service": "student-performance-prediction"}), 200


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Accepts student feature data (JSON or form-encoded) and returns
    the predicted final exam score, letter grade, and pass/fail outcome.
    Every successful prediction is logged to the SQLite history table.
    """
    try:
        data = request.get_json(silent=True) or request.form.to_dict()

        if not data:
            return jsonify({
                "success": False,
                "error": "No input data provided. Send a JSON body or form data with student details."
            }), 400

        result = predict_student_performance(data)

        # Persist to history (non-fatal if it fails -- prediction still returned)
        try:
            database.insert_prediction(data, result)
        except Exception as db_err:  # pragma: no cover
            logger.warning(f"Could not save prediction to history: {db_err}")

        return jsonify({"success": True, "data": result}), 200

    except ValueError as ve:
        # Raised by predict_student_performance() on invalid input
        return jsonify({"success": False, "error": str(ve)}), 400

    except FileNotFoundError as fe:
        # Raised if models haven't been trained yet
        logger.error(f"Model artifacts missing: {fe}")
        return jsonify({
            "success": False,
            "error": "Model is not trained yet. Run `python src/train_model.py` on the server first."
        }), 503

    except Exception as e:  # pragma: no cover
        logger.exception("Unexpected error during prediction")
        return jsonify({"success": False, "error": "An unexpected server error occurred. Please try again."}), 500


@app.route("/api/history", methods=["GET"])
def api_history():
    """Returns the most recent predictions for the dashboard's history table."""
    try:
        limit = request.args.get("limit", default=app.config["HISTORY_LIMIT"], type=int)
        limit = max(1, min(limit, 500))  # clamp to a sane range
        history = database.get_recent_predictions(limit=limit)
        return jsonify({"success": True, "data": history, "count": len(history)}), 200
    except Exception as e:  # pragma: no cover
        logger.exception("Error fetching history")
        return jsonify({"success": False, "error": "Could not fetch prediction history."}), 500


@app.route("/api/stats", methods=["GET"])
def api_stats():
    """Returns aggregate stats (total predictions, avg score, pass rate)
    used for the dashboard's live counters."""
    try:
        stats = database.get_summary_stats()
        return jsonify({"success": True, "data": stats}), 200
    except Exception as e:  # pragma: no cover
        logger.exception("Error fetching stats")
        return jsonify({"success": False, "error": "Could not fetch statistics."}), 500


@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    """Returns the saved model evaluation metrics (R2, MAE, RMSE, accuracy,
    precision, recall, F1) generated at training time."""
    try:
        metrics = load_metrics()
        if not metrics:
            return jsonify({
                "success": False,
                "error": "No metrics found. Train the model first with `python src/train_model.py`."
            }), 404
        return jsonify({"success": True, "data": metrics}), 200
    except Exception as e:  # pragma: no cover
        logger.exception("Error fetching metrics")
        return jsonify({"success": False, "error": "Could not fetch model metrics."}), 500


@app.route("/api/history/clear", methods=["POST"])
def api_clear_history():
    """Clears all stored prediction history. Intended as a demo/reset
    utility (e.g. before a viva presentation)."""
    try:
        database.clear_history()
        return jsonify({"success": True, "message": "Prediction history cleared."}), 200
    except Exception as e:  # pragma: no cover
        logger.exception("Error clearing history")
        return jsonify({"success": False, "error": "Could not clear history."}), 500


# --------------------------------------------------------------------------
# Error handlers
# --------------------------------------------------------------------------

@app.errorhandler(404)
def not_found(e):
    if request.path.startswith("/api/"):
        return jsonify({"success": False, "error": "Endpoint not found."}), 404
    return render_template("404.html"), 404


@app.errorhandler(500)
def server_error(e):
    logger.error(f"Internal server error: {e}")
    if request.path.startswith("/api/"):
        return jsonify({"success": False, "error": "Internal server error."}), 500
    return render_template("500.html"), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=app.config["DEBUG"])
