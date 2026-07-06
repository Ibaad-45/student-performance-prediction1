/**
 * main.js
 * --------
 * Shared front-end logic for the Student Performance Predictor:
 *   - Toast notification system (used on every page)
 *   - Live slider "echo" badges on the prediction form
 *   - Form submission -> POST /api/predict -> animate the gauge + badges
 *   - Hero stat strip populated from /api/metrics
 *
 * Defensive by design: every DOM lookup is guarded so this single file
 * can be safely included on pages that don't have all these elements.
 */

(function () {
    "use strict";

    // ----------------------------------------------------------------
    // Toast notifications
    // ----------------------------------------------------------------
    function showToast(message, type = "info") {
        const container = document.getElementById("toast-container");
        if (!container) return;

        const toast = document.createElement("div");
        toast.className = `toast toast-${type}`;
        toast.textContent = message;
        container.appendChild(toast);

        setTimeout(() => {
            toast.classList.add("is-leaving");
            setTimeout(() => toast.remove(), 260);
        }, 4200);
    }
    window.showToast = showToast; // exposed for dashboard.js reuse

    // ----------------------------------------------------------------
    // Slider echo badges (e.g. "Study Hours / Week: 15")
    // ----------------------------------------------------------------
    document.querySelectorAll('input[type="range"][data-echo]').forEach((slider) => {
        const echoEl = document.querySelector(`[data-echo="${slider.name}"]`);
        const suffix = slider.dataset.suffix || "";
        const updateEcho = () => {
            if (echoEl) echoEl.textContent = `${slider.value}${suffix}`;
        };
        slider.addEventListener("input", updateEcho);
        updateEcho();
    });

    // ----------------------------------------------------------------
    // Hero stat strip (R2 / accuracy) -- populated from saved metrics
    // ----------------------------------------------------------------
    const r2Stat = document.querySelector('[data-stat="r2"]');
    const accStat = document.querySelector('[data-stat="accuracy"]');
    if (r2Stat || accStat) {
        fetch("/api/metrics")
            .then((res) => res.json())
            .then((json) => {
                if (!json.success) return;
                if (r2Stat) r2Stat.textContent = json.data.regression.test_r2_score;
                if (accStat) accStat.textContent = `${Math.round(json.data.classification.accuracy * 100)}%`;
            })
            .catch(() => { /* silently ignore -- hero still renders fine without live stats */ });
    }

    // ----------------------------------------------------------------
    // Gauge animation helpers
    // ----------------------------------------------------------------
    function setGaugeScore(score) {
        const needle = document.getElementById("gauge-needle");
        const scoreEl = document.getElementById("gauge-score");
        if (!needle || !scoreEl) return;

        // Semicircle spans -90deg (score=0) to +90deg (score=100)
        const clamped = Math.max(0, Math.min(100, score));
        const angle = -90 + (clamped / 100) * 180;
        needle.style.transform = `rotate(${angle}deg)`;

        // Animate the numeric readout counting up
        const duration = 900;
        const start = performance.now();
        const from = 0;
        function step(now) {
            const progress = Math.min(1, (now - start) / duration);
            const eased = 1 - Math.pow(1 - progress, 3);
            scoreEl.textContent = (from + (clamped - from) * eased).toFixed(1);
            if (progress < 1) requestAnimationFrame(step);
            else scoreEl.textContent = clamped.toFixed(1);
        }
        requestAnimationFrame(step);

        const scoreColor = clamped >= 60 ? "#3D7A54" : clamped >= 40 ? "#C89B3C" : "#B4453B";
        scoreEl.style.color = scoreColor;
    }

    // ----------------------------------------------------------------
    // Prediction form submission
    // ----------------------------------------------------------------
    const form = document.getElementById("predict-form");
    if (form) {
        const btn = document.getElementById("predict-btn");
        const errorBox = document.getElementById("form-error");

        form.addEventListener("submit", async function (e) {
            e.preventDefault();
            errorBox.hidden = true;

            const formData = new FormData(form);
            const payload = Object.fromEntries(formData.entries());

            // ---- Client-side validation using server-provided ranges ----
            const ranges = window.NUMERIC_RANGES || {};
            const clientErrors = [];
            Object.entries(ranges).forEach(([field, bounds]) => {
                if (payload[field] === undefined) return;
                const value = parseFloat(payload[field]);
                const [low, high] = bounds;
                if (Number.isNaN(value) || value < low || value > high) {
                    clientErrors.push(`${field.replace(/_/g, " ")} must be between ${low} and ${high}.`);
                }
            });

            if (clientErrors.length > 0) {
                errorBox.textContent = clientErrors.join(" ");
                errorBox.hidden = false;
                showToast("Please fix the highlighted fields.", "error");
                return;
            }

            btn.classList.add("is-loading");
            btn.disabled = true;

            try {
                const response = await fetch("/api/predict", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload),
                });
                const result = await response.json();

                if (!response.ok || !result.success) {
                    throw new Error(result.error || "Prediction failed. Please try again.");
                }

                const data = result.data;
                setGaugeScore(data.predicted_score);

                const gradeEl = document.getElementById("grade-value");
                const passFailEl = document.getElementById("passfail-value");
                const probEl = document.getElementById("probability-value");
                const remarkEl = document.getElementById("result-remark");

                if (gradeEl) gradeEl.textContent = data.predicted_grade;
                if (passFailEl) {
                    passFailEl.textContent = data.pass_fail;
                    passFailEl.classList.remove("is-pass", "is-fail");
                    passFailEl.classList.add(data.pass_fail === "Pass" ? "is-pass" : "is-fail");
                }
                if (probEl) probEl.textContent = `${data.pass_probability}%`;
                if (remarkEl) remarkEl.textContent = data.remark;

                showToast("Prediction generated successfully.", "success");
            } catch (err) {
                errorBox.textContent = err.message || "Something went wrong. Please try again.";
                errorBox.hidden = false;
                showToast(err.message || "Prediction failed.", "error");
            } finally {
                btn.classList.remove("is-loading");
                btn.disabled = false;
            }
        });
    }
})();
