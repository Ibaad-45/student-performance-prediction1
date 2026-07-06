/**
 * dashboard.js
 * -------------
 * Populates the dashboard's live counters and prediction history table
 * by calling the REST API, and wires up the "Clear history" button.
 * Kept separate from main.js since these elements only exist on
 * /dashboard.
 */

(function () {
    "use strict";

    const totalEl = document.getElementById("stat-total");
    const avgEl = document.getElementById("stat-avg");
    const passRateEl = document.getElementById("stat-pass-rate");
    const historyBody = document.getElementById("history-body");
    const clearBtn = document.getElementById("clear-history-btn");

    if (!totalEl && !historyBody) return; // not on the dashboard page

    function formatTime(isoString) {
        try {
            const date = new Date(isoString);
            return date.toLocaleString(undefined, {
                month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
            });
        } catch {
            return isoString;
        }
    }

    function renderHistory(rows) {
        if (!historyBody) return;
        if (!rows || rows.length === 0) {
            historyBody.innerHTML = '<tr><td colspan="7" class="empty-row">No predictions yet &mdash; try the <a href="/">Predict</a> page.</td></tr>';
            return;
        }

        historyBody.innerHTML = rows.map((row) => {
            const input = row.input_json || {};
            const outcomeClass = row.pass_fail === "Pass" ? "is-pass" : "is-fail";
            return `
                <tr>
                    <td>${formatTime(row.created_at)}</td>
                    <td>${input.study_hours_per_week ?? "--"}</td>
                    <td>${input.attendance_percentage ?? "--"}%</td>
                    <td>${input.previous_exam_score ?? "--"}</td>
                    <td>${row.predicted_score}</td>
                    <td>${row.predicted_grade}</td>
                    <td><span class="result-badge-value ${outcomeClass}">${row.pass_fail}</span></td>
                </tr>`;
        }).join("");
    }

    function loadStats() {
        fetch("/api/stats")
            .then((res) => res.json())
            .then((json) => {
                if (!json.success) return;
                const s = json.data;
                if (totalEl) totalEl.textContent = s.total_predictions;
                if (avgEl) avgEl.textContent = s.average_score;
                if (passRateEl) passRateEl.textContent = `${s.pass_rate_pct}%`;
            })
            .catch(() => showToastSafe("Could not load live stats.", "error"));
    }

    function loadHistory() {
        fetch("/api/history?limit=25")
            .then((res) => res.json())
            .then((json) => {
                if (!json.success) return;
                renderHistory(json.data);
            })
            .catch(() => showToastSafe("Could not load prediction history.", "error"));
    }

    function showToastSafe(message, type) {
        if (typeof window.showToast === "function") window.showToast(message, type);
    }

    if (clearBtn) {
        clearBtn.addEventListener("click", async () => {
            if (!confirm("Clear all prediction history? This cannot be undone.")) return;
            try {
                const res = await fetch("/api/history/clear", { method: "POST" });
                const json = await res.json();
                if (json.success) {
                    showToastSafe("Prediction history cleared.", "success");
                    loadStats();
                    loadHistory();
                } else {
                    showToastSafe(json.error || "Could not clear history.", "error");
                }
            } catch {
                showToastSafe("Could not clear history.", "error");
            }
        });
    }

    loadStats();
    loadHistory();
})();
