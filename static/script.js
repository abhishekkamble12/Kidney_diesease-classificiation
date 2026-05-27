document.addEventListener('DOMContentLoaded', () => {
    // Tab Switching Logic
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            // Remove active class from all
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            // Add active class to clicked
            btn.classList.add('active');
            const targetId = btn.getAttribute('data-target');
            document.getElementById(targetId).classList.add('active');
        });
    });

    // Helper to format JSON nicely
    const formatJSON = (obj) => JSON.stringify(obj, null, 2);

    // Predict API Call
    const predictBtn = document.getElementById('predict-btn');
    const predictInput = document.getElementById('predict-input');
    const predictResultBox = document.getElementById('predict-result');
    const predictContent = document.getElementById('predict-content');

    predictBtn.addEventListener('click', async () => {
        const text = predictInput.value.trim();
        if (!text) return alert("Please enter some text!");

        predictBtn.textContent = 'Analyzing...';
        predictBtn.disabled = true;
        predictResultBox.classList.add('hidden');

        try {
            const response = await fetch(`/predict?text=${encodeURIComponent(text)}`, {
                method: 'POST'
            });
            const data = await response.json();
            predictContent.textContent = formatJSON(data);
            predictResultBox.classList.remove('hidden');
        } catch (err) {
            predictContent.textContent = `Error: ${err.message}`;
            predictResultBox.classList.remove('hidden');
        } finally {
            predictBtn.textContent = 'Analyze Sentiment';
            predictBtn.disabled = false;
        }
    });

    // Search Analysis API Call
    const searchBtn = document.getElementById('search-btn');
    const searchInput = document.getElementById('search-input');
    const searchResultBox = document.getElementById('search-result');
    const searchContent = document.getElementById('search-content');

    searchBtn.addEventListener('click', async () => {
        const query = searchInput.value.trim();
        if (!query) return alert("Please enter a search query!");

        searchBtn.textContent = 'Generating...';
        searchBtn.disabled = true;
        searchResultBox.classList.add('hidden');

        try {
            const response = await fetch('/search-analysis', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query })
            });
            const data = await response.json();
            searchContent.textContent = formatJSON(data);
            searchResultBox.classList.remove('hidden');
        } catch (err) {
            searchContent.textContent = `Error: ${err.message}`;
            searchResultBox.classList.remove('hidden');
        } finally {
            searchBtn.textContent = 'Generate Report';
            searchBtn.disabled = false;
        }
    });

    // Monitoring API Call
    const refreshMonitoringBtn = document.getElementById('refresh-monitoring-btn');
    const monitoringGrid = document.getElementById('monitoring-grid');

    const fetchMonitoring = async () => {
        refreshMonitoringBtn.textContent = 'Refreshing...';
        refreshMonitoringBtn.disabled = true;
        monitoringGrid.innerHTML = ''; // clear

        try {
            const response = await fetch('/monitoring');
            const data = await response.json();
            
            const metrics = [
                { label: 'Avg Latency', value: `${data.avg_latency_ms} ms` },
                { label: 'Avg Confidence', value: `${(data.avg_confidence * 100).toFixed(1)}%` },
                { label: 'Fallback Rate', value: `${(data.fallback_rate * 100).toFixed(1)}%` }
            ];

            // Render top level metrics
            metrics.forEach(m => {
                monitoringGrid.innerHTML += `
                    <div class="metric-card">
                        <h3>${m.label}</h3>
                        <div class="value">${m.value}</div>
                    </div>
                `;
            });

            // Render distribution as a card
            let distHtml = '<div class="metric-card"><h3>Prediction Distribution</h3><div style="text-align: left; margin-top: 10px;">';
            const dist = data.prediction_distribution;
            for (const [key, val] of Object.entries(dist)) {
                let label = key === "-1" ? "Negative" : (key === "0" ? "Neutral" : "Positive");
                distHtml += `<div><strong>${label}:</strong> ${val}</div>`;
            }
            distHtml += '</div></div>';
            monitoringGrid.innerHTML += distHtml;

        } catch (err) {
            monitoringGrid.innerHTML = `<div style="color: var(--error)">Failed to fetch monitoring data: ${err.message}</div>`;
        } finally {
            refreshMonitoringBtn.textContent = 'Refresh Stats';
            refreshMonitoringBtn.disabled = false;
        }
    };

    refreshMonitoringBtn.addEventListener('click', fetchMonitoring);
    
    // Initial fetch for monitoring
    fetchMonitoring();
});
