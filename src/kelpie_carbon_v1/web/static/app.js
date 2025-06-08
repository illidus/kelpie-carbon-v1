// Initialize map centered on British Columbia, Canada
const map = L.map('map').setView([49.2827, -123.1207], 8);

// Add OpenStreetMap tiles
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors'
}).addTo(map);

// Variables to store AOI selection
let selectedAOI = null;
let aoiMarker = null;

// Elements
const runButton = document.getElementById('run-analysis');
const statusDiv = document.getElementById('status');
const resultsDiv = document.getElementById('results');
const resultsContent = document.getElementById('results-content');
const startDateInput = document.getElementById('start-date');
const endDateInput = document.getElementById('end-date');

// Map click handler for AOI selection
map.on('click', function(e) {
    const lat = e.latlng.lat.toFixed(6);
    const lng = e.latlng.lng.toFixed(6);

    // Remove existing marker
    if (aoiMarker) {
        map.removeLayer(aoiMarker);
    }

    // Add new marker
    aoiMarker = L.marker([lat, lng])
        .addTo(map)
        .bindPopup(`AOI Selected<br>Lat: ${lat}<br>Lng: ${lng}`)
        .openPopup();

    // Store selected AOI
    selectedAOI = { lat: parseFloat(lat), lng: parseFloat(lng) };

    // Enable run button
    runButton.disabled = false;
    statusDiv.textContent = 'AOI selected. Ready to run analysis.';
});

// Run analysis button handler
runButton.addEventListener('click', async function() {
    if (!selectedAOI) {
        alert('Please select an AOI first');
        return;
    }

    const startDate = startDateInput.value;
    const endDate = endDateInput.value;

    if (!startDate || !endDate) {
        alert('Please select date range');
        return;
    }

    // Disable button and show loading
    runButton.disabled = true;
    statusDiv.textContent = 'Running analysis...';
    resultsDiv.style.display = 'none';

    try {
        // Call API endpoint
        const response = await fetch('/api/run', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                aoi: selectedAOI,
                start_date: startDate,
                end_date: endDate
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();

        // Display results
        displayResults(result);
        statusDiv.textContent = 'Analysis complete!';

    } catch (error) {
        console.error('Error running analysis:', error);
        statusDiv.textContent = 'Error running analysis. Please try again.';
        alert('Error running analysis: ' + error.message);
    } finally {
        runButton.disabled = false;
    }
});

// Function to display results
function displayResults(result) {
    resultsContent.innerHTML = '';

    // Create result items
    const resultItems = [
        { label: 'Analysis ID', value: result.analysis_id || 'N/A' },
        { label: 'AOI Coordinates', value: `${selectedAOI.lat}, ${selectedAOI.lng}` },
        { label: 'Date Range', value: `${startDateInput.value} to ${endDateInput.value}` },
        { label: 'Status', value: result.status || 'Unknown' },
        { label: 'Processing Time', value: result.processing_time || 'N/A' },
        { label: 'Biomass Estimate', value: result.biomass || 'Not available' },
        { label: 'Carbon Sequestration', value: result.carbon || 'Not available' }
    ];

    resultItems.forEach(item => {
        const resultItem = document.createElement('div');
        resultItem.className = 'result-item';
        resultItem.innerHTML = `
            <span class="result-label">${item.label}:</span>
            <span class="result-value">${item.value}</span>
        `;
        resultsContent.appendChild(resultItem);
    });

    resultsDiv.style.display = 'block';
}
