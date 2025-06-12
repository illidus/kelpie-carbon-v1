// Initialize map centered on British Columbia, Canada
const map = L.map('map').setView([49.2827, -123.1207], 8);

// Add OpenStreetMap tiles
const osmLayer = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors'
}).addTo(map);

// Initialize satellite layer manager and controls
let satelliteLayerManager = null;
let imageryControlsManager = null;
let loadingManager = null;

// Variables to store AOI selection
let selectedAOI = null;
let aoiMarker = null;
let currentAnalysisId = null;

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
        // Step 1: Run traditional analysis
        statusDiv.textContent = 'Running analysis...';
        const analysisResponse = await fetch('/api/run', {
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

        if (!analysisResponse.ok) {
            throw new Error(`HTTP error! status: ${analysisResponse.status}`);
        }

        const analysisResult = await analysisResponse.json();

        // Step 2: Generate and cache imagery
        statusDiv.textContent = 'Generating satellite imagery...';
        
        // Start timer if available
        if (window.performanceMonitor && typeof window.performanceMonitor.startTimer === 'function') {
            performanceMonitor.startTimer('imagery_generation');
        }
        
        console.log('🚀 Starting imagery analysis with request:', {
            aoi: {
                lat: selectedAOI.lat,
                lng: selectedAOI.lng
            },
            start_date: startDate,
            end_date: endDate,
            buffer_km: 1.0,
            max_cloud_cover: 0.3
        });
        
        // Use performanceMonitor if available, otherwise fall back to regular fetch
        let imageryResponse;
        if (window.performanceMonitor && typeof window.performanceMonitor.monitorApiCall === 'function') {
            imageryResponse = await performanceMonitor.monitorApiCall('/api/imagery/analyze-and-cache', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    aoi: {
                        lat: selectedAOI.lat,
                        lng: selectedAOI.lng
                    },
                    start_date: startDate,
                    end_date: endDate,
                    buffer_km: 1.0,
                    max_cloud_cover: 0.3
                })
            });
        } else {
            console.warn('⚠️ performanceMonitor not available, using regular fetch');
            imageryResponse = await fetch('/api/imagery/analyze-and-cache', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    aoi: {
                        lat: selectedAOI.lat,
                        lng: selectedAOI.lng
                    },
                    start_date: startDate,
                    end_date: endDate,
                    buffer_km: 1.0,
                    max_cloud_cover: 0.3
                })
            });
        }

        console.log('📡 Imagery response status:', imageryResponse.status);
        console.log('📡 Imagery response headers:', imageryResponse.headers);

        if (!imageryResponse.ok) {
            const errorText = await imageryResponse.text();
            console.error('❌ Imagery generation failed:', {
                status: imageryResponse.status,
                statusText: imageryResponse.statusText,
                errorText: errorText
            });
            console.warn('Continuing with text results only');
        }

        let imageryResult = null;
        try {
            imageryResult = await imageryResponse.json();
            currentAnalysisId = imageryResult.analysis_id;
            
            // End timer if available
            if (window.performanceMonitor && typeof window.performanceMonitor.endTimer === 'function') {
                performanceMonitor.endTimer('imagery_generation', { 
                    analysis_id: currentAnalysisId,
                    success: true 
                });
            }
        } catch (e) {
            console.warn('Failed to parse imagery response');
            
            // End timer if available
            if (window.performanceMonitor && typeof window.performanceMonitor.endTimer === 'function') {
                performanceMonitor.endTimer('imagery_generation', { 
                    success: false,
                    error: e.message 
                });
            }
        }

        // Step 3: Display results and load imagery
        displayResults(analysisResult, imageryResult);
        
        if (imageryResult && imageryResult.analysis_id) {
            // Start timer if available
            if (window.performanceMonitor && typeof window.performanceMonitor.startTimer === 'function') {
                performanceMonitor.startTimer('satellite_imagery_loading');
            }
            
            await loadSatelliteImagery(imageryResult.analysis_id);
            
            // End timer if available
            if (window.performanceMonitor && typeof window.performanceMonitor.endTimer === 'function') {
                performanceMonitor.endTimer('satellite_imagery_loading', {
                    analysis_id: imageryResult.analysis_id,
                    success: true
                });
            }
            statusDiv.textContent = 'Analysis complete with satellite imagery!';
        } else {
            statusDiv.textContent = 'Analysis complete!';
        }

    } catch (error) {
        console.error('Error running analysis:', error);
        statusDiv.textContent = 'Error running analysis. Please try again.';
        alert('Error running analysis: ' + error.message);
    } finally {
        runButton.disabled = false;
    }
});

// Function to display results
function displayResults(analysisResult, imageryResult = null) {
    resultsContent.innerHTML = '';

    // Create result items
    const resultItems = [
        { label: 'Analysis ID', value: analysisResult.analysis_id || 'N/A' },
        { label: 'AOI Coordinates', value: `${selectedAOI.lat}, ${selectedAOI.lng}` },
        { label: 'Date Range', value: `${startDateInput.value} to ${endDateInput.value}` },
        { label: 'Status', value: analysisResult.status || 'Unknown' },
        { label: 'Processing Time', value: analysisResult.processing_time || 'N/A' },
        { label: 'Biomass Estimate', value: analysisResult.biomass || 'Not available' },
        { label: 'Carbon Sequestration', value: analysisResult.carbon || 'Not available' }
    ];

    // Add imagery information if available
    if (imageryResult) {
        resultItems.push(
            { label: 'Imagery ID', value: imageryResult.analysis_id || 'N/A' },
            { label: 'Available Layers', value: getAvailableLayersText(imageryResult) }
        );
    }

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

// Function to get available layers text
function getAvailableLayersText(imageryResult) {
    if (!imageryResult.available_layers) return 'None';
    
    const layers = [];
    if (imageryResult.available_layers.base_layers) {
        layers.push(`Base: ${imageryResult.available_layers.base_layers.join(', ')}`);
    }
    if (imageryResult.available_layers.spectral_indices) {
        layers.push(`Spectral: ${imageryResult.available_layers.spectral_indices.join(', ')}`);
    }
    if (imageryResult.available_layers.masks) {
        layers.push(`Masks: ${imageryResult.available_layers.masks.join(', ')}`);
    }
    if (imageryResult.available_layers.biomass) {
        layers.push('Biomass heatmap');
    }
    
    return layers.join('; ') || 'None';
}

// Function to load satellite imagery with progressive loading
async function loadSatelliteImagery(analysisId) {
    console.log('🖼️ Starting loadSatelliteImagery with ID:', analysisId);
    
    try {
        // Check if required classes are available
        console.log('🔍 Checking required classes...');
        console.log('LoadingManager available:', typeof LoadingManager !== 'undefined');
        console.log('ImageryControlsManager available:', typeof ImageryControlsManager !== 'undefined');
        console.log('SatelliteLayerManager available:', typeof SatelliteLayerManager !== 'undefined');
        
        // Initialize managers if not already done
        if (!loadingManager) {
            console.log('🔧 Initializing LoadingManager...');
            if (typeof LoadingManager !== 'undefined') {
                loadingManager = new LoadingManager();
                console.log('✅ LoadingManager initialized');
            } else {
                console.error('❌ LoadingManager class not available');
                throw new Error('LoadingManager class not found');
            }
        }
        
        if (!imageryControlsManager) {
            console.log('🔧 Initializing ImageryControlsManager...');
            if (typeof ImageryControlsManager !== 'undefined') {
                imageryControlsManager = new ImageryControlsManager();
                console.log('✅ ImageryControlsManager initialized');
            } else {
                console.error('❌ ImageryControlsManager class not available');
                throw new Error('ImageryControlsManager class not found');
            }
        }
        
        if (!satelliteLayerManager) {
            console.log('🔧 Initializing SatelliteLayerManager...');
            if (typeof SatelliteLayerManager !== 'undefined') {
                satelliteLayerManager = new SatelliteLayerManager(map, imageryControlsManager);
                console.log('✅ SatelliteLayerManager initialized');
            } else {
                console.error('❌ SatelliteLayerManager class not available');
                throw new Error('SatelliteLayerManager class not found');
            }
        }

        // Use progressive loading for better performance
        console.log('🔄 Starting progressive loading...');
        loadingManager.showLoading('map', 'Loading satellite imagery...');
        
        // Load metadata first
        const metadataUrl = `/api/imagery/${analysisId}/metadata`;
        console.log('🔍 Loading metadata from:', metadataUrl);
        const metadataResponse = await fetch(metadataUrl);
        
        if (!metadataResponse.ok) {
            throw new Error(`Failed to load metadata: ${metadataResponse.status} ${metadataResponse.statusText}`);
        }
        
        const metadata = await metadataResponse.json();
        console.log('✅ Metadata loaded:', metadata);
        
        // Update controls with metadata
        console.log('🎛️ Updating controls with metadata...');
        imageryControlsManager.updateMetadata(analysisId);
        imageryControlsManager.showControls();
        console.log('✅ Controls updated');
        
        // Progressive layer loading
        console.log('🖼️ Starting progressive layer loading...');
        await loadingManager.loadLayersProgressively(analysisId, satelliteLayerManager, imageryControlsManager);
        console.log('✅ Progressive layer loading completed');
        
        // Update legend based on loaded layers
        if (metadata && metadata.available_layers) {
            console.log('📋 Updating legend...');
            imageryControlsManager.updateLegend({
                spectral: metadata.available_layers.spectral_indices || [],
                masks: metadata.available_layers.masks || []
            });
            console.log('✅ Legend updated');
        }
        
        // Add layer control event handlers
        satelliteLayerManager.onLayerAdd((e) => {
            console.log('✅ Layer added:', e.name);
        });
        
        satelliteLayerManager.onLayerRemove((e) => {
            console.log('➖ Layer removed:', e.name);
        });
        
        // Log performance statistics
        const cacheStats = loadingManager.getCacheStats();
        console.log('✅ All satellite imagery layers loaded successfully');
        console.log('📊 Cache Stats:', cacheStats);
        
    } catch (error) {
        console.error('❌ Failed to load satellite imagery:', error);
        console.error('❌ Error stack:', error.stack);
        statusDiv.textContent += ' (Imagery loading failed)';
        
        // Show error with retry option
        if (loadingManager && typeof loadingManager.showError === 'function') {
            loadingManager.showError('map', 
                'Failed to load satellite imagery. Please try again.',
                () => loadSatelliteImagery(analysisId)
            );
        } else {
            console.error('❌ LoadingManager.showError not available');
            // Simple fallback - try to load RGB image directly
            console.log('🔄 Attempting simple RGB fallback...');
            try {
                const rgbUrl = `/api/imagery/${analysisId}/rgb`;
                console.log('🖼️ Loading RGB image from:', rgbUrl);
                
                // Create a simple image overlay as fallback
                const bounds = [[36.79, -121.91], [36.81, -121.89]]; // Approximate bounds
                const imageOverlay = L.imageOverlay(rgbUrl, bounds, {
                    opacity: 0.8,
                    alt: 'Satellite RGB Image'
                }).addTo(map);
                
                console.log('✅ Simple RGB image loaded as fallback');
                
                // Fit map to bounds
                map.fitBounds(bounds);
                
            } catch (fallbackError) {
                console.error('❌ Even simple fallback failed:', fallbackError);
            }
        }
    }
}

// Function to clear satellite imagery
function clearSatelliteImagery() {
    if (satelliteLayerManager) {
        satelliteLayerManager.clearAllLayers();
    }
    if (imageryControlsManager) {
        imageryControlsManager.hideControls();
    }
    if (loadingManager) {
        loadingManager.clearCache();
        loadingManager.hideLoading('map');
    }
    currentAnalysisId = null;
    console.log('🧹 Satellite imagery cleared and cache cleaned');
}
