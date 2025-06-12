/**
 * Interactive Controls Manager for Kelpie Carbon v1
 * Manages opacity sliders, dynamic legend, and metadata display
 */

class ImageryControlsManager {
    constructor() {
        this.controlPanel = document.getElementById('layer-control-panel');
        this.toggleButton = document.getElementById('toggle-controls');
        this.closeButton = document.getElementById('close-panel');
        this.opacityContainer = document.getElementById('opacity-controls');
        this.legendContainer = document.getElementById('legend-container');
        this.metadataContainer = document.getElementById('metadata-container');
        
        this.activeLayers = new Map(); // Store layer info for opacity controls
        this.currentMetadata = null;
        
        this.initializeEventHandlers();
    }

    initializeEventHandlers() {
        // Toggle button to show/hide control panel
        this.toggleButton.addEventListener('click', () => {
            this.showPanel();
        });

        // Close button to hide panel
        this.closeButton.addEventListener('click', () => {
            this.hidePanel();
        });

        // Close panel when clicking outside (optional)
        document.addEventListener('click', (e) => {
            if (this.controlPanel.style.display === 'block' && 
                !this.controlPanel.contains(e.target) && 
                !this.toggleButton.contains(e.target)) {
                this.hidePanel();
            }
        });
    }

    showPanel() {
        this.controlPanel.style.display = 'block';
        this.toggleButton.style.display = 'none';
    }

    hidePanel() {
        this.controlPanel.style.display = 'none';
        this.toggleButton.style.display = 'block';
    }

    showControls() {
        this.toggleButton.style.display = 'block';
        console.log('Toggle button should now be visible at top-left');
        // Force visibility for debugging
        this.toggleButton.style.backgroundColor = '#667eea';
        this.toggleButton.style.color = 'white';
        this.toggleButton.style.padding = '0.75rem 1rem';
        this.toggleButton.style.borderRadius = '8px';
        this.toggleButton.style.fontWeight = '600';
    }

    hideControls() {
        this.toggleButton.style.display = 'none';
        this.controlPanel.style.display = 'none';
    }

    // Add a layer to opacity controls
    addLayerOpacityControl(layerId, layerName, initialOpacity = 1.0, layerObject = null) {
        // Remove existing control if it exists
        this.removeLayerOpacityControl(layerId);

        const controlDiv = document.createElement('div');
        controlDiv.className = 'opacity-control';
        controlDiv.dataset.layerId = layerId;

        const opacityValue = Math.round(initialOpacity * 100);

        controlDiv.innerHTML = `
            <div class="opacity-label">${layerName}</div>
            <input type="range" class="opacity-slider" 
                   min="0" max="100" value="${opacityValue}" 
                   data-layer-id="${layerId}">
            <div class="opacity-value">${opacityValue}%</div>
        `;

        // Add event listener for opacity changes
        const slider = controlDiv.querySelector('.opacity-slider');
        const valueDisplay = controlDiv.querySelector('.opacity-value');

        slider.addEventListener('input', (e) => {
            const opacity = parseFloat(e.target.value) / 100;
            valueDisplay.textContent = `${e.target.value}%`;
            
            // Update layer opacity
            if (layerObject && layerObject.setOpacity) {
                layerObject.setOpacity(opacity);
            }
            
            // Emit custom event for external handlers
            this.emitOpacityChange(layerId, opacity);
        });

        this.opacityContainer.appendChild(controlDiv);
        
        // Store layer info
        this.activeLayers.set(layerId, {
            name: layerName,
            layerObject: layerObject,
            opacity: initialOpacity
        });
    }

    // Remove layer from opacity controls
    removeLayerOpacityControl(layerId) {
        const existingControl = this.opacityContainer.querySelector(`[data-layer-id="${layerId}"]`);
        if (existingControl) {
            existingControl.remove();
        }
        this.activeLayers.delete(layerId);
    }

    // Clear all opacity controls
    clearOpacityControls() {
        this.opacityContainer.innerHTML = '';
        this.activeLayers.clear();
    }

    // Update dynamic legend based on active layers
    updateLegend(layerTypes = {}) {
        // Clear existing dynamic legend items
        const existingDynamic = this.legendContainer.querySelectorAll('.dynamic-legend');
        existingDynamic.forEach(item => item.remove());

        // Add spectral index legends
        if (layerTypes.spectral && layerTypes.spectral.length > 0) {
            layerTypes.spectral.forEach(indexName => {
                this.addSpectralLegendItem(indexName);
            });
        }

        // Show/hide static legend items based on active layers
        this.updateStaticLegendVisibility(layerTypes);
    }

    addSpectralLegendItem(indexName) {
        const legendItem = document.createElement('div');
        legendItem.className = 'legend-item dynamic-legend';
        
        const displayNames = {
            'ndvi': 'NDVI',
            'fai': 'FAI (Floating Algae)',
            'red_edge_ndvi': 'Red Edge NDVI', 
            'ndre': 'NDRE',
            'ndwi': 'NDWI',
            'evi': 'EVI'
        };

        const colorClasses = {
            'ndvi': 'ndvi-color',
            'fai': 'fai-color',
            'red_edge_ndvi': 'ndre-color',
            'ndre': 'ndre-color',
            'ndwi': 'water-color',
            'evi': 'ndvi-color'
        };

        legendItem.innerHTML = `
            <span class="legend-color ${colorClasses[indexName] || 'ndvi-color'}"></span>
            <span>${displayNames[indexName] || indexName.toUpperCase()}</span>
        `;

        this.legendContainer.appendChild(legendItem);
    }

    updateStaticLegendVisibility(layerTypes) {
        // Show/hide kelp legend
        const legendItems = this.legendContainer.querySelectorAll('.legend-item');
        
        // Handle different data types for masks
        let masks = [];
        if (layerTypes.masks) {
            if (Array.isArray(layerTypes.masks)) {
                masks = layerTypes.masks;
            } else if (typeof layerTypes.masks === 'object') {
                masks = Object.keys(layerTypes.masks);
            } else if (typeof layerTypes.masks === 'string' && layerTypes.masks.length > 0) {
                masks = layerTypes.masks.split(',').map(s => s.trim());
            }
        }
        
        legendItems.forEach(item => {
            const kelpColor = item.querySelector('.kelp-color');
            const waterColor = item.querySelector('.water-color');
            const cloudColor = item.querySelector('.cloud-color');
            
            if (kelpColor) {
                item.style.display = masks.includes('kelp') ? 'flex' : 'none';
            } else if (waterColor) {
                item.style.display = masks.includes('water') ? 'flex' : 'none';
            } else if (cloudColor) {
                item.style.display = masks.includes('cloud') ? 'flex' : 'none';
            }
        });
    }

    // Update metadata display
    async updateMetadata(analysisId) {
        if (!analysisId) {
            this.clearMetadata();
            return;
        }

        try {
            const response = await fetch(`/api/imagery/${analysisId}/metadata`);
            const metadata = await response.json();
            
            this.currentMetadata = metadata;
            this.displayMetadata(metadata);
        } catch (error) {
            console.error('Failed to load metadata:', error);
            this.clearMetadata();
        }
    }

    displayMetadata(metadata) {
        // Update date
        const dateElement = document.getElementById('metadata-date');
        if (metadata.acquisition_date) {
            const date = new Date(metadata.acquisition_date);
            dateElement.textContent = date.toLocaleDateString();
        } else {
            dateElement.textContent = 'Unknown';
        }

        // Update resolution
        const resolutionElement = document.getElementById('metadata-resolution');
        resolutionElement.textContent = metadata.resolution ? `${metadata.resolution}m` : 'Unknown';

        // Update cloud coverage
        const cloudsElement = document.getElementById('metadata-clouds');
        if (metadata.cloud_coverage !== undefined) {
            cloudsElement.textContent = `${(metadata.cloud_coverage * 100).toFixed(2)}%`;
        } else {
            cloudsElement.textContent = 'Unknown';
        }

        // Update bounds
        const boundsElement = document.getElementById('metadata-bounds');
        if (metadata.bounds && metadata.bounds.length === 4) {
            const [minLon, minLat, maxLon, maxLat] = metadata.bounds;
            boundsElement.textContent = `${minLat.toFixed(4)}, ${minLon.toFixed(4)} to ${maxLat.toFixed(4)}, ${maxLon.toFixed(4)}`;
        } else {
            boundsElement.textContent = 'Unknown';
        }
    }

    clearMetadata() {
        document.getElementById('metadata-date').textContent = '-';
        document.getElementById('metadata-resolution').textContent = '-';
        document.getElementById('metadata-clouds').textContent = '-';
        document.getElementById('metadata-bounds').textContent = '-';
        this.currentMetadata = null;
    }

    // Emit custom event for opacity changes
    emitOpacityChange(layerId, opacity) {
        const event = new CustomEvent('layerOpacityChange', {
            detail: { layerId, opacity }
        });
        document.dispatchEvent(event);
    }

    // Get current opacity for a layer
    getLayerOpacity(layerId) {
        const layerInfo = this.activeLayers.get(layerId);
        return layerInfo ? layerInfo.opacity : 1.0;
    }

    // Set opacity for a layer programmatically
    setLayerOpacity(layerId, opacity) {
        const slider = this.opacityContainer.querySelector(`[data-layer-id="${layerId}"]`);
        if (slider) {
            slider.value = Math.round(opacity * 100);
            slider.dispatchEvent(new Event('input'));
        }
    }

    // Reset all layer opacities to default
    resetOpacities() {
        const sliders = this.opacityContainer.querySelectorAll('.opacity-slider');
        sliders.forEach(slider => {
            slider.value = 70; // Default to 70%
            slider.dispatchEvent(new Event('input'));
        });
    }
}

// Export for use in other modules
window.ImageryControlsManager = ImageryControlsManager; 