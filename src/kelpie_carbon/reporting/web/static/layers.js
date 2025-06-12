/**
 * Satellite Layer Manager for Kelpie Carbon v1
 * Manages satellite imagery layers and overlays on the Leaflet map
 */

class SatelliteLayerManager {
    constructor(map, controlsManager = null) {
        this.map = map;
        this.layers = new Map();
        this.currentAnalysisId = null;
        this.layerControl = null;
        this.baseLayers = {};
        this.overlayLayers = {};
        this.controlsManager = controlsManager;
        
        this.initializeLayerControl();
    }

    initializeLayerControl() {
        // Create layer control
        this.layerControl = L.control.layers(this.baseLayers, this.overlayLayers, {
            position: 'topright',
            collapsed: false
        });
        this.layerControl.addTo(this.map);
    }

    setAnalysisId(analysisId) {
        this.currentAnalysisId = analysisId;
        this.clearAllLayers();
        
        // Update controls manager metadata
        if (this.controlsManager) {
            this.controlsManager.updateMetadata(analysisId);
        }
    }

    clearAllLayers() {
        // Remove all satellite layers
        this.layers.forEach((layer, layerId) => {
            if (this.map.hasLayer(layer)) {
                this.map.removeLayer(layer);
            }
            this.layerControl.removeLayer(layer);
        });
        this.layers.clear();
        this.baseLayers = {};
        this.overlayLayers = {};
        
        // Clear controls manager
        if (this.controlsManager) {
            this.controlsManager.clearOpacityControls();
            this.controlsManager.clearMetadata();
            this.controlsManager.hideControls();
        }
    }

    async addRGBLayer(analysisId = null) {
        const id = analysisId || this.currentAnalysisId;
        if (!id) return null;

        const layerId = 'rgb_composite';
        const imageUrl = `/api/imagery/${id}/rgb`;
        
        // Get proper bounds before creating layer
        let bounds = [[0, 0], [1, 1]]; // fallback
        try {
            const response = await fetch(`/api/imagery/${id}/metadata`);
            const metadata = await response.json();
            if (metadata.bounds && metadata.bounds.length === 4) {
                const [minLon, minLat, maxLon, maxLat] = metadata.bounds;
                bounds = [[minLat, minLon], [maxLat, maxLon]];
            }
        } catch (error) {
            console.warn('Failed to load bounds for RGB layer, using fallback:', error);
        }
        
        // Create image overlay with proper bounds
        const layer = L.imageOverlay(imageUrl, bounds, {
            opacity: 1.0,
            attribution: 'Sentinel-2 RGB Composite'
        });

        this.layers.set(layerId, layer);
        this.baseLayers['True Color (RGB)'] = layer;
        this.layerControl.addBaseLayer(layer, 'True Color (RGB)');
        
        // Add to controls manager
        if (this.controlsManager) {
            this.controlsManager.addLayerOpacityControl(layerId, 'True Color (RGB)', 1.0, layer);
        }
        
        return layer;
    }

    async addFalseColorLayer(analysisId = null) {
        const id = analysisId || this.currentAnalysisId;
        if (!id) return null;

        const layerId = 'false_color';
        const imageUrl = `/api/imagery/${id}/false-color`;
        
        // Get proper bounds before creating layer
        let bounds = [[0, 0], [1, 1]]; // fallback
        try {
            const response = await fetch(`/api/imagery/${id}/metadata`);
            const metadata = await response.json();
            if (metadata.bounds && metadata.bounds.length === 4) {
                const [minLon, minLat, maxLon, maxLat] = metadata.bounds;
                bounds = [[minLat, minLon], [maxLat, maxLon]];
            }
        } catch (error) {
            console.warn('Failed to load bounds for false color layer, using fallback:', error);
        }
        
        const layer = L.imageOverlay(imageUrl, bounds, {
            opacity: 1.0,
            attribution: 'Sentinel-2 False Color Composite'
        });

        this.layers.set(layerId, layer);
        this.baseLayers['False Color (NIR-R-G)'] = layer;
        this.layerControl.addBaseLayer(layer, 'False Color (NIR-R-G)');
        
        // Add to controls manager
        if (this.controlsManager) {
            this.controlsManager.addLayerOpacityControl(layerId, 'False Color (NIR-R-G)', 1.0, layer);
        }
        
        return layer;
    }

    async addSpectralLayer(indexName, analysisId = null) {
        const id = analysisId || this.currentAnalysisId;
        if (!id) return null;

        const layerId = `spectral_${indexName}`;
        const imageUrl = `/api/imagery/${id}/spectral/${indexName}`;
        
        const displayNames = {
            'ndvi': 'NDVI',
            'fai': 'FAI (Floating Algae Index)',
            'red_edge_ndvi': 'Red Edge NDVI',
            'ndre': 'NDRE',
            'ndwi': 'NDWI',
            'evi': 'EVI',
            'kelp_index': 'Kelp Index'
        };

        // Get proper bounds before creating layer
        let bounds = [[0, 0], [1, 1]]; // fallback
        try {
            const response = await fetch(`/api/imagery/${id}/metadata`);
            const metadata = await response.json();
            if (metadata.bounds && metadata.bounds.length === 4) {
                const [minLon, minLat, maxLon, maxLat] = metadata.bounds;
                bounds = [[minLat, minLon], [maxLat, maxLon]];
            }
        } catch (error) {
            console.warn(`Failed to load bounds for ${indexName} spectral layer, using fallback:`, error);
        }

        const layer = L.imageOverlay(imageUrl, bounds, {
            opacity: 0.7,
            attribution: `Sentinel-2 ${displayNames[indexName] || indexName.toUpperCase()}`
        });

        this.layers.set(layerId, layer);
        this.overlayLayers[displayNames[indexName] || indexName.toUpperCase()] = layer;
        this.layerControl.addOverlay(layer, displayNames[indexName] || indexName.toUpperCase());
        
        // Add to controls manager
        if (this.controlsManager) {
            this.controlsManager.addLayerOpacityControl(layerId, displayNames[indexName] || indexName.toUpperCase(), 0.7, layer);
        }
        
        return layer;
    }

    async addMaskOverlay(maskType, analysisId = null, alpha = 0.6) {
        const id = analysisId || this.currentAnalysisId;
        if (!id) return null;

        const layerId = `mask_${maskType}`;
        const imageUrl = `/api/imagery/${id}/mask/${maskType}?alpha=${alpha}`;
        
        const displayNames = {
            'kelp': 'Kelp Detection',
            'water': 'Water Areas',
            'cloud': 'Cloud Coverage'
        };

        // Get proper bounds before creating layer
        let bounds = [[0, 0], [1, 1]]; // fallback
        try {
            const response = await fetch(`/api/imagery/${id}/metadata`);
            const metadata = await response.json();
            if (metadata.bounds && metadata.bounds.length === 4) {
                const [minLon, minLat, maxLon, maxLat] = metadata.bounds;
                bounds = [[minLat, minLon], [maxLat, maxLon]];
            }
        } catch (error) {
            console.warn(`Failed to load bounds for ${maskType} mask layer, using fallback:`, error);
        }

        const layer = L.imageOverlay(imageUrl, bounds, {
            opacity: 1.0, // Alpha is handled server-side
            attribution: `${displayNames[maskType] || maskType} Mask`
        });

        this.layers.set(layerId, layer);
        this.overlayLayers[displayNames[maskType] || maskType] = layer;
        this.layerControl.addOverlay(layer, displayNames[maskType] || maskType);
        
        // Add to controls manager
        if (this.controlsManager) {
            this.controlsManager.addLayerOpacityControl(layerId, displayNames[maskType] || maskType, 1.0, layer);
        }
        
        return layer;
    }

    addBiomassHeatmap(analysisId = null, colormap = 'hot') {
        const id = analysisId || this.currentAnalysisId;
        if (!id) return null;

        const layerId = 'biomass_heatmap';
        const imageUrl = `/api/imagery/${id}/biomass?colormap=${colormap}`;
        
        const layer = L.imageOverlay(imageUrl, [[0, 0], [1, 1]], {
            opacity: 0.8,
            attribution: 'Kelp Biomass Density'
        });

        this.layers.set(layerId, layer);
        this.overlayLayers['Biomass Heatmap'] = layer;
        this.layerControl.addOverlay(layer, 'Biomass Heatmap');
        
        // Add to controls manager
        if (this.controlsManager) {
            this.controlsManager.addLayerOpacityControl(layerId, 'Biomass Heatmap', 0.8, layer);
        }
        
        return layer;
    }

    async updateLayerBounds(analysisId = null) {
        const id = analysisId || this.currentAnalysisId;
        if (!id) return;

        try {
            const response = await fetch(`/api/imagery/${id}/metadata`);
            const metadata = await response.json();
            
            if (metadata.bounds && metadata.bounds.length === 4) {
                const [minLon, minLat, maxLon, maxLat] = metadata.bounds;
                const bounds = [[minLat, minLon], [maxLat, maxLon]];
                
                console.log('🌍 Setting layer bounds:', bounds);
                
                // Update bounds for all layers
                this.layers.forEach((layer, layerId) => {
                    if (layer.setBounds) {
                        layer.setBounds(bounds);
                        console.log(`📍 Updated bounds for layer: ${layerId}`);
                    }
                });

                // Fit map to bounds
                this.map.fitBounds(bounds, { padding: [20, 20] });
                console.log('🗺️ Map fitted to bounds');
            }
        } catch (error) {
            console.error('Failed to update layer bounds:', error);
        }
    }

    toggleLayer(layerId) {
        const layer = this.layers.get(layerId);
        if (!layer) return;

        if (this.map.hasLayer(layer)) {
            this.map.removeLayer(layer);
        } else {
            this.map.addLayer(layer);
        }
    }

    setLayerOpacity(layerId, opacity) {
        const layer = this.layers.get(layerId);
        if (layer && layer.setOpacity) {
            layer.setOpacity(opacity);
        }
    }

    getLayer(layerId) {
        return this.layers.get(layerId);
    }

    getAllLayers() {
        return Array.from(this.layers.values());
    }

    async loadAllLayers(analysisId) {
        this.setAnalysisId(analysisId);
        
        try {
            // Get metadata to determine available layers
            const response = await fetch(`/api/imagery/${analysisId}/metadata`);
            const metadata = await response.json();
            
            // Add base layers (now async)
            await this.addRGBLayer(analysisId);
            await this.addFalseColorLayer(analysisId);
            
            // Add spectral index layers (now async)
            if (metadata.available_layers.spectral_indices) {
                for (const indexName of Object.keys(metadata.available_layers.spectral_indices)) {
                    await this.addSpectralLayer(indexName, analysisId);
                }
            }
            
            // Add mask overlays (now async) - map API names to endpoint names
            if (metadata.available_layers.masks) {
                for (const maskType of Object.keys(metadata.available_layers.masks)) {
                    // Map API mask names to actual endpoint names
                    const endpointName = maskType.replace('_mask', ''); // kelp_mask -> kelp, water_mask -> water
                    await this.addMaskOverlay(endpointName, analysisId);
                }
            }
            
            // Add biomass heatmap if available
            if (metadata.available_layers.biomass) {
                this.addBiomassHeatmap(analysisId);
            }
            
            // Show RGB layer by default (it should already have correct bounds)
            const rgbLayer = this.getLayer('rgb_composite');
            if (rgbLayer && !this.map.hasLayer(rgbLayer)) {
                console.log('🖼️ Adding RGB layer to map...');
                this.map.addLayer(rgbLayer);
                console.log('✅ RGB layer added to map');
                
                // Fit map to the RGB layer bounds
                if (rgbLayer.getBounds && rgbLayer.getBounds() && rgbLayer.getBounds().isValid()) {
                    this.map.fitBounds(rgbLayer.getBounds(), { padding: [20, 20] });
                    console.log('🗺️ Map fitted to RGB layer bounds');
                }
            }
            
            return metadata;
            
        } catch (error) {
            console.error('Failed to load imagery layers:', error);
            throw error;
        }
    }

    // Event handlers for layer control
    onLayerAdd(callback) {
        this.map.on('overlayadd', callback);
        this.map.on('baselayerchange', callback);
    }

    onLayerRemove(callback) {
        this.map.on('overlayremove', callback);
    }
}

// Export for use in other modules
window.SatelliteLayerManager = SatelliteLayerManager; 