/**
 * Loading Manager for Kelpie Carbon v1
 * Handles progressive loading, caching, and loading states
 */

class LoadingManager {
    constructor() {
        this.imageCache = new Map();
        this.loadingStates = new Map();
        this.retryAttempts = new Map();
        this.maxRetries = 3;
        this.loadingIndicators = new Map();

        this.initializeLoadingStyles();
    }

    initializeLoadingStyles() {
        // Add loading styles to document head
        const style = document.createElement('style');
        style.textContent = `
            .loading-overlay {
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: rgba(255, 255, 255, 0.8);
                display: flex;
                align-items: center;
                justify-content: center;
                z-index: 2000;
                border-radius: 8px;
            }

            .loading-spinner {
                width: 40px;
                height: 40px;
                border: 4px solid #e2e8f0;
                border-top: 4px solid #667eea;
                border-radius: 50%;
                animation: spin 1s linear infinite;
            }

            .loading-text {
                margin-left: 12px;
                color: #4a5568;
                font-weight: 500;
            }

            .layer-loading {
                opacity: 0.5;
                transition: opacity 0.3s ease;
            }

            .layer-error {
                opacity: 0.3;
                filter: grayscale(100%);
            }

            .retry-button {
                background: #f56565;
                color: white;
                border: none;
                padding: 0.5rem 1rem;
                border-radius: 4px;
                cursor: pointer;
                font-size: 0.8rem;
                margin-left: 8px;
            }

            .retry-button:hover {
                background: #e53e3e;
            }

            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        `;
        document.head.appendChild(style);
    }

    // Show loading indicator for a specific area
    showLoading(containerId, message = 'Loading...') {
        const container = document.getElementById(containerId);
        if (!container) return;

        // Remove existing loading indicator
        this.hideLoading(containerId);

        const loadingOverlay = document.createElement('div');
        loadingOverlay.className = 'loading-overlay';
        loadingOverlay.dataset.loadingId = containerId;

        loadingOverlay.innerHTML = `
            <div class="loading-spinner"></div>
            <div class="loading-text">${message}</div>
        `;

        container.style.position = 'relative';
        container.appendChild(loadingOverlay);
        this.loadingIndicators.set(containerId, loadingOverlay);
    }

    // Hide loading indicator
    hideLoading(containerId) {
        const existingOverlay = this.loadingIndicators.get(containerId);
        if (existingOverlay && existingOverlay.parentNode) {
            existingOverlay.parentNode.removeChild(existingOverlay);
        }
        this.loadingIndicators.delete(containerId);
    }

    // Show error state with retry option
    showError(containerId, errorMessage, retryCallback = null) {
        this.hideLoading(containerId);

        const container = document.getElementById(containerId);
        if (!container) return;

        const errorOverlay = document.createElement('div');
        errorOverlay.className = 'loading-overlay';
        errorOverlay.style.background = 'rgba(254, 226, 226, 0.9)';

        let retryButton = '';
        if (retryCallback) {
            retryButton = '<button class="retry-button">Retry</button>';
        }

        errorOverlay.innerHTML = `
            <div style="text-align: center; color: #e53e3e;">
                <div style="font-weight: 600; margin-bottom: 8px;">⚠️ Error</div>
                <div style="font-size: 0.9rem; margin-bottom: 8px;">${errorMessage}</div>
                ${retryButton}
            </div>
        `;

        if (retryCallback) {
            const retryBtn = errorOverlay.querySelector('.retry-button');
            retryBtn.addEventListener('click', () => {
                this.hideLoading(containerId);
                retryCallback();
            });
        }

        container.appendChild(errorOverlay);
        this.loadingIndicators.set(containerId, errorOverlay);
    }

    // Progressive image loading with caching
    async loadImageWithCache(url, layerId) {
        // Check cache first
        if (this.imageCache.has(url)) {
            console.log(`Loading ${layerId} from cache`);
            return this.imageCache.get(url);
        }

        // Check if already loading
        if (this.loadingStates.has(url)) {
            return this.loadingStates.get(url);
        }

        // Start loading
        const loadingPromise = this.loadImageWithRetry(url, layerId);
        this.loadingStates.set(url, loadingPromise);

        try {
            const imageData = await loadingPromise;
            this.imageCache.set(url, imageData);
            this.loadingStates.delete(url);
            return imageData;
        } catch (error) {
            this.loadingStates.delete(url);
            throw error;
        }
    }

    // Load image with retry logic
    async loadImageWithRetry(url, layerId) {
        const maxRetries = this.maxRetries;
        let lastError;

        for (let attempt = 1; attempt <= maxRetries; attempt++) {
            try {
                console.log(`Loading ${layerId} (attempt ${attempt}/${maxRetries})`);

                const response = await fetch(url, {
                    cache: 'force-cache',
                    headers: {
                        'Cache-Control': 'max-age=3600'
                    }
                });

                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }

                const blob = await response.blob();
                const imageUrl = URL.createObjectURL(blob);

                // Pre-load the image to ensure it's valid
                await this.preloadImage(imageUrl);

                console.log(`Successfully loaded ${layerId}`);
                return imageUrl;

            } catch (error) {
                lastError = error;
                console.warn(`Failed to load ${layerId} (attempt ${attempt}):`, error);

                if (attempt < maxRetries) {
                    // Exponential backoff: 1s, 2s, 4s
                    const delay = Math.pow(2, attempt - 1) * 1000;
                    await this.sleep(delay);
                }
            }
        }

        throw new Error(`Failed to load ${layerId} after ${maxRetries} attempts: ${lastError.message}`);
    }

    // Pre-load image to verify it's valid
    preloadImage(src) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => resolve(img);
            img.onerror = () => reject(new Error('Image failed to load'));
            img.src = src;
        });
    }

    // Sleep utility
    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    // Progressive layer loading
    async loadLayersProgressively(analysisId, layerManager, controlsManager) {
        const layerLoadOrder = [
            { id: 'rgb', name: 'RGB Composite', priority: 1 },
            { id: 'kelp_mask', name: 'Kelp Detection', priority: 2 },
            { id: 'fai', name: 'FAI Spectral Index', priority: 3 },
            { id: 'ndre', name: 'NDRE Spectral Index', priority: 4 },
            { id: 'water_mask', name: 'Water Areas', priority: 5 },
            { id: 'cloud_mask', name: 'Cloud Coverage', priority: 6 }
        ];

        // Load high-priority layers first
        for (const layer of layerLoadOrder.sort((a, b) => a.priority - b.priority)) {
            try {
                this.showLoading('map', `Loading ${layer.name}...`);

                if (layer.id === 'rgb') {
                    const rgbLayer = layerManager.addRGBLayer(analysisId);
                    if (rgbLayer) {
                        console.log('🖼️ RGB layer created, waiting for load...');
                        await this.waitForLayerLoad(rgbLayer);

                        // Update layer bounds before adding to map
                        await layerManager.updateLayerBounds(analysisId);

                        // Add layer to map
                        console.log('🗺️ Adding RGB layer to map...');
                        if (!layerManager.map.hasLayer(rgbLayer)) {
                            layerManager.map.addLayer(rgbLayer);
                            console.log('✅ RGB layer successfully added to map');
                        } else {
                            console.log('⚠️ RGB layer already on map');
                        }
                    }
                } else if (layer.id.endsWith('_mask')) {
                    const maskType = layer.id.replace('_mask', '');
                    const maskLayer = layerManager.addMaskOverlay(maskType, analysisId);
                    if (maskLayer) {
                        await this.waitForLayerLoad(maskLayer);
                    }
                } else {
                    const spectralLayer = layerManager.addSpectralLayer(layer.id, analysisId);
                    if (spectralLayer) {
                        await this.waitForLayerLoad(spectralLayer);
                    }
                }

                console.log(`✅ Loaded ${layer.name}`);

                // Small delay between layers to prevent overwhelming the server
                await this.sleep(200);

            } catch (error) {
                console.error(`❌ Failed to load ${layer.name}:`, error);

                // Show error in controls but continue loading other layers
                if (controlsManager) {
                    this.showError('opacity-controls',
                        `Failed to load ${layer.name}`,
                        () => this.loadLayersProgressively(analysisId, layerManager, controlsManager)
                    );
                }
            }
        }

        this.hideLoading('map');
    }

    // Wait for layer to load
    waitForLayerLoad(layer) {
        return new Promise((resolve, reject) => {
            if (layer._url) {
                // For image overlays, wait for the image to load
                const img = new Image();
                img.onload = () => resolve(layer);
                img.onerror = () => reject(new Error('Layer image failed to load'));
                img.src = layer._url;
            } else {
                // For other layer types, resolve immediately
                resolve(layer);
            }
        });
    }

    // Clear cache (useful for memory management)
    clearCache() {
        // Revoke object URLs to free memory
        for (const [url, cachedUrl] of this.imageCache) {
            if (cachedUrl.startsWith('blob:')) {
                URL.revokeObjectURL(cachedUrl);
            }
        }

        this.imageCache.clear();
        this.loadingStates.clear();
        this.retryAttempts.clear();
        console.log('Image cache cleared');
    }

    // Get cache statistics
    getCacheStats() {
        return {
            cached_images: this.imageCache.size,
            active_loads: this.loadingStates.size,
            cache_size_mb: this.estimateCacheSize()
        };
    }

    // Estimate cache size (rough approximation)
    estimateCacheSize() {
        return Math.round(this.imageCache.size * 0.5); // Rough estimate: 0.5MB per image
    }
}

// Export for use in other modules
window.LoadingManager = LoadingManager;
