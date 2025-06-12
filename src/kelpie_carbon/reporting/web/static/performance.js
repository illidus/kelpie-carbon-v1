/**
 * Performance Monitor for Kelpie Carbon v1
 * Tracks loading times, memory usage, and system performance
 */

class PerformanceMonitor {
    constructor() {
        this.metrics = new Map();
        this.startTimes = new Map();
        this.performanceLog = [];
        this.maxLogEntries = 100;

        this.initializeMonitoring();
    }

    initializeMonitoring() {
        // Monitor page load performance
        if (performance && performance.navigation) {
            window.addEventListener('load', () => {
                this.recordPageLoad();
            });
        }

        // Monitor memory usage (if available)
        if (performance && performance.memory) {
            setInterval(() => {
                this.recordMemoryUsage();
            }, 30000); // Every 30 seconds
        }
    }

    // Start timing an operation
    startTimer(operationName) {
        this.startTimes.set(operationName, performance.now());
        console.log(`⏱️ Started: ${operationName}`);
    }

    // End timing and record results
    endTimer(operationName, additionalData = {}) {
        const startTime = this.startTimes.get(operationName);
        if (!startTime) {
            console.warn(`Timer not found for operation: ${operationName}`);
            return null;
        }

        const duration = performance.now() - startTime;
        const metric = {
            operation: operationName,
            duration: Math.round(duration),
            timestamp: new Date().toISOString(),
            ...additionalData
        };

        this.metrics.set(operationName, metric);
        this.addToLog(metric);
        this.startTimes.delete(operationName);

        console.log(`✅ Completed: ${operationName} (${metric.duration}ms)`);
        return metric;
    }

    // Record page load performance
    recordPageLoad() {
        const navigation = performance.getEntriesByType('navigation')[0];
        if (navigation) {
            const loadMetric = {
                operation: 'page_load',
                duration: Math.round(navigation.loadEventEnd - navigation.fetchStart),
                dom_content_loaded: Math.round(navigation.domContentLoadedEventEnd - navigation.fetchStart),
                first_paint: this.getFirstPaint(),
                timestamp: new Date().toISOString()
            };

            this.metrics.set('page_load', loadMetric);
            this.addToLog(loadMetric);
            console.log('📄 Page Load Metrics:', loadMetric);
        }
    }

    // Get first paint timing
    getFirstPaint() {
        const paintEntries = performance.getEntriesByType('paint');
        const firstPaint = paintEntries.find(entry => entry.name === 'first-paint');
        return firstPaint ? Math.round(firstPaint.startTime) : null;
    }

    // Record memory usage
    recordMemoryUsage() {
        if (performance.memory) {
            const memoryMetric = {
                operation: 'memory_usage',
                used_heap: Math.round(performance.memory.usedJSHeapSize / 1024 / 1024 * 100) / 100, // MB
                total_heap: Math.round(performance.memory.totalJSHeapSize / 1024 / 1024 * 100) / 100, // MB
                heap_limit: Math.round(performance.memory.jsHeapSizeLimit / 1024 / 1024 * 100) / 100, // MB
                timestamp: new Date().toISOString()
            };

            this.metrics.set('memory_usage', memoryMetric);

            // Only log if memory usage is high
            if (memoryMetric.used_heap > 50) {
                console.log('🧠 Memory Usage:', memoryMetric);
            }
        }
    }

    // Record API request performance
    recordApiRequest(url, duration, success = true, error = null) {
        const metric = {
            operation: 'api_request',
            url: url,
            duration: Math.round(duration),
            success: success,
            error: error,
            timestamp: new Date().toISOString()
        };

        this.addToLog(metric);

        if (success) {
            console.log(`🌐 API Request: ${url} (${metric.duration}ms)`);
        } else {
            console.error(`❌ API Request Failed: ${url} (${metric.duration}ms)`, error);
        }

        return metric;
    }

    // Record image loading performance
    recordImageLoad(layerId, size, duration, fromCache = false) {
        const metric = {
            operation: 'image_load',
            layer_id: layerId,
            size_mb: Math.round(size / 1024 / 1024 * 100) / 100,
            duration: Math.round(duration),
            from_cache: fromCache,
            timestamp: new Date().toISOString()
        };

        this.addToLog(metric);
        console.log(`🖼️ Image Load: ${layerId} (${metric.duration}ms, ${fromCache ? 'cached' : 'network'})`);
        return metric;
    }

    // Add metric to log with rotation
    addToLog(metric) {
        this.performanceLog.push(metric);

        // Rotate log to prevent memory bloat
        if (this.performanceLog.length > this.maxLogEntries) {
            this.performanceLog = this.performanceLog.slice(-this.maxLogEntries);
        }
    }

    // Get performance summary
    getPerformanceSummary() {
        const summary = {
            page_load: this.metrics.get('page_load'),
            memory_usage: this.metrics.get('memory_usage'),
            recent_operations: this.performanceLog.slice(-10),
            cache_stats: this.getCacheEfficiency(),
            total_operations: this.performanceLog.length
        };

        return summary;
    }

    // Calculate cache efficiency
    getCacheEfficiency() {
        const imageLoads = this.performanceLog.filter(m => m.operation === 'image_load');
        if (imageLoads.length === 0) return null;

        const cachedLoads = imageLoads.filter(m => m.from_cache).length;
        const efficiency = Math.round((cachedLoads / imageLoads.length) * 100);

        return {
            total_loads: imageLoads.length,
            cached_loads: cachedLoads,
            cache_hit_rate: `${efficiency}%`,
            avg_load_time: Math.round(imageLoads.reduce((sum, m) => sum + m.duration, 0) / imageLoads.length)
        };
    }

    // Monitor API calls with automatic timing
    async monitorApiCall(url, fetchOptions = {}) {
        const startTime = performance.now();
        let success = true;
        let error = null;

        try {
            const response = await fetch(url, fetchOptions);
            success = response.ok;

            if (!response.ok) {
                error = `HTTP ${response.status}: ${response.statusText}`;
            }

            const duration = performance.now() - startTime;
            this.recordApiRequest(url, duration, success, error);

            return response;
        } catch (err) {
            success = false;
            error = err.message;
            const duration = performance.now() - startTime;
            this.recordApiRequest(url, duration, success, error);
            throw err;
        }
    }

    // Export performance data
    exportPerformanceData() {
        const exportData = {
            summary: this.getPerformanceSummary(),
            full_log: this.performanceLog,
            metrics: Object.fromEntries(this.metrics),
            export_timestamp: new Date().toISOString(),
            user_agent: navigator.userAgent,
            viewport: {
                width: window.innerWidth,
                height: window.innerHeight
            }
        };

        // Create download link
        const dataStr = JSON.stringify(exportData, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        const url = URL.createObjectURL(dataBlob);

        const link = document.createElement('a');
        link.href = url;
        link.download = `kelpie-performance-${Date.now()}.json`;
        link.click();

        URL.revokeObjectURL(url);
        console.log('📊 Performance data exported');
    }

    // Show performance dashboard
    showDashboard() {
        const summary = this.getPerformanceSummary();
        const dashboardHtml = `
            <div style="position: fixed; top: 10px; right: 10px; background: white; border: 1px solid #ccc;
                        padding: 15px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                        z-index: 3000; max-width: 300px; font-family: monospace; font-size: 12px;">
                <h4 style="margin: 0 0 10px 0; color: #333;">🔍 Performance Dashboard</h4>
                <div><strong>Page Load:</strong> ${summary.page_load?.duration || 'N/A'}ms</div>
                <div><strong>Memory:</strong> ${summary.memory_usage?.used_heap || 'N/A'}MB</div>
                <div><strong>Operations:</strong> ${summary.total_operations}</div>
                <div><strong>Cache Hit Rate:</strong> ${summary.cache_stats?.cache_hit_rate || 'N/A'}</div>
                <button onclick="performanceMonitor.exportPerformanceData()"
                        style="margin-top: 10px; padding: 5px 10px; border: none; background: #667eea;
                               color: white; border-radius: 4px; cursor: pointer;">
                    Export Data
                </button>
                <button onclick="this.parentElement.remove()"
                        style="margin-top: 5px; margin-left: 5px; padding: 5px 10px; border: none;
                               background: #e53e3e; color: white; border-radius: 4px; cursor: pointer;">
                    Close
                </button>
            </div>
        `;

        // Remove existing dashboard
        const existing = document.querySelector('.performance-dashboard');
        if (existing) existing.remove();

        // Add new dashboard
        const dashboardDiv = document.createElement('div');
        dashboardDiv.className = 'performance-dashboard';
        dashboardDiv.innerHTML = dashboardHtml;
        document.body.appendChild(dashboardDiv);
    }

    // Clear all performance data
    clearData() {
        this.metrics.clear();
        this.startTimes.clear();
        this.performanceLog = [];
        console.log('🧹 Performance data cleared');
    }
}

// Create global performance monitor
window.performanceMonitor = new PerformanceMonitor();

// Add keyboard shortcut to show dashboard (Ctrl+Shift+P)
document.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.shiftKey && e.key === 'P') {
        window.performanceMonitor.showDashboard();
    }
});

// Export for use in other modules
window.PerformanceMonitor = PerformanceMonitor;
