"""Performance monitoring utilities for Kelpie Carbon v1."""

import time
import psutil
import functools
from typing import Any, Callable, Dict, List, Optional, TypeVar, cast
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
import threading
import logging

logger = logging.getLogger(__name__)

F = TypeVar('F', bound=Callable[..., Any])


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    
    function_name: str
    execution_time: float
    memory_usage_mb: float
    peak_memory_mb: float
    timestamp: datetime
    args_count: int
    kwargs_count: int
    success: bool
    error_message: Optional[str] = None


class PerformanceMonitor:
    """Performance monitoring class for tracking function execution metrics."""
    
    def __init__(self, max_history: int = 1000):
        """Initialize performance monitor.
        
        Args:
            max_history: Maximum number of performance records to keep
        """
        self.max_history = max_history
        self.metrics_history: List[PerformanceMetrics] = []
        self.lock = threading.Lock()
        self._process = psutil.Process()
    
    def record_metrics(self, metrics: PerformanceMetrics) -> None:
        """Record performance metrics.
        
        Args:
            metrics: PerformanceMetrics object to record
        """
        with self.lock:
            self.metrics_history.append(metrics)
            
            # Maintain max history size
            if len(self.metrics_history) > self.max_history:
                self.metrics_history.pop(0)
    
    def get_function_stats(self, function_name: str) -> Dict[str, Any]:
        """Get performance statistics for a specific function.
        
        Args:
            function_name: Name of the function to get stats for
            
        Returns:
            Dictionary containing performance statistics
        """
        try:
            with self.lock:
                function_metrics = [
                    m for m in self.metrics_history 
                    if m.function_name == function_name
                ]
            
            if not function_metrics:
                return {"error": f"No metrics found for function: {function_name}"}
            
            execution_times = [m.execution_time for m in function_metrics]
            memory_usages = [m.memory_usage_mb for m in function_metrics]
            success_count = sum(1 for m in function_metrics if m.success)
            
            return {
                "function_name": function_name,
                "total_calls": len(function_metrics),
                "successful_calls": success_count,
                "success_rate": success_count / len(function_metrics) if function_metrics else 0.0,
                "execution_time": {
                    "min": min(execution_times) if execution_times else 0.0,
                    "max": max(execution_times) if execution_times else 0.0,
                    "avg": sum(execution_times) / len(execution_times) if execution_times else 0.0,
                    "total": sum(execution_times) if execution_times else 0.0,
                },
                "memory_usage": {
                    "min": min(memory_usages) if memory_usages else 0.0,
                    "max": max(memory_usages) if memory_usages else 0.0,
                    "avg": sum(memory_usages) / len(memory_usages) if memory_usages else 0.0,
                },
                "last_called": max(m.timestamp for m in function_metrics) if function_metrics else None,
            }
        except Exception as e:
            logger.warning(f"Error getting function stats for {function_name}: {e}")
            return {"error": f"Failed to get stats for {function_name}: {str(e)}"}
    
    def get_overall_stats(self) -> Dict[str, Any]:
        """Get overall performance statistics.
        
        Returns:
            Dictionary containing overall performance statistics
        """
        try:
            with self.lock:
                if not self.metrics_history:
                    return {"error": "No metrics recorded"}
                
                # Simplified stats without recursive calls
                total_execution_time = sum(m.execution_time for m in self.metrics_history)
                total_calls = len(self.metrics_history)
                successful_calls = sum(1 for m in self.metrics_history if m.success)
                function_names = set(m.function_name for m in self.metrics_history)
                
                return {
                    "total_functions_monitored": len(function_names),
                    "total_calls": total_calls,
                    "successful_calls": successful_calls,
                    "overall_success_rate": successful_calls / total_calls if total_calls > 0 else 0.0,
                    "total_execution_time": total_execution_time,
                    "average_execution_time": total_execution_time / total_calls if total_calls > 0 else 0.0,
                    "monitoring_period": {
                        "start": min(m.timestamp for m in self.metrics_history),
                        "end": max(m.timestamp for m in self.metrics_history),
                    } if self.metrics_history else None,
                    "function_names": list(function_names),
                }
        except Exception as e:
            logger.warning(f"Error getting performance stats: {e}")
            return {"error": f"Failed to get stats: {str(e)}"}
    
    def clear_history(self) -> None:
        """Clear all recorded metrics."""
        with self.lock:
            self.metrics_history.clear()


# Global performance monitor instance
_global_monitor = PerformanceMonitor()


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    return _global_monitor


@contextmanager
def timing_context(name: str = "operation"):
    """Context manager for timing operations.
    
    Args:
        name: Name of the operation being timed
        
    Yields:
        Dictionary that will contain timing results
    """
    start_time = time.time()
    start_memory = memory_usage()
    result = {"name": name}
    
    try:
        yield result
        success = True
        error_msg = None
    except Exception as e:
        success = False
        error_msg = str(e)
        raise
    finally:
        end_time = time.time()
        end_memory = memory_usage()
        
        execution_time = end_time - start_time
        memory_delta = end_memory - start_memory
        
        result.update({
            "execution_time": execution_time,
            "memory_usage_mb": end_memory,
            "memory_delta_mb": memory_delta,
            "success": success,
            "error_message": error_msg,
        })
        
        logger.info(
            f"Operation '{name}' completed in {execution_time:.3f}s "
            f"(memory: {end_memory:.1f}MB, delta: {memory_delta:+.1f}MB)"
        )


def memory_usage() -> float:
    """Get current memory usage in MB.
    
    Returns:
        Current memory usage in megabytes
    """
    try:
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0.0


def profile_function(
    monitor: Optional[PerformanceMonitor] = None,
    log_calls: bool = True,
    log_threshold: float = 1.0
) -> Callable[[F], F]:
    """Decorator to profile function performance.
    
    Args:
        monitor: PerformanceMonitor instance (uses global if None)
        log_calls: Whether to log function calls
        log_threshold: Minimum execution time (seconds) to log
        
    Returns:
        Decorated function with performance monitoring
    """
    if monitor is None:
        monitor = _global_monitor
    
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = memory_usage()
            peak_memory = start_memory
            error_msg = None
            success = False
            
            try:
                # Monitor peak memory during execution
                result = func(*args, **kwargs)
                success = True
                return result
            
            except Exception as e:
                error_msg = str(e)
                raise
            
            finally:
                end_time = time.time()
                end_memory = memory_usage()
                execution_time = end_time - start_time
                peak_memory = max(peak_memory, end_memory)
                
                # Record metrics
                metrics = PerformanceMetrics(
                    function_name=func.__name__,
                    execution_time=execution_time,
                    memory_usage_mb=end_memory,
                    peak_memory_mb=peak_memory,
                    timestamp=datetime.now(),
                    args_count=len(args),
                    kwargs_count=len(kwargs),
                    success=success,
                    error_message=error_msg,
                )
                
                monitor.record_metrics(metrics)
                
                # Log if enabled and above threshold
                if log_calls and execution_time >= log_threshold:
                    status = "SUCCESS" if success else "ERROR"
                    logger.info(
                        f"[{status}] {func.__name__} executed in {execution_time:.3f}s "
                        f"(memory: {end_memory:.1f}MB, peak: {peak_memory:.1f}MB)"
                    )
                    
                    if error_msg:
                        logger.error(f"Error in {func.__name__}: {error_msg}")
        
        return cast(F, wrapper)
    
    return decorator


def benchmark_function(
    func: Callable, 
    args: tuple = (), 
    kwargs: Optional[Dict] = None,
    iterations: int = 10,
    warmup_iterations: int = 2
) -> Dict[str, Any]:
    """Benchmark a function's performance.
    
    Args:
        func: Function to benchmark
        args: Arguments to pass to function
        kwargs: Keyword arguments to pass to function
        iterations: Number of benchmark iterations
        warmup_iterations: Number of warmup iterations (not counted)
        
    Returns:
        Dictionary containing benchmark results
    """
    if kwargs is None:
        kwargs = {}
    
    # Warmup runs
    for _ in range(warmup_iterations):
        try:
            func(*args, **kwargs)
        except Exception:
            pass  # Ignore warmup errors
    
    # Benchmark runs
    execution_times = []
    memory_usages = []
    success_count = 0
    
    for i in range(iterations):
        start_memory = memory_usage()
        start_time = time.time()
        
        try:
            func(*args, **kwargs)
            success_count += 1
        except Exception as e:
            logger.warning(f"Benchmark iteration {i+1} failed: {e}")
        
        end_time = time.time()
        end_memory = memory_usage()
        
        execution_times.append(end_time - start_time)
        memory_usages.append(end_memory)
    
    if not execution_times:
        return {"error": "All benchmark iterations failed"}
    
    return {
        "function_name": func.__name__,
        "iterations": iterations,
        "successful_iterations": success_count,
        "success_rate": success_count / iterations,
        "execution_time": {
            "min": min(execution_times),
            "max": max(execution_times),
            "avg": sum(execution_times) / len(execution_times),
            "total": sum(execution_times),
        },
        "memory_usage": {
            "min": min(memory_usages),
            "max": max(memory_usages),
            "avg": sum(memory_usages) / len(memory_usages),
        },
    }


class ResourceMonitor:
    """Monitor system resource usage."""
    
    def __init__(self, interval: float = 1.0, enable_threading: bool = False):
        """Initialize resource monitor.
        
        Args:
            interval: Monitoring interval in seconds
            enable_threading: Whether to enable background threading (disabled by default)
        """
        self.interval = interval
        self.enable_threading = enable_threading
        self.monitoring = False
        self.thread: Optional[threading.Thread] = None
        self.data: List[Dict[str, Any]] = []
        self.lock = threading.Lock()
    
    def start_monitoring(self) -> None:
        """Start resource monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        
        if self.enable_threading:
            self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.thread.start()
            logger.info("Resource monitoring started (background thread)")
        else:
            # Just take a single snapshot without threading
            self._take_snapshot()
            logger.info("Resource monitoring started (single snapshot mode)")
    
    def stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        self.monitoring = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)  # Reduced timeout
            if self.thread.is_alive():
                logger.warning("Resource monitoring thread did not stop cleanly")
        logger.info("Resource monitoring stopped")
    
    def _take_snapshot(self) -> None:
        """Take a single resource usage snapshot."""
        try:
            timestamp = datetime.now()
            
            # CPU and memory info
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            
            # Process-specific info
            process = psutil.Process()
            process_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            data_point = {
                "timestamp": timestamp,
                "cpu_percent": cpu_percent,
                "memory_total_gb": memory.total / 1024 / 1024 / 1024,
                "memory_used_gb": memory.used / 1024 / 1024 / 1024,
                "memory_percent": memory.percent,
                "process_memory_mb": process_memory,
            }
            
            with self.lock:
                self.data.append(data_point)
                
                # Keep only last 1000 data points
                if len(self.data) > 1000:
                    self.data.pop(0)
                    
        except Exception as e:
            logger.warning(f"Error taking resource snapshot: {e}")
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop (only used when threading is enabled)."""
        while self.monitoring:
            try:
                self._take_snapshot()
            except Exception as e:
                logger.warning(f"Error in resource monitoring: {e}")
            
            # Check if we should stop before sleeping
            if not self.monitoring:
                break
                
            time.sleep(self.interval)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get resource usage statistics.
        
        Returns:
            Dictionary containing resource statistics
        """
        # Take a fresh snapshot if we have no data
        if not self.data:
            self._take_snapshot()
        
        with self.lock:
            if not self.data:
                return {"error": "No monitoring data available"}
            
            cpu_values = [d["cpu_percent"] for d in self.data]
            memory_values = [d["memory_percent"] for d in self.data]
            process_memory_values = [d["process_memory_mb"] for d in self.data]
            
            return {
                "monitoring_duration": len(self.data) * self.interval,
                "data_points": len(self.data),
                "cpu_usage": {
                    "min": min(cpu_values),
                    "max": max(cpu_values),
                    "avg": sum(cpu_values) / len(cpu_values),
                },
                "system_memory": {
                    "min": min(memory_values),
                    "max": max(memory_values),
                    "avg": sum(memory_values) / len(memory_values),
                },
                "process_memory": {
                    "min": min(process_memory_values),
                    "max": max(process_memory_values),
                    "avg": sum(process_memory_values) / len(process_memory_values),
                },
                "latest": self.data[-1] if self.data else None,
            }
    
    def clear_data(self) -> None:
        """Clear all monitoring data."""
        with self.lock:
            self.data.clear()


# Global resource monitor instance (threading disabled by default)
_global_resource_monitor = ResourceMonitor(enable_threading=False)


def get_resource_monitor() -> ResourceMonitor:
    """Get the global resource monitor instance."""
    return _global_resource_monitor 