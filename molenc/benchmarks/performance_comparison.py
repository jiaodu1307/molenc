"""Performance comparison tools for molecular encoders.

This module provides comprehensive benchmarking and comparison
tools to help users select the best encoder configuration
for their specific use cases and requirements.
"""

import time
import psutil
import logging
import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
from collections import defaultdict
import threading
import queue
from contextlib import contextmanager

import numpy as np


class BenchmarkType(Enum):
    """Types of benchmarks."""
    ENCODING_SPEED = "encoding_speed"
    MEMORY_USAGE = "memory_usage"
    ACCURACY = "accuracy"
    BATCH_PERFORMANCE = "batch_performance"
    INITIALIZATION_TIME = "initialization_time"
    RESOURCE_EFFICIENCY = "resource_efficiency"
    SCALABILITY = "scalability"
    STABILITY = "stability"


class MetricType(Enum):
    """Types of performance metrics."""
    TIME_SECONDS = "time_seconds"
    MEMORY_MB = "memory_mb"
    THROUGHPUT_PER_SECOND = "throughput_per_second"
    ACCURACY_SCORE = "accuracy_score"
    CPU_PERCENT = "cpu_percent"
    GPU_MEMORY_MB = "gpu_memory_mb"
    ERROR_RATE = "error_rate"
    LATENCY_MS = "latency_ms"


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""
    name: str
    benchmark_types: List[BenchmarkType]
    test_molecules: List[str]
    batch_sizes: List[int] = field(default_factory=lambda: [1, 10, 100, 1000])
    num_iterations: int = 5
    warmup_iterations: int = 2
    timeout_seconds: int = 300
    memory_sampling_interval: float = 0.1
    include_gpu_metrics: bool = True
    save_detailed_logs: bool = True
    comparison_baseline: Optional[str] = None


@dataclass
class PerformanceMetric:
    """A single performance metric measurement."""
    name: str
    value: float
    unit: str
    metric_type: MetricType
    timestamp: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    encoder_name: str
    encoder_config: Dict[str, Any]
    benchmark_type: BenchmarkType
    metrics: List[PerformanceMetric]
    success: bool
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    system_info: Optional[Dict[str, Any]] = None


@dataclass
class ComparisonReport:
    """Comprehensive comparison report."""
    config: BenchmarkConfig
    results: List[BenchmarkResult]
    summary: Dict[str, Any]
    recommendations: List[str]
    timestamp: float
    system_info: Dict[str, Any]


class ResourceMonitor:
    """Monitor system resources during benchmark execution."""
    
    def __init__(self, sampling_interval: float = 0.1):
        self.sampling_interval = sampling_interval
        self.logger = logging.getLogger(__name__)
        self._monitoring = False
        self._metrics: List[Dict[str, Any]] = []
        self._thread: Optional[threading.Thread] = None
    
    def start_monitoring(self):
        """Start resource monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._metrics.clear()
        self._thread = threading.Thread(target=self._monitor_loop)
        self._thread.daemon = True
        self._thread.start()
    
    def stop_monitoring(self) -> List[Dict[str, Any]]:
        """Stop monitoring and return collected metrics."""
        self._monitoring = False
        
        if self._thread:
            self._thread.join(timeout=1.0)
        
        return self._metrics.copy()
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        process = psutil.Process()
        
        while self._monitoring:
            try:
                # CPU and memory metrics
                cpu_percent = process.cpu_percent()
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                
                # System-wide metrics
                system_cpu = psutil.cpu_percent()
                system_memory = psutil.virtual_memory()
                
                metric = {
                    'timestamp': time.time(),
                    'process_cpu_percent': cpu_percent,
                    'process_memory_mb': memory_mb,
                    'system_cpu_percent': system_cpu,
                    'system_memory_percent': system_memory.percent,
                    'system_memory_available_mb': system_memory.available / 1024 / 1024
                }
                
                # GPU metrics (if available)
                try:
                    gpu_metrics = self._get_gpu_metrics()
                    metric.update(gpu_metrics)
                except Exception:
                    pass  # GPU monitoring not available
                
                self._metrics.append(metric)
                
            except Exception as e:
                self.logger.warning(f"Resource monitoring error: {e}")
            
            time.sleep(self.sampling_interval)
    
    def _get_gpu_metrics(self) -> Dict[str, Any]:
        """Get GPU metrics if available."""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            
            if gpus:
                gpu = gpus[0]  # Use first GPU
                return {
                    'gpu_memory_used_mb': gpu.memoryUsed,
                    'gpu_memory_total_mb': gpu.memoryTotal,
                    'gpu_memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                    'gpu_utilization_percent': gpu.load * 100,
                    'gpu_temperature': gpu.temperature
                }
        except ImportError:
            pass
        
        return {}


class PerformanceBenchmark:
    """Main performance benchmarking class."""
    
    def __init__(self, results_dir: Optional[Path] = None):
        self.results_dir = results_dir or Path.home() / ".molenc" / "benchmarks"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.resource_monitor = ResourceMonitor()
        
        # Test molecules for benchmarking
        self.default_test_molecules = [
            "CCO",  # Ethanol
            "CC(=O)O",  # Acetic acid
            "c1ccccc1",  # Benzene
            "CC",  # Ethane
            "O",  # Water
            "C",  # Methane
            "CC(C)O",  # Isopropanol
            "c1ccc(cc1)O",  # Phenol
            "CCN(CC)CC",  # Triethylamine
            "c1ccc2c(c1)cccn2",  # Quinoline
            "CC(C)(C)c1ccc(cc1)O",  # 4-tert-Butylphenol
            "c1ccc(cc1)c2ccccc2",  # Biphenyl
            "CCc1ccccc1",  # Ethylbenzene
            "COc1ccccc1",  # Anisole
            "c1ccc(cc1)N",  # Aniline
        ]
    
    def create_benchmark_config(
        self,
        name: str,
        benchmark_types: Optional[List[BenchmarkType]] = None,
        test_molecules: Optional[List[str]] = None,
        **kwargs
    ) -> BenchmarkConfig:
        """Create a benchmark configuration."""
        if benchmark_types is None:
            benchmark_types = [
                BenchmarkType.ENCODING_SPEED,
                BenchmarkType.MEMORY_USAGE,
                BenchmarkType.BATCH_PERFORMANCE
            ]
        
        if test_molecules is None:
            test_molecules = self.default_test_molecules
        
        return BenchmarkConfig(
            name=name,
            benchmark_types=benchmark_types,
            test_molecules=test_molecules,
            **kwargs
        )
    
    def benchmark_encoder(
        self,
        encoder,
        encoder_name: str,
        config: BenchmarkConfig
    ) -> List[BenchmarkResult]:
        """Benchmark a single encoder."""
        results = []
        
        # Get encoder configuration
        encoder_config = self._get_encoder_config(encoder)
        
        # Get system information
        system_info = self._get_system_info()
        
        for benchmark_type in config.benchmark_types:
            try:
                result = self._run_single_benchmark(
                    encoder, encoder_name, encoder_config,
                    benchmark_type, config, system_info
                )
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Benchmark {benchmark_type.value} failed for {encoder_name}: {e}")
                
                result = BenchmarkResult(
                    encoder_name=encoder_name,
                    encoder_config=encoder_config,
                    benchmark_type=benchmark_type,
                    metrics=[],
                    success=False,
                    error_message=str(e),
                    system_info=system_info
                )
                results.append(result)
        
        return results
    
    def _run_single_benchmark(
        self,
        encoder,
        encoder_name: str,
        encoder_config: Dict[str, Any],
        benchmark_type: BenchmarkType,
        config: BenchmarkConfig,
        system_info: Dict[str, Any]
    ) -> BenchmarkResult:
        """Run a single benchmark."""
        start_time = time.time()
        
        if benchmark_type == BenchmarkType.ENCODING_SPEED:
            metrics = self._benchmark_encoding_speed(encoder, config)
        elif benchmark_type == BenchmarkType.MEMORY_USAGE:
            metrics = self._benchmark_memory_usage(encoder, config)
        elif benchmark_type == BenchmarkType.BATCH_PERFORMANCE:
            metrics = self._benchmark_batch_performance(encoder, config)
        elif benchmark_type == BenchmarkType.INITIALIZATION_TIME:
            metrics = self._benchmark_initialization_time(encoder, config)
        elif benchmark_type == BenchmarkType.SCALABILITY:
            metrics = self._benchmark_scalability(encoder, config)
        elif benchmark_type == BenchmarkType.STABILITY:
            metrics = self._benchmark_stability(encoder, config)
        else:
            raise ValueError(f"Unsupported benchmark type: {benchmark_type}")
        
        execution_time = time.time() - start_time
        
        return BenchmarkResult(
            encoder_name=encoder_name,
            encoder_config=encoder_config,
            benchmark_type=benchmark_type,
            metrics=metrics,
            success=True,
            execution_time=execution_time,
            system_info=system_info
        )
    
    def _benchmark_encoding_speed(self, encoder, config: BenchmarkConfig) -> List[PerformanceMetric]:
        """Benchmark encoding speed."""
        metrics = []
        
        # Warmup
        for _ in range(config.warmup_iterations):
            try:
                encoder.encode(config.test_molecules[0])
            except Exception:
                pass
        
        # Single molecule encoding
        times = []
        for _ in range(config.num_iterations):
            start_time = time.time()
            
            for smiles in config.test_molecules:
                try:
                    encoder.encode(smiles)
                except Exception:
                    pass
            
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0
        throughput = len(config.test_molecules) / avg_time
        
        metrics.extend([
            PerformanceMetric(
                name="average_encoding_time",
                value=avg_time,
                unit="seconds",
                metric_type=MetricType.TIME_SECONDS,
                timestamp=time.time()
            ),
            PerformanceMetric(
                name="encoding_time_std",
                value=std_time,
                unit="seconds",
                metric_type=MetricType.TIME_SECONDS,
                timestamp=time.time()
            ),
            PerformanceMetric(
                name="encoding_throughput",
                value=throughput,
                unit="molecules/second",
                metric_type=MetricType.THROUGHPUT_PER_SECOND,
                timestamp=time.time()
            )
        ])
        
        return metrics
    
    def _benchmark_memory_usage(self, encoder, config: BenchmarkConfig) -> List[PerformanceMetric]:
        """Benchmark memory usage."""
        metrics = []
        
        # Start monitoring
        self.resource_monitor.start_monitoring()
        
        try:
            # Baseline memory
            time.sleep(0.5)  # Let monitoring stabilize
            
            # Encode molecules
            for smiles in config.test_molecules:
                try:
                    encoder.encode(smiles)
                except Exception:
                    pass
                time.sleep(0.1)  # Allow memory monitoring
            
            # Stop monitoring
            resource_data = self.resource_monitor.stop_monitoring()
            
            if resource_data:
                memory_values = [d['process_memory_mb'] for d in resource_data]
                
                metrics.extend([
                    PerformanceMetric(
                        name="peak_memory_usage",
                        value=max(memory_values),
                        unit="MB",
                        metric_type=MetricType.MEMORY_MB,
                        timestamp=time.time()
                    ),
                    PerformanceMetric(
                        name="average_memory_usage",
                        value=statistics.mean(memory_values),
                        unit="MB",
                        metric_type=MetricType.MEMORY_MB,
                        timestamp=time.time()
                    )
                ])
                
                # GPU memory if available
                gpu_memory_values = [d.get('gpu_memory_used_mb', 0) for d in resource_data]
                if any(gpu_memory_values):
                    metrics.append(
                        PerformanceMetric(
                            name="peak_gpu_memory_usage",
                            value=max(gpu_memory_values),
                            unit="MB",
                            metric_type=MetricType.GPU_MEMORY_MB,
                            timestamp=time.time()
                        )
                    )
        
        finally:
            self.resource_monitor.stop_monitoring()
        
        return metrics
    
    def _benchmark_batch_performance(self, encoder, config: BenchmarkConfig) -> List[PerformanceMetric]:
        """Benchmark batch encoding performance."""
        metrics = []
        
        for batch_size in config.batch_sizes:
            if batch_size > len(config.test_molecules):
                # Create larger batch by repeating molecules
                batch = (config.test_molecules * ((batch_size // len(config.test_molecules)) + 1))[:batch_size]
            else:
                batch = config.test_molecules[:batch_size]
            
            times = []
            for _ in range(config.num_iterations):
                start_time = time.time()
                
                try:
                    if hasattr(encoder, 'encode_batch'):
                        encoder.encode_batch(batch)
                    else:
                        # Fall back to individual encoding
                        for smiles in batch:
                            encoder.encode(smiles)
                except Exception:
                    pass
                
                end_time = time.time()
                times.append(end_time - start_time)
            
            if times:
                avg_time = statistics.mean(times)
                throughput = batch_size / avg_time
                
                metrics.extend([
                    PerformanceMetric(
                        name=f"batch_time_size_{batch_size}",
                        value=avg_time,
                        unit="seconds",
                        metric_type=MetricType.TIME_SECONDS,
                        timestamp=time.time(),
                        metadata={'batch_size': batch_size}
                    ),
                    PerformanceMetric(
                        name=f"batch_throughput_size_{batch_size}",
                        value=throughput,
                        unit="molecules/second",
                        metric_type=MetricType.THROUGHPUT_PER_SECOND,
                        timestamp=time.time(),
                        metadata={'batch_size': batch_size}
                    )
                ])
        
        return metrics
    
    def _benchmark_initialization_time(self, encoder, config: BenchmarkConfig) -> List[PerformanceMetric]:
        """Benchmark encoder initialization time."""
        # This is tricky since the encoder is already initialized
        # We can measure the time for the first encoding (cold start)
        
        start_time = time.time()
        try:
            encoder.encode(config.test_molecules[0])
        except Exception:
            pass
        end_time = time.time()
        
        return [
            PerformanceMetric(
                name="cold_start_time",
                value=end_time - start_time,
                unit="seconds",
                metric_type=MetricType.TIME_SECONDS,
                timestamp=time.time()
            )
        ]
    
    def _benchmark_scalability(self, encoder, config: BenchmarkConfig) -> List[PerformanceMetric]:
        """Benchmark scalability with increasing load."""
        metrics = []
        
        # Test with increasing number of molecules
        molecule_counts = [10, 50, 100, 500, 1000]
        
        for count in molecule_counts:
            if count > len(config.test_molecules):
                test_batch = (config.test_molecules * ((count // len(config.test_molecules)) + 1))[:count]
            else:
                test_batch = config.test_molecules[:count]
            
            start_time = time.time()
            
            try:
                for smiles in test_batch:
                    encoder.encode(smiles)
            except Exception:
                pass
            
            end_time = time.time()
            total_time = end_time - start_time
            throughput = count / total_time if total_time > 0 else 0
            
            metrics.append(
                PerformanceMetric(
                    name=f"scalability_throughput_{count}",
                    value=throughput,
                    unit="molecules/second",
                    metric_type=MetricType.THROUGHPUT_PER_SECOND,
                    timestamp=time.time(),
                    metadata={'molecule_count': count}
                )
            )
        
        return metrics
    
    def _benchmark_stability(self, encoder, config: BenchmarkConfig) -> List[PerformanceMetric]:
        """Benchmark stability over extended usage."""
        metrics = []
        
        error_count = 0
        total_runs = config.num_iterations * 10  # Extended testing
        
        for i in range(total_runs):
            try:
                smiles = config.test_molecules[i % len(config.test_molecules)]
                encoder.encode(smiles)
            except Exception:
                error_count += 1
        
        error_rate = error_count / total_runs
        
        metrics.append(
            PerformanceMetric(
                name="error_rate",
                value=error_rate,
                unit="ratio",
                metric_type=MetricType.ERROR_RATE,
                timestamp=time.time(),
                metadata={'total_runs': total_runs, 'errors': error_count}
            )
        )
        
        return metrics
    
    def compare_encoders(
        self,
        encoders: Dict[str, Any],
        config: BenchmarkConfig
    ) -> ComparisonReport:
        """Compare multiple encoders."""
        all_results = []
        
        for encoder_name, encoder in encoders.items():
            self.logger.info(f"Benchmarking encoder: {encoder_name}")
            
            try:
                results = self.benchmark_encoder(encoder, encoder_name, config)
                all_results.extend(results)
            except Exception as e:
                self.logger.error(f"Failed to benchmark {encoder_name}: {e}")
        
        # Generate summary and recommendations
        summary = self._generate_summary(all_results, config)
        recommendations = self._generate_recommendations(all_results, summary)
        
        report = ComparisonReport(
            config=config,
            results=all_results,
            summary=summary,
            recommendations=recommendations,
            timestamp=time.time(),
            system_info=self._get_system_info()
        )
        
        # Save report
        self._save_report(report)
        
        return report
    
    def _generate_summary(self, results: List[BenchmarkResult], config: BenchmarkConfig) -> Dict[str, Any]:
        """Generate summary statistics from benchmark results."""
        summary = {
            'total_encoders': len(set(r.encoder_name for r in results)),
            'total_benchmarks': len(results),
            'successful_benchmarks': len([r for r in results if r.success]),
            'by_encoder': {},
            'by_benchmark_type': {},
            'best_performers': {}
        }
        
        # Group by encoder
        by_encoder = defaultdict(list)
        for result in results:
            by_encoder[result.encoder_name].append(result)
        
        for encoder_name, encoder_results in by_encoder.items():
            encoder_summary = {
                'total_benchmarks': len(encoder_results),
                'successful_benchmarks': len([r for r in encoder_results if r.success]),
                'metrics': {}
            }
            
            # Aggregate metrics
            for result in encoder_results:
                if result.success:
                    for metric in result.metrics:
                        if metric.name not in encoder_summary['metrics']:
                            encoder_summary['metrics'][metric.name] = []
                        encoder_summary['metrics'][metric.name].append(metric.value)
            
            # Calculate averages
            for metric_name, values in encoder_summary['metrics'].items():
                if values:
                    encoder_summary['metrics'][metric_name] = {
                        'average': statistics.mean(values),
                        'min': min(values),
                        'max': max(values),
                        'std': statistics.stdev(values) if len(values) > 1 else 0
                    }
            
            summary['by_encoder'][encoder_name] = encoder_summary
        
        # Find best performers
        self._find_best_performers(summary, results)
        
        return summary
    
    def _find_best_performers(self, summary: Dict[str, Any], results: List[BenchmarkResult]):
        """Find best performing encoders for different metrics."""
        metric_values = defaultdict(dict)
        
        # Collect all metric values by encoder
        for result in results:
            if result.success:
                for metric in result.metrics:
                    if result.encoder_name not in metric_values[metric.name]:
                        metric_values[metric.name][result.encoder_name] = []
                    metric_values[metric.name][result.encoder_name].append(metric.value)
        
        # Find best performers for key metrics
        best_performers = {}
        
        for metric_name, encoder_values in metric_values.items():
            if not encoder_values:
                continue
            
            # Calculate average for each encoder
            avg_values = {}
            for encoder_name, values in encoder_values.items():
                avg_values[encoder_name] = statistics.mean(values)
            
            # Determine if higher or lower is better
            if any(keyword in metric_name.lower() for keyword in ['time', 'latency', 'error']):
                # Lower is better
                best_encoder = min(avg_values.items(), key=lambda x: x[1])
            else:
                # Higher is better (throughput, accuracy, etc.)
                best_encoder = max(avg_values.items(), key=lambda x: x[1])
            
            best_performers[metric_name] = {
                'encoder': best_encoder[0],
                'value': best_encoder[1]
            }
        
        summary['best_performers'] = best_performers
    
    def _generate_recommendations(self, results: List[BenchmarkResult], summary: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on benchmark results."""
        recommendations = []
        
        # Overall best performer
        if summary['best_performers']:
            # Count how many metrics each encoder wins
            encoder_wins = defaultdict(int)
            for metric_info in summary['best_performers'].values():
                encoder_wins[metric_info['encoder']] += 1
            
            if encoder_wins:
                best_overall = max(encoder_wins.items(), key=lambda x: x[1])
                recommendations.append(
                    f"Overall best performer: {best_overall[0]} (wins {best_overall[1]} metrics)"
                )
        
        # Speed recommendations
        if 'encoding_throughput' in summary['best_performers']:
            fastest = summary['best_performers']['encoding_throughput']
            recommendations.append(
                f"Fastest encoding: {fastest['encoder']} ({fastest['value']:.2f} molecules/second)"
            )
        
        # Memory recommendations
        if 'peak_memory_usage' in summary['best_performers']:
            most_efficient = summary['best_performers']['peak_memory_usage']
            recommendations.append(
                f"Most memory efficient: {most_efficient['encoder']} ({most_efficient['value']:.2f} MB peak)"
            )
        
        # Stability recommendations
        if 'error_rate' in summary['best_performers']:
            most_stable = summary['best_performers']['error_rate']
            recommendations.append(
                f"Most stable: {most_stable['encoder']} ({most_stable['value']:.4f} error rate)"
            )
        
        # Use case specific recommendations
        recommendations.extend([
            "For real-time applications: Choose the encoder with lowest latency",
            "For batch processing: Choose the encoder with highest throughput",
            "For resource-constrained environments: Choose the most memory efficient encoder",
            "For production systems: Choose the most stable encoder with acceptable performance"
        ])
        
        return recommendations
    
    def _get_encoder_config(self, encoder) -> Dict[str, Any]:
        """Extract encoder configuration."""
        config = {
            'type': type(encoder).__name__,
            'module': type(encoder).__module__
        }
        
        # Try to get additional config if available
        if hasattr(encoder, 'get_config'):
            try:
                config.update(encoder.get_config())
            except Exception:
                pass
        
        return config
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        import platform
        
        info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'timestamp': time.time()
        }
        
        # GPU information if available
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                info['gpu_info'] = [
                    {
                        'name': gpu.name,
                        'memory_total_mb': gpu.memoryTotal,
                        'driver_version': gpu.driver
                    }
                    for gpu in gpus
                ]
        except ImportError:
            pass
        
        return info
    
    def _save_report(self, report: ComparisonReport):
        """Save comparison report to disk."""
        timestamp_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(report.timestamp))
        filename = f"benchmark_report_{report.config.name}_{timestamp_str}.json"
        filepath = self.results_dir / filename
        
        try:
            # Convert report to JSON-serializable format
            report_dict = {
                'config': asdict(report.config),
                'results': [asdict(result) for result in report.results],
                'summary': report.summary,
                'recommendations': report.recommendations,
                'timestamp': report.timestamp,
                'system_info': report.system_info
            }
            
            # Convert enums to strings
            for result in report_dict['results']:
                result['benchmark_type'] = result['benchmark_type'].value
                for metric in result['metrics']:
                    metric['metric_type'] = metric['metric_type'].value
            
            for benchmark_type in report_dict['config']['benchmark_types']:
                report_dict['config']['benchmark_types'] = [
                    bt.value if hasattr(bt, 'value') else bt 
                    for bt in report_dict['config']['benchmark_types']
                ]
            
            with open(filepath, 'w') as f:
                json.dump(report_dict, f, indent=2, default=str)
            
            self.logger.info(f"Benchmark report saved: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save benchmark report: {e}")


# Global benchmark instance
_benchmark = None


def get_benchmark(results_dir: Optional[Path] = None) -> PerformanceBenchmark:
    """Get the global benchmark instance."""
    global _benchmark
    if _benchmark is None:
        _benchmark = PerformanceBenchmark(results_dir)
    return _benchmark


def quick_compare_encoders(
    encoders: Dict[str, Any],
    test_molecules: Optional[List[str]] = None
) -> ComparisonReport:
    """Quick comparison of encoders with default settings."""
    benchmark = get_benchmark()
    
    config = benchmark.create_benchmark_config(
        name="quick_comparison",
        test_molecules=test_molecules,
        num_iterations=3,
        batch_sizes=[1, 10, 100]
    )
    
    return benchmark.compare_encoders(encoders, config)


def benchmark_single_encoder(
    encoder,
    encoder_name: str,
    benchmark_types: Optional[List[BenchmarkType]] = None
) -> List[BenchmarkResult]:
    """Benchmark a single encoder with default settings."""
    benchmark = get_benchmark()
    
    config = benchmark.create_benchmark_config(
        name=f"single_encoder_{encoder_name}",
        benchmark_types=benchmark_types
    )
    
    return benchmark.benchmark_encoder(encoder, encoder_name, config)