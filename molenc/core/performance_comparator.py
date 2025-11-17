"""Performance comparison tools for MolEnc encoders.

This module provides:
- Encoder performance benchmarking
- Speed and accuracy comparisons
- Resource usage monitoring
- Recommendation system
"""

import time
import psutil
import logging
import statistics
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from ..core.exceptions import MolEncError


@dataclass
class PerformanceMetrics:
    """Performance metrics for an encoder."""
    encoder_name: str
    encoding_time: float  # seconds
    memory_usage: float  # MB
    cpu_usage: float  # percentage
    accuracy_score: Optional[float] = None
    throughput: float = 0.0  # molecules per second
    error_rate: float = 0.0  # percentage of failed encodings
    
    # Additional metrics
    initialization_time: float = 0.0
    batch_efficiency: float = 1.0  # batch vs individual speedup
    resource_efficiency: float = 1.0  # output quality per resource unit
    
    # Detailed timing
    min_time: float = 0.0
    max_time: float = 0.0
    median_time: float = 0.0
    std_time: float = 0.0
    
    # Error details
    errors: List[str] = field(default_factory=list)
    

@dataclass
class BenchmarkConfig:
    """Configuration for performance benchmarks."""
    test_molecules: List[str]
    num_iterations: int = 3
    warmup_iterations: int = 1
    batch_sizes: List[int] = field(default_factory=lambda: [1, 10, 50, 100])
    timeout: float = 300.0  # seconds
    measure_accuracy: bool = False
    reference_encodings: Optional[Dict[str, np.ndarray]] = None
    

class PerformanceComparator:
    """Tool for comparing encoder performance."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._results: Dict[str, PerformanceMetrics] = {}
        
    def benchmark_encoder(self, 
                         encoder: Any,
                         config: BenchmarkConfig,
                         encoder_name: Optional[str] = None) -> PerformanceMetrics:
        """Benchmark a single encoder.
        
        Args:
            encoder: Encoder instance to benchmark
            config: Benchmark configuration
            encoder_name: Name for the encoder
            
        Returns:
            Performance metrics
        """
        if encoder_name is None:
            encoder_name = encoder.__class__.__name__
            
        self.logger.info(f"Benchmarking encoder: {encoder_name}")
        
        # Initialize metrics
        metrics = PerformanceMetrics(
            encoder_name=encoder_name,
            encoding_time=0.0,
            memory_usage=0.0,
            cpu_usage=0.0
        )
        
        try:
            # Measure initialization time
            init_start = time.time()
            if hasattr(encoder, 'initialize') and callable(encoder.initialize):
                encoder.initialize()
            metrics.initialization_time = time.time() - init_start
            
            # Warmup
            if config.warmup_iterations > 0:
                self._run_warmup(encoder, config)
                
            # Benchmark individual encoding
            individual_metrics = self._benchmark_individual_encoding(
                encoder, config
            )
            
            # Benchmark batch encoding
            batch_metrics = self._benchmark_batch_encoding(
                encoder, config
            )
            
            # Combine metrics
            metrics.encoding_time = individual_metrics['avg_time']
            metrics.min_time = individual_metrics['min_time']
            metrics.max_time = individual_metrics['max_time']
            metrics.median_time = individual_metrics['median_time']
            metrics.std_time = individual_metrics['std_time']
            metrics.memory_usage = individual_metrics['memory_usage']
            metrics.cpu_usage = individual_metrics['cpu_usage']
            metrics.error_rate = individual_metrics['error_rate']
            metrics.errors = individual_metrics['errors']
            
            # Calculate throughput
            if metrics.encoding_time > 0:
                metrics.throughput = 1.0 / metrics.encoding_time
                
            # Calculate batch efficiency
            if batch_metrics and batch_metrics['avg_time'] > 0:
                expected_batch_time = metrics.encoding_time * len(config.test_molecules)
                actual_batch_time = batch_metrics['avg_time']
                metrics.batch_efficiency = expected_batch_time / actual_batch_time
                
            # Calculate resource efficiency
            if metrics.memory_usage > 0 and metrics.cpu_usage > 0:
                resource_cost = (metrics.memory_usage / 1000) + (metrics.cpu_usage / 100)
                metrics.resource_efficiency = metrics.throughput / resource_cost
                
            # Measure accuracy if requested
            if config.measure_accuracy and config.reference_encodings:
                metrics.accuracy_score = self._measure_accuracy(
                    encoder, config
                )
                
        except Exception as e:
            self.logger.error(f"Benchmark failed for {encoder_name}: {e}")
            metrics.errors.append(str(e))
            metrics.error_rate = 100.0
            
        self._results[encoder_name] = metrics
        return metrics
        
    def _run_warmup(self, encoder: Any, config: BenchmarkConfig) -> None:
        """Run warmup iterations."""
        warmup_molecules = config.test_molecules[:min(5, len(config.test_molecules))]
        
        for _ in range(config.warmup_iterations):
            for smiles in warmup_molecules:
                try:
                    if hasattr(encoder, 'encode'):
                        encoder.encode(smiles)
                    elif hasattr(encoder, '__call__'):
                        encoder(smiles)
                except Exception:
                    pass  # Ignore warmup errors
                    
    def _benchmark_individual_encoding(self, 
                                     encoder: Any,
                                     config: BenchmarkConfig) -> Dict[str, Any]:
        """Benchmark individual molecule encoding."""
        times = []
        memory_usage = []
        cpu_usage = []
        errors = []
        
        # Get initial resource usage
        process = psutil.Process()
        
        for iteration in range(config.num_iterations):
            iteration_times = []
            
            for smiles in config.test_molecules:
                try:
                    # Measure resources before
                    mem_before = process.memory_info().rss / 1024 / 1024  # MB
                    cpu_before = process.cpu_percent()
                    
                    # Time the encoding
                    start_time = time.time()
                    
                    if hasattr(encoder, 'encode'):
                        result = encoder.encode(smiles)
                    elif hasattr(encoder, '__call__'):
                        result = encoder(smiles)
                    else:
                        raise AttributeError("Encoder has no encode method or __call__")
                        
                    end_time = time.time()
                    
                    # Measure resources after
                    mem_after = process.memory_info().rss / 1024 / 1024  # MB
                    cpu_after = process.cpu_percent()
                    
                    # Record metrics
                    encoding_time = end_time - start_time
                    iteration_times.append(encoding_time)
                    memory_usage.append(mem_after - mem_before)
                    cpu_usage.append(max(0, cpu_after - cpu_before))
                    
                except Exception as e:
                    errors.append(f"SMILES '{smiles}': {str(e)}")
                    
            if iteration_times:
                times.extend(iteration_times)
                
        # Calculate statistics
        if times:
            avg_time = statistics.mean(times)
            min_time = min(times)
            max_time = max(times)
            median_time = statistics.median(times)
            std_time = statistics.stdev(times) if len(times) > 1 else 0.0
        else:
            avg_time = min_time = max_time = median_time = std_time = 0.0
            
        avg_memory = statistics.mean(memory_usage) if memory_usage else 0.0
        avg_cpu = statistics.mean(cpu_usage) if cpu_usage else 0.0
        
        total_attempts = config.num_iterations * len(config.test_molecules)
        error_rate = (len(errors) / total_attempts * 100) if total_attempts > 0 else 0.0
        
        return {
            'avg_time': avg_time,
            'min_time': min_time,
            'max_time': max_time,
            'median_time': median_time,
            'std_time': std_time,
            'memory_usage': avg_memory,
            'cpu_usage': avg_cpu,
            'error_rate': error_rate,
            'errors': errors
        }
        
    def _benchmark_batch_encoding(self, 
                                encoder: Any,
                                config: BenchmarkConfig) -> Optional[Dict[str, Any]]:
        """Benchmark batch encoding if supported."""
        if not hasattr(encoder, 'encode_batch'):
            return None
            
        batch_times = []
        
        for batch_size in config.batch_sizes:
            if batch_size > len(config.test_molecules):
                continue
                
            batch = config.test_molecules[:batch_size]
            
            for _ in range(config.num_iterations):
                try:
                    start_time = time.time()
                    encoder.encode_batch(batch)
                    end_time = time.time()
                    
                    batch_times.append(end_time - start_time)
                    
                except Exception as e:
                    self.logger.warning(f"Batch encoding failed: {e}")
                    
        if batch_times:
            return {
                'avg_time': statistics.mean(batch_times),
                'min_time': min(batch_times),
                'max_time': max(batch_times)
            }
            
        return None
        
    def _measure_accuracy(self, 
                         encoder: Any,
                         config: BenchmarkConfig) -> float:
        """Measure encoding accuracy against reference."""
        if not config.reference_encodings:
            return 0.0
            
        similarities = []
        
        for smiles in config.test_molecules:
            if smiles not in config.reference_encodings:
                continue
                
            try:
                if hasattr(encoder, 'encode'):
                    encoding = encoder.encode(smiles)
                elif hasattr(encoder, '__call__'):
                    encoding = encoder(smiles)
                else:
                    continue
                    
                reference = config.reference_encodings[smiles]
                
                # Calculate cosine similarity
                if hasattr(encoding, 'numpy'):
                    encoding = encoding.numpy()
                if hasattr(reference, 'numpy'):
                    reference = reference.numpy()
                    
                encoding = np.array(encoding).flatten()
                reference = np.array(reference).flatten()
                
                if len(encoding) == len(reference):
                    similarity = np.dot(encoding, reference) / (
                        np.linalg.norm(encoding) * np.linalg.norm(reference)
                    )
                    similarities.append(similarity)
                    
            except Exception as e:
                self.logger.warning(f"Accuracy measurement failed for {smiles}: {e}")
                
        return statistics.mean(similarities) if similarities else 0.0
        
    def compare_encoders(self, 
                        encoders: Dict[str, Any],
                        config: BenchmarkConfig) -> Dict[str, PerformanceMetrics]:
        """Compare multiple encoders.
        
        Args:
            encoders: Dictionary of encoder name -> encoder instance
            config: Benchmark configuration
            
        Returns:
            Dictionary of encoder name -> performance metrics
        """
        results = {}
        
        for name, encoder in encoders.items():
            self.logger.info(f"Benchmarking {name}...")
            try:
                metrics = self.benchmark_encoder(encoder, config, name)
                results[name] = metrics
            except Exception as e:
                self.logger.error(f"Failed to benchmark {name}: {e}")
                
        return results

    def quick_compare(self, encoder_a: Any, encoder_b: Any, smiles_list: List[str]) -> Dict[str, Any]:
        """Quickly compare outputs of two encoders on the same inputs."""
        import numpy as np
        import time
        t0 = time.time()
        va = encoder_a.encode_batch(smiles_list)
        t1 = time.time()
        vb = encoder_b.encode_batch(smiles_list)
        t2 = time.time()
        shape_a = list(va.shape)
        shape_b = list(vb.shape)
        diff = None
        if shape_a == shape_b and va.size > 0:
            diff = np.linalg.norm(va - vb, axis=1).tolist()
        return {
            'shape_a': shape_a,
            'shape_b': shape_b,
            'encode_time_a': t1 - t0,
            'encode_time_b': t2 - t1,
            'l2_diff': diff
        }
        
    def get_recommendations(self, 
                          metrics: Dict[str, PerformanceMetrics],
                          priorities: Optional[Dict[str, float]] = None) -> List[Tuple[str, float]]:
        """Get encoder recommendations based on performance.
        
        Args:
            metrics: Performance metrics for encoders
            priorities: Weights for different metrics (speed, memory, accuracy, etc.)
            
        Returns:
            List of (encoder_name, score) tuples, sorted by score (descending)
        """
        if not metrics:
            return []
            
        # Default priorities
        if priorities is None:
            priorities = {
                'speed': 0.3,
                'memory': 0.2,
                'accuracy': 0.3,
                'reliability': 0.2
            }
            
        scores = {}
        
        # Normalize metrics
        all_speeds = [m.throughput for m in metrics.values() if m.throughput > 0]
        all_memory = [m.memory_usage for m in metrics.values() if m.memory_usage > 0]
        all_accuracy = [m.accuracy_score for m in metrics.values() if m.accuracy_score is not None]
        all_reliability = [100 - m.error_rate for m in metrics.values()]
        
        max_speed = max(all_speeds) if all_speeds else 1.0
        min_memory = min(all_memory) if all_memory else 1.0
        max_accuracy = max(all_accuracy) if all_accuracy else 1.0
        max_reliability = max(all_reliability) if all_reliability else 1.0
        
        for name, metric in metrics.items():
            score = 0.0
            
            # Speed score (higher is better)
            if metric.throughput > 0 and max_speed > 0:
                speed_score = metric.throughput / max_speed
                score += priorities.get('speed', 0) * speed_score
                
            # Memory score (lower is better)
            if metric.memory_usage > 0 and min_memory > 0:
                memory_score = min_memory / metric.memory_usage
                score += priorities.get('memory', 0) * memory_score
                
            # Accuracy score (higher is better)
            if metric.accuracy_score is not None and max_accuracy > 0:
                accuracy_score = metric.accuracy_score / max_accuracy
                score += priorities.get('accuracy', 0) * accuracy_score
                
            # Reliability score (higher is better)
            reliability = 100 - metric.error_rate
            if max_reliability > 0:
                reliability_score = reliability / max_reliability
                score += priorities.get('reliability', 0) * reliability_score
                
            scores[name] = score
            
        # Sort by score (descending)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
    def generate_report(self, 
                       metrics: Dict[str, PerformanceMetrics],
                       output_format: str = 'text') -> str:
        """Generate a performance comparison report.
        
        Args:
            metrics: Performance metrics for encoders
            output_format: Output format ('text', 'markdown', 'json')
            
        Returns:
            Formatted report string
        """
        if output_format == 'json':
            import json
            report_data = {}
            for name, metric in metrics.items():
                report_data[name] = {
                    'encoding_time': metric.encoding_time,
                    'memory_usage': metric.memory_usage,
                    'cpu_usage': metric.cpu_usage,
                    'throughput': metric.throughput,
                    'error_rate': metric.error_rate,
                    'accuracy_score': metric.accuracy_score,
                    'batch_efficiency': metric.batch_efficiency,
                    'resource_efficiency': metric.resource_efficiency
                }
            return json.dumps(report_data, indent=2)
            
        elif output_format == 'markdown':
            return self._generate_markdown_report(metrics)
        else:
            return self._generate_text_report(metrics)
        
    def export_csv(self, metrics: Dict[str, PerformanceMetrics]) -> str:
        """Export metrics to CSV string."""
        import io, csv
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(['encoder','encoding_time','throughput','memory_usage','cpu_usage','error_rate','accuracy_score','batch_efficiency','resource_efficiency'])
        for name, m in metrics.items():
            writer.writerow([
                name,
                m.encoding_time,
                m.throughput,
                m.memory_usage,
                m.cpu_usage,
                m.error_rate,
                m.accuracy_score if m.accuracy_score is not None else '',
                m.batch_efficiency,
                m.resource_efficiency,
            ])
        return buf.getvalue()
            
    def _generate_text_report(self, metrics: Dict[str, PerformanceMetrics]) -> str:
        """Generate a text report."""
        lines = []
        lines.append("=" * 60)
        lines.append("MolEnc Encoder Performance Comparison")
        lines.append("=" * 60)
        lines.append("")
        
        for name, metric in metrics.items():
            lines.append(f"Encoder: {name}")
            lines.append("-" * 40)
            lines.append(f"Encoding Time: {metric.encoding_time:.4f}s")
            lines.append(f"Throughput: {metric.throughput:.2f} molecules/s")
            lines.append(f"Memory Usage: {metric.memory_usage:.2f} MB")
            lines.append(f"CPU Usage: {metric.cpu_usage:.2f}%")
            lines.append(f"Error Rate: {metric.error_rate:.2f}%")
            
            if metric.accuracy_score is not None:
                lines.append(f"Accuracy Score: {metric.accuracy_score:.4f}")
                
            lines.append(f"Batch Efficiency: {metric.batch_efficiency:.2f}x")
            lines.append(f"Resource Efficiency: {metric.resource_efficiency:.4f}")
            lines.append("")
            
        # Add recommendations
        recommendations = self.get_recommendations(metrics)
        if recommendations:
            lines.append("Recommendations (by overall score):")
            lines.append("-" * 40)
            for i, (name, score) in enumerate(recommendations, 1):
                lines.append(f"{i}. {name} (score: {score:.4f})")
                
        return "\n".join(lines)
        
    def _generate_markdown_report(self, metrics: Dict[str, PerformanceMetrics]) -> str:
        """Generate a markdown report."""
        lines = []
        lines.append("# MolEnc Encoder Performance Comparison")
        lines.append("")
        
        # Summary table
        lines.append("## Performance Summary")
        lines.append("")
        lines.append("| Encoder | Time (s) | Throughput | Memory (MB) | CPU (%) | Error Rate (%) | Accuracy |")
        lines.append("|---------|----------|------------|-------------|---------|----------------|----------|")
        
        for name, metric in metrics.items():
            accuracy = f"{metric.accuracy_score:.4f}" if metric.accuracy_score is not None else "N/A"
            lines.append(
                f"| {name} | {metric.encoding_time:.4f} | {metric.throughput:.2f} | "
                f"{metric.memory_usage:.2f} | {metric.cpu_usage:.2f} | {metric.error_rate:.2f} | {accuracy} |"
            )
            
        lines.append("")
        
        # Detailed metrics
        lines.append("## Detailed Metrics")
        lines.append("")
        
        for name, metric in metrics.items():
            lines.append(f"### {name}")
            lines.append("")
            lines.append(f"- **Encoding Time**: {metric.encoding_time:.4f}s")
            lines.append(f"- **Throughput**: {metric.throughput:.2f} molecules/s")
            lines.append(f"- **Memory Usage**: {metric.memory_usage:.2f} MB")
            lines.append(f"- **CPU Usage**: {metric.cpu_usage:.2f}%")
            lines.append(f"- **Error Rate**: {metric.error_rate:.2f}%")
            lines.append(f"- **Batch Efficiency**: {metric.batch_efficiency:.2f}x")
            lines.append(f"- **Resource Efficiency**: {metric.resource_efficiency:.4f}")
            
            if metric.accuracy_score is not None:
                lines.append(f"- **Accuracy Score**: {metric.accuracy_score:.4f}")
                
            lines.append("")
            
        # Recommendations
        recommendations = self.get_recommendations(metrics)
        if recommendations:
            lines.append("## Recommendations")
            lines.append("")
            lines.append("Based on overall performance score:")
            lines.append("")
            
            for i, (name, score) in enumerate(recommendations, 1):
                lines.append(f"{i}. **{name}** (score: {score:.4f})")
                
        return "\n".join(lines)


# Global performance comparator instance
performance_comparator = PerformanceComparator()

# Default test molecules for benchmarking
DEFAULT_TEST_MOLECULES = [
    'CCO',  # Ethanol
    'CC(C)O',  # Isopropanol
    'c1ccccc1',  # Benzene
    'CCc1ccccc1',  # Ethylbenzene
    'CC(=O)O',  # Acetic acid
    'CC(C)(C)O',  # tert-Butanol
    'c1ccc2ccccc2c1',  # Naphthalene
    'CC(C)CC(C)(C)O',  # More complex molecule
    'Cc1ccc(cc1)C(C)(C)C',  # para-tert-Butyl toluene
    'CC1=CC=C(C=C1)C(C)(C)C'  # Another representation
]