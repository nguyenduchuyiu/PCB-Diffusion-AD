import time
import psutil
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from collections import defaultdict, deque
from contextlib import contextmanager

class PerformanceProfiler:
    """
    Performance profiling tool for monitoring training performance
    """
    
    def __init__(self, log_dir="outputs/profiling", window_size=100):
        self.log_dir = log_dir
        self.window_size = window_size
        os.makedirs(log_dir, exist_ok=True)
        
        # Performance metrics
        self.metrics = defaultdict(deque)
        self.timers = {}
        self.counters = defaultdict(int)
        
        # System monitoring
        self.cpu_usage = deque(maxlen=window_size)
        self.memory_usage = deque(maxlen=window_size)
        self.gpu_memory = deque(maxlen=window_size)
        self.gpu_utilization = deque(maxlen=window_size)
        
        # Training specific
        self.batch_times = deque(maxlen=window_size)
        self.epoch_times = deque(maxlen=window_size)
        self.loss_values = defaultdict(lambda: deque(maxlen=window_size))
        
    @contextmanager
    def timer(self, name):
        """Context manager for timing operations"""
        start_time = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            self.metrics[name].append(elapsed)
            if len(self.metrics[name]) > self.window_size:
                self.metrics[name].popleft()
    
    def log_system_metrics(self):
        """Log current system metrics"""
        # CPU
        cpu_percent = psutil.cpu_percent(interval=0.1)
        self.cpu_usage.append(cpu_percent)
        
        # Memory
        memory = psutil.virtual_memory()
        self.memory_usage.append(memory.percent)
        
        # GPU metrics (if available)
        if torch.cuda.is_available():
            gpu_memory_allocated = torch.cuda.memory_allocated() / (1024**3)
            gpu_memory_reserved = torch.cuda.memory_reserved() / (1024**3)
            self.gpu_memory.append({
                'allocated': gpu_memory_allocated,
                'reserved': gpu_memory_reserved
            })
            
            # Try to get GPU utilization (requires nvidia-ml-py3)
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                self.gpu_utilization.append(util.gpu)
            except:
                self.gpu_utilization.append(0)
    
    def log_training_metrics(self, epoch, batch_idx, losses, batch_time=None, epoch_time=None):
        """Log training specific metrics"""
        if batch_time:
            self.batch_times.append(batch_time)
        if epoch_time:
            self.epoch_times.append(epoch_time)
            
        # Log losses
        for loss_name, loss_value in losses.items():
            self.loss_values[loss_name].append(loss_value)
    
    def get_summary_stats(self):
        """Get summary statistics"""
        stats = {}
        
        # System metrics
        if self.cpu_usage:
            stats['cpu'] = {
                'avg': np.mean(self.cpu_usage),
                'max': np.max(self.cpu_usage),
                'current': self.cpu_usage[-1] if self.cpu_usage else 0
            }
        
        if self.memory_usage:
            stats['memory'] = {
                'avg': np.mean(self.memory_usage),
                'max': np.max(self.memory_usage),
                'current': self.memory_usage[-1] if self.memory_usage else 0
            }
        
        if self.gpu_memory:
            allocated = [m['allocated'] for m in self.gpu_memory]
            reserved = [m['reserved'] for m in self.gpu_memory]
            stats['gpu_memory'] = {
                'allocated_avg': np.mean(allocated),
                'allocated_max': np.max(allocated),
                'reserved_avg': np.mean(reserved),
                'reserved_max': np.max(reserved)
            }
        
        # Training metrics
        if self.batch_times:
            stats['batch_time'] = {
                'avg': np.mean(self.batch_times),
                'std': np.std(self.batch_times),
                'throughput': 1.0 / np.mean(self.batch_times)  # batches per second
            }
        
        if self.epoch_times:
            stats['epoch_time'] = {
                'avg': np.mean(self.epoch_times),
                'std': np.std(self.epoch_times),
                'total': np.sum(self.epoch_times)
            }
        
        # Custom timers
        for timer_name, times in self.metrics.items():
            if times:
                stats[timer_name] = {
                    'avg': np.mean(times),
                    'std': np.std(times),
                    'total': np.sum(times)
                }
        
        return stats
    
    def plot_metrics(self, save_path=None, inline=False):
        """Plot performance metrics"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # CPU usage
        if self.cpu_usage:
            axes[0, 0].plot(list(self.cpu_usage))
            axes[0, 0].set_title('CPU Usage (%)')
            axes[0, 0].set_ylabel('Usage %')
            axes[0, 0].grid(True)
        
        # Memory usage
        if self.memory_usage:
            axes[0, 1].plot(list(self.memory_usage))
            axes[0, 1].set_title('Memory Usage (%)')
            axes[0, 1].set_ylabel('Usage %')
            axes[0, 1].grid(True)
        
        # GPU memory
        if self.gpu_memory:
            allocated = [m['allocated'] for m in self.gpu_memory]
            reserved = [m['reserved'] for m in self.gpu_memory]
            axes[0, 2].plot(allocated, label='Allocated')
            axes[0, 2].plot(reserved, label='Reserved')
            axes[0, 2].set_title('GPU Memory (GB)')
            axes[0, 2].set_ylabel('Memory GB')
            axes[0, 2].legend()
            axes[0, 2].grid(True)
        
        # Batch times
        if self.batch_times:
            axes[1, 0].plot(list(self.batch_times))
            axes[1, 0].set_title('Batch Processing Time')
            axes[1, 0].set_ylabel('Time (s)')
            axes[1, 0].grid(True)
        
        # Loss values
        if self.loss_values:
            for loss_name, values in self.loss_values.items():
                axes[1, 1].plot(list(values), label=loss_name)
            axes[1, 1].set_title('Training Losses')
            axes[1, 1].set_ylabel('Loss Value')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        # GPU utilization
        if self.gpu_utilization:
            axes[1, 2].plot(list(self.gpu_utilization))
            axes[1, 2].set_title('GPU Utilization (%)')
            axes[1, 2].set_ylabel('Utilization %')
            axes[1, 2].grid(True)
        
        plt.tight_layout()
        
        if inline:
            # Display inline for Jupyter notebooks
            try:
                from IPython.display import display
                display(fig)
                plt.close()
            except ImportError:
                # Fallback to regular display if IPython not available
                plt.show()
        else:
            # Save to file
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            else:
                plt.savefig(os.path.join(self.log_dir, 'performance_metrics.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    def save_stats(self, filename=None):
        """Save statistics to JSON file"""
        stats = self.get_summary_stats()
        
        if filename is None:
            filename = os.path.join(self.log_dir, 'performance_stats.json')
        
        with open(filename, 'w') as f:
            json.dump(stats, f, indent=2)
        
        return stats
    
    def print_summary(self):
        """Print performance summary"""
        stats = self.get_summary_stats()
        
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)
        
        if 'cpu' in stats:
            print(f"CPU Usage: {stats['cpu']['avg']:.1f}% avg, {stats['cpu']['max']:.1f}% max")
        
        if 'memory' in stats:
            print(f"Memory Usage: {stats['memory']['avg']:.1f}% avg, {stats['memory']['max']:.1f}% max")
        
        if 'gpu_memory' in stats:
            print(f"GPU Memory: {stats['gpu_memory']['allocated_avg']:.2f}GB avg, {stats['gpu_memory']['allocated_max']:.2f}GB max")
        
        if 'batch_time' in stats:
            print(f"Batch Time: {stats['batch_time']['avg']:.3f}s avg, {stats['batch_time']['throughput']:.2f} batches/s")
        
        if 'epoch_time' in stats:
            print(f"Epoch Time: {stats['epoch_time']['avg']:.2f}s avg, {stats['epoch_time']['total']:.2f}s total")
        
        print("="*60)
    
    def detect_bottlenecks(self):
        """Detect potential performance bottlenecks"""
        bottlenecks = []
        stats = self.get_summary_stats()
        
        # High CPU usage
        if 'cpu' in stats and stats['cpu']['avg'] > 80:
            bottlenecks.append("High CPU usage detected - consider reducing data preprocessing or num_workers")
        
        # High memory usage
        if 'memory' in stats and stats['memory']['avg'] > 85:
            bottlenecks.append("High memory usage detected - consider reducing batch size or enabling gradient checkpointing")
        
        # High GPU memory usage
        if 'gpu_memory' in stats and stats['gpu_memory']['allocated_avg'] > 8:
            bottlenecks.append("High GPU memory usage - consider reducing batch size or model size")
        
        # Slow batch processing
        if 'batch_time' in stats and stats['batch_time']['avg'] > 2.0:
            bottlenecks.append("Slow batch processing - check data loading pipeline and model efficiency")
        
        # Low GPU utilization
        if self.gpu_utilization and np.mean(self.gpu_utilization) < 50:
            bottlenecks.append("Low GPU utilization - data loading might be the bottleneck")
        
        return bottlenecks

# Decorator for automatic timing
def profile_function(profiler, name=None):
    """Decorator to automatically profile function execution time"""
    def decorator(func):
        func_name = name or func.__name__
        def wrapper(*args, **kwargs):
            with profiler.timer(func_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator

# Global profiler instance
global_profiler = None

def get_profiler():
    """Get or create global profiler instance"""
    global global_profiler
    if global_profiler is None:
        global_profiler = PerformanceProfiler()
    return global_profiler

