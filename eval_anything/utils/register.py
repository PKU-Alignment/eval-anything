"""
注册器，实现metric和template的注册
"""

class MetricRegistry:
    _registry = {}

    @classmethod
    def register(cls, name: str):
        def decorator(metric_cls):
            cls._registry[name] = metric_cls
            metric_cls._registered_name = name
            return metric_cls
        return decorator

    @classmethod
    def get_metric(cls, name: str, *args, **kwargs):
        if name not in cls._registry:
            raise ValueError(f"Metric '{name}' is not registered!")
        return cls._registry[name](*args, **kwargs)
    
class TemplateRegistry:
    _registry = {}

    @classmethod
    def register(cls, name: str):
        def decorator(template_cls):
            cls._registry[name] = template_cls
            template_cls.__name__ = name
            return template_cls
        return decorator

    @classmethod
    def get_template(cls, name: str, *args, **kwargs):
        if name not in cls._registry:
            raise ValueError(f"Template '{name}' is not registered!")
        return cls._registry[name](*args, **kwargs)
    
class BenchmarkRegistry:
    _registry = {}

    @classmethod
    def register(cls, name: str):
        def decorator(benchmark_cls):
            cls._registry[name] = benchmark_cls
            benchmark_cls._registered_name = name
            return benchmark_cls
        return decorator
    
    @classmethod
    def get_benchmark(cls, name: str):
        if name not in cls._registry:
            raise ValueError(f"Benchmark '{name}' is not registered!")
        return cls._registry[name]
    
class MMDatasetRegistry:
    _registry = {}

    @classmethod
    def register(cls, name: str):
        def decorator(mm_dataset_cls):
            cls._registry[name] = mm_dataset_cls
            mm_dataset_cls._registered_name = name
            return mm_dataset_cls
        return decorator
    
    @classmethod
    def get_mm_dataset(cls, name: str):
        if name not in cls._registry:
            raise ValueError(f"Dataset '{name}' is not registered!")
        return cls._registry[name]

class MMDataManagerRegistry:
    _registry = {}

    @classmethod
    def register(cls, name: str):
        def decorator(mm_data_manager_cls):
            cls._registry[name] = mm_data_manager_cls
            mm_data_manager_cls._registered_name = name
            return mm_data_manager_cls
        return decorator
    
    @classmethod
    def get_mm_data_manager(cls, name: str):
        if name not in cls._registry:
            raise ValueError(f"DataManager '{name}' is not registered!")
        return cls._registry[name]