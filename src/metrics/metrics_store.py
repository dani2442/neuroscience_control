"""Metrics storage and logging."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime


class MetricsStore:
    """
    Store and manage training/evaluation metrics.
    
    Supports:
    - Storing metrics per epoch/step
    - Saving/loading from JSON
    - Computing best metrics
    - Comparing multiple experiments
    """
    
    def __init__(
        self,
        experiment_name: str = "experiment",
        save_dir: str = "results/metrics"
    ):
        """
        Initialize metrics store.
        
        Args:
            experiment_name: Name of the experiment
            save_dir: Directory to save metrics
        """
        self.experiment_name = experiment_name
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.train_metrics: List[Dict[str, float]] = []
        self.val_metrics: List[Dict[str, float]] = []
        self.test_metrics: Dict[str, Dict[str, float]] = {}
        self.hyperparameters: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {
            'created': datetime.now().isoformat(),
            'experiment_name': experiment_name
        }
    
    def log_train(self, epoch: int, metrics: Dict[str, float]):
        """
        Log training metrics for an epoch.
        
        Args:
            epoch: Epoch number
            metrics: Dictionary of metric values
        """
        entry = {'epoch': epoch, **metrics}
        self.train_metrics.append(entry)
    
    def log_val(self, epoch: int, metrics: Dict[str, float]):
        """
        Log validation metrics for an epoch.
        
        Args:
            epoch: Epoch number
            metrics: Dictionary of metric values
        """
        entry = {'epoch': epoch, **metrics}
        self.val_metrics.append(entry)
    
    def log_test(self, metrics: Dict[str, float], label: str = "test"):
        """
        Log test metrics under a label (e.g. ``"test_inter"``, ``"test_intra"``).

        Args:
            metrics: Dictionary of metric values.
            label: Key under which to store these metrics.
        """
        self.test_metrics[label] = metrics
    
    def set_hyperparameters(self, hyperparams: Dict[str, Any]):
        """Store hyperparameters."""
        self.hyperparameters = hyperparams
    
    def get_best_val_metric(
        self,
        metric_name: str,
        mode: str = "min"
    ) -> Dict[str, Any]:
        """
        Get best validation metric.
        
        Args:
            metric_name: Name of metric to find best for
            mode: "min" or "max"
            
        Returns:
            Dictionary with best epoch and metric value
        """
        if not self.val_metrics:
            return {'epoch': None, 'value': None}
        
        if mode == "min":
            best_idx = min(
                range(len(self.val_metrics)),
                key=lambda i: self.val_metrics[i].get(metric_name, float('inf'))
            )
        else:
            best_idx = max(
                range(len(self.val_metrics)),
                key=lambda i: self.val_metrics[i].get(metric_name, float('-inf'))
            )
        
        return {
            'epoch': self.val_metrics[best_idx]['epoch'],
            'value': self.val_metrics[best_idx].get(metric_name)
        }
    
    def get_metric_history(
        self,
        metric_name: str,
        split: str = "train"
    ) -> List[float]:
        """
        Get history of a specific metric.
        
        Args:
            metric_name: Name of the metric
            split: "train" or "val"
            
        Returns:
            List of metric values
        """
        metrics = self.train_metrics if split == "train" else self.val_metrics
        return [m.get(metric_name) for m in metrics if metric_name in m]
    
    def save(self, filename: Optional[str] = None):
        """
        Save metrics to JSON file.
        
        Args:
            filename: Optional custom filename
        """
        if filename is None:
            filename = f"{self.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = self.save_dir / filename
        
        data = {
            'metadata': self.metadata,
            'hyperparameters': self.hyperparameters,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'test_metrics': self.test_metrics
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"Metrics saved to {filepath}")
        return filepath
    
    @classmethod
    def load(cls, filepath: str) -> "MetricsStore":
        """
        Load metrics from JSON file.
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            MetricsStore instance
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        store = cls(
            experiment_name=data['metadata'].get('experiment_name', 'loaded'),
            save_dir=str(Path(filepath).parent)
        )
        
        store.metadata = data['metadata']
        store.hyperparameters = data.get('hyperparameters', {})
        store.train_metrics = data.get('train_metrics', [])
        store.val_metrics = data.get('val_metrics', [])
        raw_test = data.get('test_metrics', {})
        # Support old flat format (Dict[str, float]) and new labeled format
        if raw_test and not isinstance(next(iter(raw_test.values())), dict):
            store.test_metrics = {"test": raw_test}
        else:
            store.test_metrics = raw_test
        
        return store
    

