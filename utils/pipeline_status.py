"""
Pipeline status tracking for experiment post-processing.

Tracks the status of aggregation, zero-cost proxy computation, and analysis
steps for experiment collections.
"""
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime


class PipelineStatus:
    """Tracks pipeline status for an experiment collection."""

    def __init__(self, collection_dir: Path):
        self.collection_dir = collection_dir
        self.status_file = collection_dir / "pipeline_status.yaml"

    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        if not self.status_file.exists():
            return self._get_default_status()

        try:
            with open(self.status_file, 'r') as f:
                return yaml.safe_load(f) or self._get_default_status()
        except Exception:
            return self._get_default_status()

    def _get_default_status(self) -> Dict[str, Any]:
        """Get default status structure."""
        return {
            "aggregation": {
                "status": "not_started",
                "timestamp": None,
                "output_path": None,
                "log_path": None
            },
            "zero_cost_proxies": {
                "status": "not_started",
                "timestamp": None,
                "output_paths": [],
                "log_path": None
            },
            "analysis": {
                "status": "not_started",
                "timestamp": None,
                "output_paths": [],
                "log_path": None
            }
        }

    def update_step(self, step: str, status: str, **kwargs) -> None:
        """
        Update status for a pipeline step.

        Args:
            step: One of "aggregation", "zero_cost_proxies", "analysis"
            status: One of "not_started", "completed", "failed"
            **kwargs: Additional fields like output_path, log_path, etc.
        """
        current_status = self.get_status()

        if step not in current_status:
            raise ValueError(f"Unknown step: {step}")

        current_status[step]["status"] = status
        current_status[step]["timestamp"] = datetime.now().isoformat()

        # Update other fields
        for key, value in kwargs.items():
            current_status[step][key] = value

        # Write back to file
        self.collection_dir.mkdir(parents=True, exist_ok=True)
        with open(self.status_file, 'w') as f:
            yaml.dump(current_status, f, default_flow_style=False, sort_keys=False)

    def mark_aggregation_complete(self, parquet_path: Path, log_path: Optional[Path] = None) -> None:
        """Mark aggregation as completed."""
        self.update_step(
            "aggregation",
            "completed",
            output_path=str(parquet_path.relative_to(self.collection_dir)) if parquet_path else None,
            log_path=str(log_path.relative_to(self.collection_dir)) if log_path else None
        )

    def mark_aggregation_failed(self, log_path: Optional[Path] = None) -> None:
        """Mark aggregation as failed."""
        self.update_step(
            "aggregation",
            "failed",
            log_path=str(log_path.relative_to(self.collection_dir)) if log_path else None
        )

    def mark_zcp_complete(self, output_paths: List[Path], log_path: Optional[Path] = None) -> None:
        """Mark zero-cost proxy computation as completed."""
        output_paths_str = [str(p.relative_to(self.collection_dir)) for p in output_paths] if output_paths else []
        self.update_step(
            "zero_cost_proxies",
            "completed",
            output_paths=output_paths_str,
            log_path=str(log_path.relative_to(self.collection_dir)) if log_path else None
        )

    def mark_zcp_failed(self, log_path: Optional[Path] = None) -> None:
        """Mark zero-cost proxy computation as failed."""
        self.update_step(
            "zero_cost_proxies",
            "failed",
            log_path=str(log_path.relative_to(self.collection_dir)) if log_path else None
        )

    def mark_analysis_complete(self, output_paths: List[Path], log_path: Optional[Path] = None) -> None:
        """Mark analysis as completed."""
        output_paths_str = [str(p.relative_to(self.collection_dir)) for p in output_paths] if output_paths else []
        self.update_step(
            "analysis",
            "completed",
            output_paths=output_paths_str,
            log_path=str(log_path.relative_to(self.collection_dir)) if log_path else None
        )

    def mark_analysis_failed(self, log_path: Optional[Path] = None) -> None:
        """Mark analysis as failed."""
        self.update_step(
            "analysis",
            "failed",
            log_path=str(log_path.relative_to(self.collection_dir)) if log_path else None
        )


def get_pipeline_status(collection_dir: Path) -> Dict[str, Any]:
    """Get pipeline status for a collection."""
    tracker = PipelineStatus(collection_dir)
    return tracker.get_status()


def update_pipeline_status(collection_dir: Path, step: str, status: str, **kwargs) -> None:
    """Update pipeline status for a collection."""
    tracker = PipelineStatus(collection_dir)
    tracker.update_step(step, status, **kwargs)


def mark_aggregation_complete(collection_dir: Path, parquet_path: Path, log_path: Optional[Path] = None) -> None:
    """Mark aggregation as completed for a collection."""
    tracker = PipelineStatus(collection_dir)
    tracker.mark_aggregation_complete(parquet_path, log_path)


def mark_aggregation_failed(collection_dir: Path, log_path: Optional[Path] = None) -> None:
    """Mark aggregation as failed for a collection."""
    tracker = PipelineStatus(collection_dir)
    tracker.mark_aggregation_failed(log_path)


def mark_zcp_complete(collection_dir: Path, output_paths: List[Path], log_path: Optional[Path] = None) -> None:
    """Mark zero-cost proxy computation as completed for a collection."""
    tracker = PipelineStatus(collection_dir)
    tracker.mark_zcp_complete(output_paths, log_path)


def mark_zcp_failed(collection_dir: Path, log_path: Optional[Path] = None) -> None:
    """Mark zero-cost proxy computation as failed for a collection."""
    tracker = PipelineStatus(collection_dir)
    tracker.mark_zcp_failed(log_path)


def mark_analysis_complete(collection_dir: Path, output_paths: List[Path], log_path: Optional[Path] = None) -> None:
    """Mark analysis as completed for a collection."""
    tracker = PipelineStatus(collection_dir)
    tracker.mark_analysis_complete(output_paths, log_path)


def mark_analysis_failed(collection_dir: Path, log_path: Optional[Path] = None) -> None:
    """Mark analysis as failed for a collection."""
    tracker = PipelineStatus(collection_dir)
    tracker.mark_analysis_failed(log_path)