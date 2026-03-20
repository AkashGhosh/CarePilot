#!/usr/bin/env python3
"""
Compute tracking for CVPR compute reporting.
Tracks GPU/CPU hours, hardware specs, and training metrics.
"""

import os
import json
import time
import platform
import psutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import torch

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

class ComputeTracker:
    """
    Track compute usage for CVPR reporting.
    """
    
    def __init__(self, output_dir: str = None):
        """
        Initialize compute tracker.
        
        Args:
            output_dir: Directory to save compute report
        """
        self.output_dir = Path(output_dir) if output_dir else Path("finetuning/logs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.start_time = None
        self.end_time = None
        self.gpu_start_times = {}
        self.gpu_utilization = []
        self.cpu_utilization = []
        
        # Hardware specs
        self.hardware_specs = self._get_hardware_specs()
        
        # Training metrics
        self.training_metrics = {
            "model_params": None,
            "training_set_size": None,
            "epochs": None,
            "batch_size": None,
            "gradient_accumulation_steps": None,
            "effective_batch_size": None,
            "total_steps": None,
            "framework": "PyTorch",
            "mixed_precision": None,
            "distributed_training": False,
            "distributed_method": None,
        }
        
        # Compute metrics
        self.compute_metrics = {
            "gpu_hours": 0.0,
            "cpu_hours": 0.0,
            "total_compute_hours": 0.0,
            "training_time_seconds": 0.0,
            "inference_compute_per_1k": None,
            "total_inference_compute": None,
        }
        
    def _get_hardware_specs(self) -> Dict[str, Any]:
        """Get hardware specifications."""
        specs = {}
        
        # CPU
        try:
            specs["cpu_model"] = platform.processor()
            if not specs["cpu_model"] or specs["cpu_model"] == "":
                # Try alternative method
                if platform.system() == "Linux":
                    try:
                        with open("/proc/cpuinfo", "r") as f:
                            for line in f:
                                if "model name" in line:
                                    specs["cpu_model"] = line.split(":")[1].strip()
                                    break
                    except:
                        specs["cpu_model"] = "Unknown"
                else:
                    specs["cpu_model"] = "Unknown"
            specs["cpu_cores"] = psutil.cpu_count(logical=False)  # Physical cores
            specs["cpu_threads"] = psutil.cpu_count(logical=True)  # Logical cores
        except Exception as e:
            specs["cpu_model"] = "Unknown"
            specs["cpu_cores"] = 0
            specs["cpu_threads"] = 0
        
        # RAM
        try:
            specs["system_ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 2)
        except:
            specs["system_ram_gb"] = 0
        
        # GPU
        specs["gpu_model"] = "None"
        specs["num_gpus"] = 0
        specs["gpu_memory_gb"] = 0
        
        if torch.cuda.is_available():
            specs["num_gpus"] = torch.cuda.device_count()
            if specs["num_gpus"] > 0:
                gpu_name = torch.cuda.get_device_name(0)
                specs["gpu_model"] = gpu_name
                specs["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        
        # Infrastructure detection
        specs["infrastructure_type"] = self._detect_infrastructure()
        specs["cloud_provider"] = self._detect_cloud_provider()
        specs["instance_types"] = None  # User should fill this
        
        return specs
    
    def _detect_infrastructure(self) -> str:
        """Detect infrastructure type."""
        # Check for Colab
        if os.path.exists("/content") or os.environ.get("COLAB_GPU", "").strip() != "":
            return "Cloud - Commercial cloud provider"
        
        # Check for common cloud indicators
        if os.path.exists("/sys/class/dmi/id/product_name"):
            try:
                with open("/sys/class/dmi/id/product_name", "r") as f:
                    product = f.read().strip().lower()
                    if "google" in product or "gcp" in product:
                        return "Cloud - Commercial cloud provider"
                    if "amazon" in product or "aws" in product:
                        return "Cloud - Commercial cloud provider"
                    if "microsoft" in product or "azure" in product:
                        return "Cloud - Commercial cloud provider"
            except:
                pass
        
        # Check environment variables
        if os.environ.get("SLURM_JOB_ID"):
            return "Shared cluster - Multi-user academic/research cluster"
        
        # Default to on-premises
        return "On-premises - Institution/company owned hardware"
    
    def _detect_cloud_provider(self) -> Optional[str]:
        """Detect cloud provider."""
        if os.path.exists("/content"):
            return "Google Cloud"
        
        # Check metadata services
        try:
            import urllib.request
            # AWS
            try:
                response = urllib.request.urlopen("http://169.254.169.254/latest/meta-data/instance-type", timeout=1)
                return "AWS"
            except:
                pass
            
            # GCP
            try:
                response = urllib.request.urlopen("http://metadata.google.internal/computeMetadata/v1/instance/name", 
                                                 timeout=1, headers={"Metadata-Flavor": "Google"})
                return "Google Cloud"
            except:
                pass
            
            # Azure
            try:
                response = urllib.request.urlopen("http://169.254.169.254/metadata/instance?api-version=2021-02-01",
                                                 timeout=1, headers={"Metadata": "true"})
                return "Microsoft Azure"
            except:
                pass
        except:
            pass
        
        return None
    
    def start_training(self):
        """Start tracking training."""
        self.start_time = time.time()
        self.gpu_start_times = {}
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                self.gpu_start_times[i] = time.time()
    
    def end_training(self):
        """End tracking training."""
        self.end_time = time.time()
        
        if self.start_time:
            self.compute_metrics["training_time_seconds"] = self.end_time - self.start_time
            training_hours = self.compute_metrics["training_time_seconds"] / 3600
            
            # Calculate GPU hours
            if torch.cuda.is_available() and self.gpu_start_times:
                num_gpus = torch.cuda.device_count()
                self.compute_metrics["gpu_hours"] = training_hours * num_gpus
            else:
                self.compute_metrics["gpu_hours"] = 0.0
            
            # Calculate CPU hours (approximate - using all cores)
            cpu_cores = self.hardware_specs.get("cpu_cores", 1)
            self.compute_metrics["cpu_hours"] = training_hours * cpu_cores
            
            # Total compute hours (GPU + CPU)
            self.compute_metrics["total_compute_hours"] = (
                self.compute_metrics["gpu_hours"] + self.compute_metrics["cpu_hours"]
            )
    
    def set_training_metrics(
        self,
        model_params: Optional[int] = None,
        training_set_size: Optional[int] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        gradient_accumulation_steps: Optional[int] = None,
        total_steps: Optional[int] = None,
        mixed_precision: Optional[str] = None,
        distributed_training: bool = False,
        distributed_method: Optional[str] = None,
    ):
        """Set training metrics."""
        if model_params is not None:
            self.training_metrics["model_params"] = model_params
        if training_set_size is not None:
            self.training_metrics["training_set_size"] = training_set_size
        if epochs is not None:
            self.training_metrics["epochs"] = epochs
        if batch_size is not None:
            self.training_metrics["batch_size"] = batch_size
        if gradient_accumulation_steps is not None:
            self.training_metrics["gradient_accumulation_steps"] = gradient_accumulation_steps
            if batch_size is not None:
                self.training_metrics["effective_batch_size"] = (
                    batch_size * gradient_accumulation_steps
                )
        if total_steps is not None:
            self.training_metrics["total_steps"] = total_steps
        if mixed_precision is not None:
            self.training_metrics["mixed_precision"] = mixed_precision
        if distributed_training:
            self.training_metrics["distributed_training"] = True
        if distributed_method is not None:
            self.training_metrics["distributed_method"] = distributed_method
    
    def save_report(self, filename: str = "compute_report.json"):
        """
        Save compute report to JSON file in CVPR form format.
        
        Args:
            filename: Output filename
        """
        # Calculate calendar months if dates are available
        calendar_months = None
        if self.start_time and self.end_time:
            start_date = datetime.fromtimestamp(self.start_time)
            end_date = datetime.fromtimestamp(self.end_time)
            delta = end_date - start_date
            calendar_months = round(delta.days / 30.44, 2)  # Average days per month
        
        report = {
            # Section 1: Hardware Configuration
            "hardware_configuration": {
                "cpu_model": self.hardware_specs.get("cpu_model", "Unknown"),
                "cpu_cores": self.hardware_specs.get("cpu_cores", 0),
                "gpu_model": self.hardware_specs.get("gpu_model", "None"),
                "num_gpus": self.hardware_specs.get("num_gpus", 0),
                "gpu_memory_gb": self.hardware_specs.get("gpu_memory_gb", 0),
                "system_ram_gb": self.hardware_specs.get("system_ram_gb", 0),
                "storage_type": None,  # User should fill
            },
            
            # Section 2: Infrastructure
            "infrastructure": {
                "infrastructure_type": self.hardware_specs.get("infrastructure_type", "Unknown"),
                "cloud_provider": self.hardware_specs.get("cloud_provider", None),
                "instance_types": self.hardware_specs.get("instance_types", None),  # User should fill
            },
            
            # Section 3: Task Information
            "task_information": {
                "task_category": "Multimodal Learning (Vision + Language)",  # User should verify
                "task_category_other": None,  # User should fill if "Other"
            },
            
            # Section 4: Performance Metrics
            "performance_metrics": {
                "performance_metric": None,  # User should fill (e.g., "Action Accuracy", "Target Accuracy")
                "method_performance": None,  # User should fill (e.g., 0.85)
                "baseline_method": None,  # User should fill
                "baseline_performance": None,  # User should fill
                "percentage_improvement": None,  # User should fill
            },
            
            # Section 5: Dataset Information
            "dataset_information": {
                "dataset_name": None,  # User should fill
                "training_set_size": self.training_metrics.get("training_set_size", None),
                "test_set_size": None,  # User should fill
            },
            
            # Section 6: Model Information
            "model_information": {
                "model_params": self.training_metrics.get("model_params", None),
                "framework": self.training_metrics.get("framework", "PyTorch"),
                "mixed_precision": self.training_metrics.get("mixed_precision", None),
                "distributed_training": self.training_metrics.get("distributed_training", False),
                "distributed_method": self.training_metrics.get("distributed_method", None),
            },
            
            # Section 7: Training Configuration
            "training_configuration": {
                "epochs": self.training_metrics.get("epochs", None),
                "batch_size": self.training_metrics.get("batch_size", None),
                "gradient_accumulation_steps": self.training_metrics.get("gradient_accumulation_steps", None),
                "effective_batch_size": self.training_metrics.get("effective_batch_size", None),
                "total_steps": self.training_metrics.get("total_steps", None),
            },
            
            # Section 8: Compute Metrics
            "compute_metrics": {
                "compute_metric": "GPU+CPU Hours",  # or "FLOPs"
                "method_compute_total": self.compute_metrics["total_compute_hours"],
                "gpu_hours": self.compute_metrics["gpu_hours"],
                "cpu_hours": self.compute_metrics["cpu_hours"],
                "total_compute_hours": self.compute_metrics["total_compute_hours"],
                "training_time_seconds": self.compute_metrics["training_time_seconds"],
            },
            
            # Section 9: FLOPs (if using FLOPs metric)
            "flops_information": {
                "flops_per_forward": None,  # User should calculate
                "flops_per_instance": None,  # User should fill
                "flops_tool": None,  # User should fill (ptflops, fvcore, thop, etc.)
                "flops_method_description": None,  # User should fill
            },
            
            # Section 10: Inference Compute
            "inference_compute": {
                "inference_compute_per_1k": None,  # User should fill
                "total_inference_compute": None,  # User should fill
            },
            
            # Section 11: Efficiency
            "efficiency": {
                "efficiency_ratio": None,  # User should fill
                "efficiency_reasoning": None,  # User should fill (3 lines)
            },
            
            # Section 12: Development Compute Breakdown
            "development_breakdown": {
                "total_dev_compute_hours": self.compute_metrics["total_compute_hours"],
                "pct_training": 100.0,  # Default - user should adjust
                "training_stage_compute_hours": self.compute_metrics["total_compute_hours"],
                "pct_finetuning": 100.0,  # This is finetuning
                "finetuning_compute_hours": self.compute_metrics["total_compute_hours"],
                "pct_distillation": 0.0,
                "distillation_compute_hours": 0.0,
                "pct_hyper_search": 0.0,
                "search_hyper_compute_hours": 0.0,
                "search_configs": 0,
                "pct_ablation": 0.0,
                "ablation_compute_hours": 0.0,
                "ablation_configs": 0,
                "pct_inference_dev": 0.0,
                "inference_dev_compute_hours": 0.0,
                "pct_failed": 0.0,
                "failed_compute_hours": 0.0,
                "pct_other": 0.0,
                "other_compute_hours": 0.0,
            },
            
            # Section 13: Development Timeline
            "development_timeline": {
                "dev_start_date": datetime.fromtimestamp(self.start_time).strftime("%Y-%m-%d") if self.start_time else None,
                "dev_end_date": datetime.fromtimestamp(self.end_time).strftime("%Y-%m-%d") if self.end_time else None,
                "calendar_months": calendar_months,
                "training_start_time": datetime.fromtimestamp(self.start_time).isoformat() if self.start_time else None,
                "training_end_time": datetime.fromtimestamp(self.end_time).isoformat() if self.end_time else None,
            },
            
            # Section 14: Log File Information
            "log_file_information": {
                "log_file_size": None,  # User should fill
                "experiment_date_range": None,  # User should fill
                "logs_anonymized": False,  # User should verify
                "logs_reviewed": False,  # User should verify
                "logs_consent": False,  # User should verify
            },
            
            # Section 15: Additional Information
            "additional_information": {
                "additional_notes": None,  # User should fill (3 lines)
            },
        }
        
        output_path = self.output_dir / filename
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        
        # Also save a human-readable markdown version
        self._save_markdown_report(report, output_path.with_suffix('.md'))
        
        # Also save a CSV summary for easy form filling
        self.save_csv_summary("compute_report_summary.csv")
        
        print(f"\n📊 Compute report saved to: {output_path}")
        print(f"📄 Human-readable report saved to: {output_path.with_suffix('.md')}")
        print(f"📋 CSV summary saved to: {self.output_dir / 'compute_report_summary.csv'}")
        print(f"⚠️  Please review and fill in user-specified fields before CVPR submission")
        
        return output_path
    
    def _save_markdown_report(self, report: Dict[str, Any], output_path: Path):
        """Save a human-readable markdown version of the report."""
        md_content = []
        md_content.append("# CVPR Compute Reporting Form Data\n")
        md_content.append("**Generated automatically during training**\n")
        md_content.append(f"**Generated on**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        md_content.append("---\n\n")
        
        # Section 1: Hardware Configuration
        md_content.append("## 1. Hardware Configuration\n\n")
        hw = report["hardware_configuration"]
        md_content.append(f"- **CPU Model**: {hw.get('cpu_model', 'N/A')}\n")
        md_content.append(f"- **CPU Cores**: {hw.get('cpu_cores', 'N/A')}\n")
        md_content.append(f"- **GPU Model**: {hw.get('gpu_model', 'N/A')}\n")
        md_content.append(f"- **Number of GPUs**: {hw.get('num_gpus', 'N/A')}\n")
        md_content.append(f"- **GPU Memory (GB)**: {hw.get('gpu_memory_gb', 'N/A')}\n")
        md_content.append(f"- **System RAM (GB)**: {hw.get('system_ram_gb', 'N/A')}\n")
        md_content.append(f"- **Storage Type**: {hw.get('storage_type', '⚠️ MANUAL: Fill this')}\n\n")
        
        # Section 2: Infrastructure
        md_content.append("## 2. Infrastructure\n\n")
        infra = report["infrastructure"]
        md_content.append(f"- **Infrastructure Type**: {infra.get('infrastructure_type', 'N/A')}\n")
        md_content.append(f"- **Cloud Provider**: {infra.get('cloud_provider', 'N/A')}\n")
        md_content.append(f"- **Instance Types**: {infra.get('instance_types', '⚠️ MANUAL: Fill this')}\n\n")
        
        # Section 3: Task Information
        md_content.append("## 3. Task Information\n\n")
        task = report["task_information"]
        md_content.append(f"- **Task Category**: {task.get('task_category', 'N/A')}\n")
        if task.get('task_category_other'):
            md_content.append(f"- **Task Category (Other)**: {task.get('task_category_other')}\n")
        md_content.append("\n")
        
        # Section 4: Performance Metrics
        md_content.append("## 4. Performance Metrics\n\n")
        perf = report["performance_metrics"]
        md_content.append(f"- **Performance Metric**: {perf.get('performance_metric', '⚠️ MANUAL: Fill this (e.g., Action Accuracy)')}\n")
        md_content.append(f"- **Method Performance**: {perf.get('method_performance', '⚠️ MANUAL: Fill this')}\n")
        md_content.append(f"- **Baseline Method**: {perf.get('baseline_method', '⚠️ MANUAL: Fill this')}\n")
        md_content.append(f"- **Baseline Performance**: {perf.get('baseline_performance', '⚠️ MANUAL: Fill this')}\n")
        md_content.append(f"- **Percentage Improvement**: {perf.get('percentage_improvement', '⚠️ MANUAL: Fill this')}\n\n")
        
        # Section 5: Dataset Information
        md_content.append("## 5. Dataset Information\n\n")
        dataset = report["dataset_information"]
        md_content.append(f"- **Dataset Name**: {dataset.get('dataset_name', '⚠️ MANUAL: Fill this')}\n")
        md_content.append(f"- **Training Set Size**: {dataset.get('training_set_size', 'N/A')}\n")
        md_content.append(f"- **Test Set Size**: {dataset.get('test_set_size', '⚠️ MANUAL: Fill this')}\n\n")
        
        # Section 6: Model Information
        md_content.append("## 6. Model Information\n\n")
        model = report["model_information"]
        md_content.append(f"- **Model Parameters**: {model.get('model_params', 'N/A')}\n")
        md_content.append(f"- **Framework**: {model.get('framework', 'N/A')}\n")
        md_content.append(f"- **Mixed Precision**: {model.get('mixed_precision', 'N/A')}\n")
        md_content.append(f"- **Distributed Training**: {model.get('distributed_training', 'N/A')}\n")
        if model.get('distributed_method'):
            md_content.append(f"- **Distributed Method**: {model.get('distributed_method')}\n")
        md_content.append("\n")
        
        # Section 7: Training Configuration
        md_content.append("## 7. Training Configuration\n\n")
        train = report["training_configuration"]
        md_content.append(f"- **Epochs**: {train.get('epochs', 'N/A')}\n")
        md_content.append(f"- **Batch Size**: {train.get('batch_size', 'N/A')}\n")
        md_content.append(f"- **Gradient Accumulation Steps**: {train.get('gradient_accumulation_steps', 'N/A')}\n")
        md_content.append(f"- **Effective Batch Size**: {train.get('effective_batch_size', 'N/A')}\n")
        md_content.append(f"- **Total Steps**: {train.get('total_steps', 'N/A')}\n\n")
        
        # Section 8: Compute Metrics
        md_content.append("## 8. Compute Metrics\n\n")
        compute = report["compute_metrics"]
        md_content.append(f"- **Compute Metric**: {compute.get('compute_metric', 'N/A')}\n")
        md_content.append(f"- **Total Compute Hours (GPU+CPU)**: {compute.get('total_compute_hours', 0):.2f}\n")
        md_content.append(f"- **GPU Hours**: {compute.get('gpu_hours', 0):.2f}\n")
        md_content.append(f"- **CPU Hours**: {compute.get('cpu_hours', 0):.2f}\n")
        md_content.append(f"- **Training Time**: {compute.get('training_time_seconds', 0):.0f} seconds ({compute.get('training_time_seconds', 0) / 3600:.2f} hours)\n\n")
        
        # Section 9: FLOPs (if applicable)
        md_content.append("## 9. FLOPs Information (if using FLOPs metric)\n\n")
        flops = report["flops_information"]
        md_content.append(f"- **FLOPs per Forward**: {flops.get('flops_per_forward', '⚠️ MANUAL: Calculate this')}\n")
        md_content.append(f"- **FLOPs per Instance**: {flops.get('flops_per_instance', '⚠️ MANUAL: Fill this')}\n")
        md_content.append(f"- **FLOPs Tool**: {flops.get('flops_tool', '⚠️ MANUAL: Fill this (ptflops, fvcore, etc.)')}\n")
        md_content.append(f"- **FLOPs Method Description**: {flops.get('flops_method_description', '⚠️ MANUAL: Fill this')}\n\n")
        
        # Section 10: Inference Compute
        md_content.append("## 10. Inference Compute\n\n")
        inference = report["inference_compute"]
        md_content.append(f"- **Inference Compute per 1k**: {inference.get('inference_compute_per_1k', '⚠️ MANUAL: Fill this')}\n")
        md_content.append(f"- **Total Inference Compute**: {inference.get('total_inference_compute', '⚠️ MANUAL: Fill this')}\n\n")
        
        # Section 11: Efficiency
        md_content.append("## 11. Efficiency\n\n")
        eff = report["efficiency"]
        md_content.append(f"- **Efficiency Ratio**: {eff.get('efficiency_ratio', '⚠️ MANUAL: Fill this')}\n")
        md_content.append(f"- **Efficiency Reasoning**: {eff.get('efficiency_reasoning', '⚠️ MANUAL: Fill this (3 lines)')}\n\n")
        
        # Section 12: Development Breakdown
        md_content.append("## 12. Development Compute Breakdown\n\n")
        breakdown = report["development_breakdown"]
        md_content.append(f"- **Total Dev Compute Hours**: {breakdown.get('total_dev_compute_hours', 0):.2f}\n")
        md_content.append(f"- **Training Stage**: {breakdown.get('pct_training', 0):.1f}% ({breakdown.get('training_stage_compute_hours', 0):.2f} hours)\n")
        md_content.append(f"- **Finetuning**: {breakdown.get('pct_finetuning', 0):.1f}% ({breakdown.get('finetuning_compute_hours', 0):.2f} hours)\n")
        md_content.append(f"- **Distillation**: {breakdown.get('pct_distillation', 0):.1f}% ({breakdown.get('distillation_compute_hours', 0):.2f} hours)\n")
        md_content.append(f"- **Hyperparameter Search**: {breakdown.get('pct_hyper_search', 0):.1f}% ({breakdown.get('search_hyper_compute_hours', 0):.2f} hours, {breakdown.get('search_configs', 0)} configs)\n")
        md_content.append(f"- **Ablation Studies**: {breakdown.get('pct_ablation', 0):.1f}% ({breakdown.get('ablation_compute_hours', 0):.2f} hours, {breakdown.get('ablation_configs', 0)} configs)\n")
        md_content.append(f"- **Inference Development**: {breakdown.get('pct_inference_dev', 0):.1f}% ({breakdown.get('inference_dev_compute_hours', 0):.2f} hours)\n")
        md_content.append(f"- **Failed Runs**: {breakdown.get('pct_failed', 0):.1f}% ({breakdown.get('failed_compute_hours', 0):.2f} hours)\n")
        md_content.append(f"- **Other**: {breakdown.get('pct_other', 0):.1f}% ({breakdown.get('other_compute_hours', 0):.2f} hours)\n\n")
        md_content.append("⚠️ **Note**: Adjust percentages to match your actual development breakdown\n\n")
        
        # Section 13: Development Timeline
        md_content.append("## 13. Development Timeline\n\n")
        timeline = report["development_timeline"]
        md_content.append(f"- **Start Date**: {timeline.get('dev_start_date', 'N/A')}\n")
        md_content.append(f"- **End Date**: {timeline.get('dev_end_date', 'N/A')}\n")
        md_content.append(f"- **Calendar Months**: {timeline.get('calendar_months', 'N/A')}\n\n")
        
        # Section 14: Log File Information
        md_content.append("## 14. Log File Information\n\n")
        logs = report["log_file_information"]
        md_content.append(f"- **Log File Size**: {logs.get('log_file_size', '⚠️ MANUAL: Fill this')}\n")
        md_content.append(f"- **Experiment Date Range**: {logs.get('experiment_date_range', '⚠️ MANUAL: Fill this')}\n")
        md_content.append(f"- **Logs Anonymized**: {'✅' if logs.get('logs_anonymized') else '⚠️ MANUAL: Verify'} {logs.get('logs_anonymized', False)}\n")
        md_content.append(f"- **Logs Reviewed**: {'✅' if logs.get('logs_reviewed') else '⚠️ MANUAL: Verify'} {logs.get('logs_reviewed', False)}\n")
        md_content.append(f"- **Logs Consent**: {'✅' if logs.get('logs_consent') else '⚠️ MANUAL: Verify'} {logs.get('logs_consent', False)}\n\n")
        
        # Section 15: Additional Information
        md_content.append("## 15. Additional Information\n\n")
        additional = report["additional_information"]
        md_content.append(f"- **Additional Notes**: {additional.get('additional_notes', '⚠️ MANUAL: Fill this (3 lines)')}\n\n")
        
        # Footer
        md_content.append("---\n\n")
        md_content.append("## Instructions\n\n")
        md_content.append("1. Review all fields marked with ⚠️ MANUAL\n")
        md_content.append("2. Fill in performance metrics after running evaluation\n")
        md_content.append("3. Calculate FLOPs if using FLOPs metric\n")
        md_content.append("4. Adjust development breakdown percentages\n")
        md_content.append("5. Verify log file information\n")
        md_content.append("6. Use this data to fill the CVPR compute reporting form\n\n")
        md_content.append("**JSON file**: Use `compute_report.json` for programmatic access\n")
        md_content.append("**Markdown file**: Use this file for human review\n")
        
        with open(output_path, "w") as f:
            f.write("".join(md_content))
    
    def save_csv_summary(self, filename: str = "compute_report_summary.csv"):
        """
        Save a CSV summary for easy CVPR form filling.
        
        Args:
            filename: Output filename
        """
        import csv
        
        output_path = self.output_dir / filename
        
        # Create CSV with CVPR form fields
        rows = [
            # Hardware Configuration
            ["Section", "Field", "Value", "Status"],
            ["Hardware Configuration", "CPU Model", self.hardware_specs.get("cpu_model", "Unknown"), "Auto"],
            ["Hardware Configuration", "CPU Cores", self.hardware_specs.get("cpu_cores", 0), "Auto"],
            ["Hardware Configuration", "GPU Model", self.hardware_specs.get("gpu_model", "None"), "Auto"],
            ["Hardware Configuration", "Number of GPUs", self.hardware_specs.get("num_gpus", 0), "Auto"],
            ["Hardware Configuration", "GPU Memory (GB)", self.hardware_specs.get("gpu_memory_gb", 0), "Auto"],
            ["Hardware Configuration", "System RAM (GB)", self.hardware_specs.get("system_ram_gb", 0), "Auto"],
            ["Hardware Configuration", "Storage Type", "MANUAL: Fill this", "Manual"],
            
            # Infrastructure
            ["Infrastructure", "Infrastructure Type", self.hardware_specs.get("infrastructure_type", "Unknown"), "Auto"],
            ["Infrastructure", "Cloud Provider", self.hardware_specs.get("cloud_provider", "N/A"), "Auto"],
            ["Infrastructure", "Instance Types", "MANUAL: Fill this", "Manual"],
            
            # Task Information
            ["Task Information", "Task Category", "Multimodal Learning (Vision + Language)", "Verify"],
            ["Task Information", "Task Category (Other)", "N/A", "Manual"],
            
            # Performance Metrics
            ["Performance Metrics", "Performance Metric", "MANUAL: Fill this", "Manual"],
            ["Performance Metrics", "Method Performance", "MANUAL: Fill this", "Manual"],
            ["Performance Metrics", "Baseline Method", "MANUAL: Fill this", "Manual"],
            ["Performance Metrics", "Baseline Performance", "MANUAL: Fill this", "Manual"],
            ["Performance Metrics", "Percentage Improvement", "MANUAL: Fill this", "Manual"],
            
            # Dataset Information
            ["Dataset Information", "Dataset Name", "MANUAL: Fill this", "Manual"],
            ["Dataset Information", "Training Set Size", self.training_metrics.get("training_set_size", "N/A"), "Auto"],
            ["Dataset Information", "Test Set Size", "MANUAL: Fill this", "Manual"],
            
            # Model Information
            ["Model Information", "Model Parameters", self.training_metrics.get("model_params", "N/A"), "Auto"],
            ["Model Information", "Framework", self.training_metrics.get("framework", "PyTorch"), "Auto"],
            ["Model Information", "Mixed Precision", self.training_metrics.get("mixed_precision", "N/A"), "Auto"],
            ["Model Information", "Distributed Training", self.training_metrics.get("distributed_training", False), "Auto"],
            ["Model Information", "Distributed Method", self.training_metrics.get("distributed_method", "N/A"), "Auto"],
            
            # Training Configuration
            ["Training Configuration", "Epochs", self.training_metrics.get("epochs", "N/A"), "Auto"],
            ["Training Configuration", "Batch Size", self.training_metrics.get("batch_size", "N/A"), "Auto"],
            ["Training Configuration", "Gradient Accumulation Steps", self.training_metrics.get("gradient_accumulation_steps", "N/A"), "Auto"],
            ["Training Configuration", "Effective Batch Size", self.training_metrics.get("effective_batch_size", "N/A"), "Auto"],
            ["Training Configuration", "Total Steps", self.training_metrics.get("total_steps", "N/A"), "Auto"],
            
            # Compute Metrics
            ["Compute Metrics", "Compute Metric", "GPU+CPU Hours", "Auto"],
            ["Compute Metrics", "Total Compute Hours (GPU+CPU)", f"{self.compute_metrics['total_compute_hours']:.2f}", "Auto"],
            ["Compute Metrics", "GPU Hours", f"{self.compute_metrics['gpu_hours']:.2f}", "Auto"],
            ["Compute Metrics", "CPU Hours", f"{self.compute_metrics['cpu_hours']:.2f}", "Auto"],
            ["Compute Metrics", "Training Time (hours)", f"{self.compute_metrics['training_time_seconds'] / 3600:.2f}", "Auto"],
            
            # FLOPs (if applicable)
            ["FLOPs Information", "FLOPs per Forward", "MANUAL: Calculate this", "Manual"],
            ["FLOPs Information", "FLOPs per Instance", "MANUAL: Fill this", "Manual"],
            ["FLOPs Information", "FLOPs Tool", "MANUAL: Fill this (ptflops, fvcore, etc.)", "Manual"],
            ["FLOPs Information", "FLOPs Method Description", "MANUAL: Fill this", "Manual"],
            
            # Inference Compute
            ["Inference Compute", "Inference Compute per 1k", "MANUAL: Fill this", "Manual"],
            ["Inference Compute", "Total Inference Compute", "MANUAL: Fill this", "Manual"],
            
            # Efficiency
            ["Efficiency", "Efficiency Ratio", "MANUAL: Fill this", "Manual"],
            ["Efficiency", "Efficiency Reasoning", "MANUAL: Fill this (3 lines)", "Manual"],
            
            # Development Breakdown
            ["Development Breakdown", "Total Dev Compute Hours", f"{self.compute_metrics['total_compute_hours']:.2f}", "Auto"],
            ["Development Breakdown", "Training Stage (%)", "100.0", "Adjust"],
            ["Development Breakdown", "Training Stage (hours)", f"{self.compute_metrics['total_compute_hours']:.2f}", "Auto"],
            ["Development Breakdown", "Finetuning (%)", "100.0", "Adjust"],
            ["Development Breakdown", "Finetuning (hours)", f"{self.compute_metrics['total_compute_hours']:.2f}", "Auto"],
            ["Development Breakdown", "Distillation (%)", "0.0", "Adjust"],
            ["Development Breakdown", "Distillation (hours)", "0.0", "Adjust"],
            ["Development Breakdown", "Hyperparameter Search (%)", "0.0", "Adjust"],
            ["Development Breakdown", "Hyperparameter Search (hours)", "0.0", "Adjust"],
            ["Development Breakdown", "Hyperparameter Search (configs)", "0", "Adjust"],
            ["Development Breakdown", "Ablation Studies (%)", "0.0", "Adjust"],
            ["Development Breakdown", "Ablation Studies (hours)", "0.0", "Adjust"],
            ["Development Breakdown", "Ablation Studies (configs)", "0", "Adjust"],
            ["Development Breakdown", "Inference Development (%)", "0.0", "Adjust"],
            ["Development Breakdown", "Inference Development (hours)", "0.0", "Adjust"],
            ["Development Breakdown", "Failed Runs (%)", "0.0", "Adjust"],
            ["Development Breakdown", "Failed Runs (hours)", "0.0", "Adjust"],
            ["Development Breakdown", "Other (%)", "0.0", "Adjust"],
            ["Development Breakdown", "Other (hours)", "0.0", "Adjust"],
            
            # Development Timeline
            ["Development Timeline", "Start Date", datetime.fromtimestamp(self.start_time).strftime("%Y-%m-%d") if self.start_time else "N/A", "Auto"],
            ["Development Timeline", "End Date", datetime.fromtimestamp(self.end_time).strftime("%Y-%m-%d") if self.end_time else "N/A", "Auto"],
            ["Development Timeline", "Calendar Months", 
             f"{(datetime.fromtimestamp(self.end_time) - datetime.fromtimestamp(self.start_time)).days / 30.44:.2f}" 
             if self.start_time and self.end_time else "N/A", "Auto"],
            
            # Log File Information
            ["Log File Information", "Log File Size", "MANUAL: Fill this", "Manual"],
            ["Log File Information", "Experiment Date Range", "MANUAL: Fill this", "Manual"],
            ["Log File Information", "Logs Anonymized", "MANUAL: Verify", "Manual"],
            ["Log File Information", "Logs Reviewed", "MANUAL: Verify", "Manual"],
            ["Log File Information", "Logs Consent", "MANUAL: Verify", "Manual"],
            
            # Additional Information
            ["Additional Information", "Additional Notes", "MANUAL: Fill this (3 lines)", "Manual"],
        ]
        
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        
        print(f"📊 CSV summary saved to: {output_path}")
        return output_path
    
    def print_summary(self):
        """Print summary of tracked metrics."""
        print("\n" + "=" * 60)
        print("COMPUTE TRACKING SUMMARY")
        print("=" * 60)
        print(f"\nHardware:")
        print(f"  CPU: {self.hardware_specs.get('cpu_model', 'Unknown')} ({self.hardware_specs.get('cpu_cores', 0)} cores)")
        print(f"  GPU: {self.hardware_specs.get('gpu_model', 'None')} ({self.hardware_specs.get('num_gpus', 0)} GPUs, {self.hardware_specs.get('gpu_memory_gb', 0)} GB)")
        print(f"  RAM: {self.hardware_specs.get('system_ram_gb', 0)} GB")
        print(f"  Infrastructure: {self.hardware_specs.get('infrastructure_type', 'Unknown')}")
        if self.hardware_specs.get('cloud_provider'):
            print(f"  Cloud Provider: {self.hardware_specs.get('cloud_provider')}")
        
        print(f"\nTraining Metrics:")
        print(f"  Model Parameters: {self.training_metrics.get('model_params', 'N/A')}")
        print(f"  Training Set Size: {self.training_metrics.get('training_set_size', 'N/A')}")
        print(f"  Epochs: {self.training_metrics.get('epochs', 'N/A')}")
        print(f"  Batch Size: {self.training_metrics.get('batch_size', 'N/A')}")
        print(f"  Gradient Accumulation: {self.training_metrics.get('gradient_accumulation_steps', 'N/A')}")
        print(f"  Effective Batch Size: {self.training_metrics.get('effective_batch_size', 'N/A')}")
        print(f"  Framework: {self.training_metrics.get('framework', 'N/A')}")
        print(f"  Mixed Precision: {self.training_metrics.get('mixed_precision', 'N/A')}")
        
        print(f"\nCompute Metrics:")
        if self.compute_metrics["training_time_seconds"] > 0:
            hours = self.compute_metrics["training_time_seconds"] / 3600
            print(f"  Training Time: {hours:.2f} hours ({self.compute_metrics['training_time_seconds']:.0f} seconds)")
        print(f"  GPU Hours: {self.compute_metrics['gpu_hours']:.2f}")
        print(f"  CPU Hours: {self.compute_metrics['cpu_hours']:.2f}")
        print(f"  Total Compute Hours: {self.compute_metrics['total_compute_hours']:.2f}")
        print("=" * 60)

