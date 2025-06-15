import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
import mlflow
from src.utils.utils import get_metrics
import logging
import os
from lightning import seed_everything
import torch
from datetime import datetime

# A logger for this file
log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed, workers=True)
    
    # Record start time
    start_time = datetime.now()
    start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"Starting ImageNet baseline testing at: {start_time_str}")
    print("Testing with original ImageNet pretrained weights...")
    
    # Check if we should perform calibration first
    perform_calibration = getattr(cfg, 'calibrate', True)  # Default to True
    
    # Set up MLflow experiment for ImageNet baseline testing
    experiment_name = f"imagenet_baseline_{cfg.paths.split_name}-{start_time_str}"
    mlflow.set_experiment(experiment_name)
    
    # Create model configuration with ImageNet pretrained weights
    model_cfg = OmegaConf.create({
        "_target_": cfg.model._target_,  # Use the wrapper (SSLGeneral)
        "model": {
            "_target_": cfg.model.model._target_,
            "backbone_name": cfg.model.model.backbone_name,
            "pretrained": True,  # Use ImageNet pretrained weights
            "num_classes": cfg.model.model.get("num_classes", -1)
        },
        "method": cfg.model.get("method", 3),
        "accelerator": "cuda"  # Use GPU
    })
    
    # Instantiate model once for calibration and testing
    model = instantiate(model_cfg)
    
    # Instantiate decision method
    decision = instantiate(cfg.decision)
    original_threshold = decision.th
    
    # CALIBRATION STEP
    if perform_calibration:
        print("=" * 50)
        print("CALIBRATION PHASE")
        print("=" * 50)
        
        try:
            # Get validation data for calibration
            data_val = instantiate(cfg.data.train)  # Using train data for calibration like in calibration.py
            
            print(f"Calibrating decision threshold using validation data...")
            print(f"Original threshold: {original_threshold}")
            
            # Perform calibration (tune threshold)
            metrics, optimized_th = decision.tune(data_val, model)
            decision.th = float(optimized_th)
            
            print(f"Calibration completed!")
            print(f"Best F-score found: {metrics['fscore']:.4f}")
            print(f"Optimized threshold: {optimized_th:.6f}")
            print(f"Threshold change: {original_threshold:.6f} → {optimized_th:.6f}")
            
        except Exception as e:
            print(f"Calibration failed: {str(e)}")
            print("Continuing with original threshold...")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
    else:
        print(f"Skipping calibration, using threshold: {original_threshold}")
    
    print("=" * 50)
    print("TESTING PHASE")
    print("=" * 50)
    
    with mlflow.start_run():
        mlflow.set_tag("mlflow.runName", f"imagenet_baseline_{cfg.paths.split_name}")
        mlflow.log_param("backbone_type", "imagenet_pretrained")
        mlflow.log_param("backbone_name", cfg.model.model.backbone_name)
        mlflow.log_param("split_name", cfg.paths.split_name)
        mlflow.log_param("pretrained", True)
        mlflow.log_param("checkpoint_used", "None - ImageNet baseline")
        mlflow.log_param("calibration_performed", perform_calibration)
        mlflow.log_param("original_threshold", original_threshold)
        mlflow.log_param("final_threshold", decision.th)
        
        # Log calibration results if performed
        if perform_calibration and 'metrics' in locals():
            mlflow.log_metrics({f"calibration_{k}": v for k, v in metrics.items()})
        
        # Test on all test datasets
        total_metrics = []
        for test_d in cfg.data.test:
            seed_everything(cfg.seed, workers=True)
            
            try:
                print(f"Testing on dataset: {test_d}")
                
                # Instantiate test data
                data_test = instantiate(cfg.data.test[test_d])
                
                # Log dataset information to MLflow
                mlflow.log_param(f"{test_d}_dataset_name", test_d)
                mlflow.log_param(f"{test_d}_dataset_type", type(data_test).__name__)
                
                # Log detailed fraud information from labels_dict
                if hasattr(data_test, 'labels_dict'):
                    fraud_count = sum(len(videos) for key, videos in data_test.labels_dict.items() if 'fraud' in key)
                    origin_count = len(data_test.labels_dict.get('origins', {}))
                    
                    mlflow.log_param(f"{test_d}_num_frauds", fraud_count)
                    mlflow.log_param(f"{test_d}_num_originals", origin_count)
                    
                    # Log specific fraud type counts
                    for fraud_type, videos in data_test.labels_dict.items():
                        if 'fraud' in fraud_type:
                            clean_name = fraud_type.replace('fraud/', '')
                            mlflow.log_param(f"{test_d}_fraud_{clean_name}_count", len(videos))
                
                # Log dataset composition
                if hasattr(data_test, 'input_dir'):
                    mlflow.log_param(f"{test_d}_input_dir", data_test.input_dir)
                
                # Log total dataset size (number of videos/documents)
                if hasattr(data_test, '__len__'):
                    mlflow.log_param(f"{test_d}_num_videos", len(data_test))
                
                # Log input size and transform info
                if hasattr(data_test, 'input_size'):
                    mlflow.log_param(f"{test_d}_input_size", data_test.input_size)
                if hasattr(data_test, 'applytransform'):
                    mlflow.log_param(f"{test_d}_apply_transform", data_test.applytransform)
                
                # Create detailed description
                description_parts = []
                if hasattr(data_test, 'fraud_names') and data_test.fraud_names:
                    description_parts.append(f"Fraud types: {', '.join([fn.replace('fraud/', '') for fn in data_test.fraud_names])}")
                if hasattr(data_test, 'labels_dict'):
                    if 'origins' in data_test.labels_dict:
                        description_parts.append(f"Contains {len(data_test.labels_dict['origins'])} original documents")
                    else:
                        description_parts.append("Fraud-only dataset (no originals)")
                
                dataset_description = '; '.join(description_parts) if description_parts else f"{test_d} dataset"
                mlflow.log_param(f"{test_d}_description", dataset_description)
                
                # Get metrics
                print(f"  Computing metrics for {test_d}...")
                metrics = get_metrics(data_test, model, decision)
                total_metrics.append(metrics)
                
                # Log metrics with test dataset prefix
                mlflow.log_metrics({f"{test_d}_{k}": metrics[k] for k in metrics})
                
                print(f"  {test_d} results: {metrics}")
                
            except Exception as e:
                print(f"  Error testing {test_d} with ImageNet weights: {str(e)}")
                import traceback
                print(f"  Full traceback: {traceback.format_exc()}")
                continue
        
        # Log summary
        if total_metrics:
            summary_text = f"ImageNet baseline results: {total_metrics}"
            mlflow.log_text(summary_text, "results.txt")
            
            # Calculate and log average metrics if multiple datasets
            if len(total_metrics) > 1:
                avg_metrics = {}
                # Get all possible keys from all metric dictionaries
                all_keys = set()
                for metrics in total_metrics:
                    all_keys.update(metrics.keys())
                
                # Calculate average for each key that exists in all metrics
                for key in all_keys:
                    values = []
                    for metrics in total_metrics:
                        if key in metrics and isinstance(metrics[key], (int, float)):
                            values.append(metrics[key])
                    
                    if len(values) == len(total_metrics):  # Key exists in all metrics
                        avg_metrics[f"avg_{key}"] = sum(values) / len(values)
                    elif len(values) > 0:  # Key exists in some metrics
                        avg_metrics[f"partial_avg_{key}"] = sum(values) / len(values)
                        mlflow.log_param(f"{key}_available_in", f"{len(values)}/{len(total_metrics)}_datasets")
                
                if avg_metrics:
                    mlflow.log_metrics(avg_metrics)
                    print(f"Average metrics across {len(total_metrics)} datasets: {avg_metrics}")
                else:
                    print("No common numeric metrics found for averaging")
        else:
            print("No metrics were computed successfully")
    
    print("ImageNet baseline testing completed!")

if __name__ == "__main__":
    main() 