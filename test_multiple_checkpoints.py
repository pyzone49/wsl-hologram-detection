import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
import mlflow
from src.utils.utils import get_metrics
import logging
import os
import glob
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
    
    print(f"Starting checkpoint testing at: {start_time_str}")
    
    # Get checkpoint directory from config
    if not hasattr(cfg, 'checkpoint_dir') or not cfg.checkpoint_dir:
        print("Please provide checkpoint directory using: checkpoint_dir=/path/to/checkpoints")
        print("Example: python test_multiple_checkpoints.py --config-name=wsl checkpoint_dir=/path/to/checkpoints")
        return
    
    checkpoint_dir = cfg.checkpoint_dir
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory {checkpoint_dir} does not exist!")
        return
    
    # Find all epoch checkpoints (exclude backbone_ and last.ckpt)
    checkpoint_pattern = os.path.join(checkpoint_dir, "epoch=*.ckpt")
    checkpoints = sorted(glob.glob(checkpoint_pattern))
    
    if not checkpoints:
        print(f"No epoch checkpoints found in {checkpoint_dir}")
        return
    
    print(f"Found {len(checkpoints)} checkpoints to test")
    
    # Set up MLflow experiment for checkpoint testing
    experiment_name = f"checkpoint_test_{cfg.paths.split_name}-{start_time_str}"
    mlflow.set_experiment(experiment_name)
    
    # Test each checkpoint
    for checkpoint_path in checkpoints:
        checkpoint_name = os.path.basename(checkpoint_path)
        epoch_num = checkpoint_name.split('=')[1].split('.')[0]
        
        print(f"Testing checkpoint: {checkpoint_name}")
        
        try:
            # Load PyTorch Lightning checkpoint
            lightning_checkpoint = torch.load(checkpoint_path, map_location='cuda')
            
            # Load backbone checkpoint from specified directory
            backbone_dir = "/home/diva/Documents/other/pouliquen.24.icdar/checkpoints/backbones/"
            backbone_checkpoint_path = os.path.join(backbone_dir, f"backbone_epoch_{epoch_num.zfill(2)}.pth")
            
            if not os.path.exists(backbone_checkpoint_path):
                print(f"  Backbone checkpoint not found: {backbone_checkpoint_path}")
                continue
            
            print(f"  Using Lightning checkpoint: {checkpoint_name}")
            print(f"  Using backbone checkpoint: {backbone_checkpoint_path}")
            
            # Extract the model state dict from the lightning checkpoint
            if 'state_dict' in lightning_checkpoint:
                model_state_dict = lightning_checkpoint['state_dict']
                
                # Create model configuration with the backbone checkpoint from specified directory
                model_cfg = OmegaConf.create({
                    "_target_": cfg.model._target_,  # Use the wrapper (SSLGeneral)
                    "model": {
                        "_target_": cfg.model.model._target_,
                        "backbone_name": cfg.model.model.backbone_name,
                        "pretrained": False,  # Don't use pretrained since we're loading from checkpoint
                        "model_path": backbone_checkpoint_path,  # Use backbone from specified directory
                        "num_classes": cfg.model.model.get("num_classes", -1)
                    },
                    "method": cfg.model.get("method", 3),
                    "accelerator": "cuda"  # Use GPU
                })
                
                with mlflow.start_run():
                    mlflow.set_tag("mlflow.runName", f"checkpoint_epoch_{epoch_num}_{cfg.paths.split_name}")
                    mlflow.log_param("lightning_checkpoint_path", checkpoint_path)
                    mlflow.log_param("backbone_checkpoint_path", backbone_checkpoint_path)
                    mlflow.log_param("epoch", epoch_num)
                    mlflow.log_param("split_name", cfg.paths.split_name)
                    
                    # Test on all test datasets
                    total_metrics = []
                    for test_d in cfg.data.test:
                        seed_everything(cfg.seed, workers=True)
                        
                        try:
                            # Instantiate model with backbone checkpoint
                            model = instantiate(model_cfg)
                            
                            # Instantiate decision method
                            decision = instantiate(cfg.decision)
                            
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
                            metrics = get_metrics(data_test, model, decision)
                            total_metrics.append(metrics)
                            
                            # Log metrics with test dataset prefix
                            mlflow.log_metrics({f"{test_d}_{k}": metrics[k] for k in metrics})
                            
                            print(f"  {test_d}: {metrics}")
                            
                        except Exception as e:
                            print(f"  Error testing {test_d} with {checkpoint_name}: {str(e)}")
                            continue
                        break
                    # Log summary
                    if total_metrics:
                        summary_text = f"Epoch {epoch_num} results: {total_metrics}"
                        mlflow.log_text(summary_text, "results.txt")
                    
                    
            else:
                print(f"  No state_dict found in {checkpoint_name}")
                
        except Exception as e:
            print(f"  Error loading checkpoint {checkpoint_name}: {str(e)}")
            continue
    
    print("Checkpoint testing completed!")

if __name__ == "__main__":
    main() 