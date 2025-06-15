import torch
from captum.attr import IntegratedGradients
from torchvision import transforms
from PIL import Image
import sys
import os
import glob
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from src.utils.utils import mlruntodict

# Import your model classes
from src.models.generic_model import GenericBackbone
from src.models.ssl_general import SSLGeneral

@hydra.main(version_base=None, config_path="conf")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    
    # ---- CONFIG FROM HYDRA ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ---- GET CALIBRATED PARAMETERS (same as test.py) ----
    task_name_full = HydraConfig.get().runtime.choices.decision + "_" + cfg.task_name
    if cfg.get("training", ""):
        task_name_full += cfg.training.trainer.run_name
    cfg.tuner.experiment_name = task_name_full
    tuner = instantiate(cfg.tuner)

    run = tuner.getBestRun()
    
    if run is None:
        print("No calibration run found! Please run calibration.py first.")
        return

    params = mlruntodict(run.data.params)
    params = DictConfig(params)
    
    print(f"Using calibrated parameters from run: {run.info.run_name}")
    print(f"Calibrated threshold: {params.decision.th}")
    
    # ---- MODEL SETUP ----
    # Use the same model instantiation as in calibration/test
    model = instantiate(params.model)
    
    # Load checkpoint if specified
    checkpoint_path = getattr(cfg, 'checkpoint_path', None)
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=device)
        if "state_dict" in state:
            state = state["state_dict"]
        # Load state dict into the underlying backbone model
        model.model.load_state_dict(state)
        print("Checkpoint loaded successfully!")

    # ---- DECISION SETUP ----
    decision = instantiate(params.decision)
    threshold = decision.th
    print(f"Using decision threshold: {threshold}")

    # ---- INPUT PREP ----
    # Get parameters from config with defaults
    input_size = getattr(cfg, 'input_size', 224)
    image_dir = getattr(cfg, 'image_dir', None)
    image_path = getattr(cfg, 'image_path', None)
    
    # Determine input source
    if image_dir and os.path.isdir(image_dir):
        # Load all images from directory
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(image_dir, ext)))
            image_files.extend(glob.glob(os.path.join(image_dir, ext.upper())))
        
        image_files.sort()  # Sort for consistent ordering
        
        if not image_files:
            print(f"No images found in directory: {image_dir}")
            return
            
        print(f"Found {len(image_files)} images in directory: {image_dir}")
        
    elif image_path and os.path.exists(image_path):
        # Single image fallback
        image_files = [image_path]
        print(f"Using single image: {image_path}")
        
    else:
        print("Please specify either 'image_dir' (directory of images) or 'image_path' (single image)")
        return
    
    IMAGENET_NORMALIZE = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    preprocess = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_NORMALIZE["mean"], std=IMAGENET_NORMALIZE["std"]),
    ])

    # Load and preprocess all images
    input_tensors = []
    for img_file in image_files:
        try:
            img = Image.open(img_file).convert('RGB')
            input_tensor = preprocess(img).to(device)
            input_tensors.append(input_tensor)
        except Exception as e:
            print(f"Error loading image {img_file}: {e}")
            continue
    
    if not input_tensors:
        print("No valid images could be loaded!")
        return
        
    print(f"Successfully loaded {len(input_tensors)} images")

    # ---- GET MODEL PREDICTION (using decision.process_frame_by_frame) ----
    print(f"Processing {len(input_tensors)} frames using decision.process_frame_by_frame...")
    
    is_fraud, last_frame_idx = decision.process_frame_by_frame(input_tensors, model)
    
    print(f"Processed {last_frame_idx + 1} frames")
    print(f"Decision result: is_fraud = {is_fraud}")
    
    # Convert to consistent format (True = Fraud, False = Genuine)
    if hasattr(decision, '__class__') and decision.__class__.__name__ == 'Cumulative':
        # Cumulative returns False for fraud, True for genuine
        is_genuine = is_fraud
        is_fraud = not is_fraud
    else:
        # NDecision returns True for fraud, False for genuine
        is_genuine = not is_fraud
    
    prediction = "Fraud" if is_fraud else "Genuine"
    print(f"Binary decision: {prediction}")

    # Get the final score for display (optional, for backward compatibility)
    model.reset()
    model_scores = []
    for i, input_tensor in enumerate(input_tensors):
        score = model.apply(input_tensor)
        if score is not None:
            model_scores.append(score)
        if i >= last_frame_idx:  # Only process up to the decision point
            break
    
    if model_scores:
        final_score = model_scores[-1]
        print(f"Final SSL model score: {final_score:.6f} (at frame {len(model_scores)})")
    else:
        print("No valid scores obtained")

    # ---- INTEGRATED GRADIENTS (using first image only) ----
    print(f"\nPerforming Integrated Gradients analysis on first image: {os.path.basename(image_files[0])}")
    
    # For Integrated Gradients, we need to use the underlying backbone model
    backbone_model = model.model  # Get the underlying backbone
    backbone_model.eval()
    
    def model_forward(x):
        return backbone_model(x)

    ig = IntegratedGradients(model_forward)
    target_class = getattr(cfg, 'target_class', 0)

    # Use first image for Integrated Gradients
    first_image_batched = input_tensors[34].unsqueeze(0)
    attributions, delta = ig.attribute(first_image_batched, 
                                       target=target_class, 
                                       return_convergence_delta=True,
                                       n_steps=50)

    # ---- VISUALIZATION ----
    import matplotlib.pyplot as plt
    import numpy as np

    attr = attributions.squeeze().cpu().detach().numpy()
    if attr.ndim == 3:
        attr = np.transpose(attr, (1, 2, 0))  # CHW to HWC

    # Create subplot for better visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
     
    # Original first image
    first_img = Image.open(image_files[34]).convert('RGB')
    original_img = first_img.resize((input_size, input_size))
    ax1.imshow(original_img)
    
    # Create title with available information
    title_parts = [f'First Image: {os.path.basename(image_files[0])}']
    if 'final_score' in locals():
        title_parts.append(f'SSL Score: {final_score:.6f} (from {len(model_scores)} frames)')
    title_parts.append(f'Prediction: {prediction}')
    
    ax1.set_title('\n'.join(title_parts))
    ax1.axis('off')
    
    # Attribution map
    im = ax2.imshow(attr.sum(axis=2), cmap='hot')
    ax2.set_title(f'Integrated Gradients Attribution\n(Backbone {os.path.basename(checkpoint_path)})')
    ax2.axis('off')
    plt.colorbar(im, ax=ax2)
    
    plt.tight_layout()
    
    # Save or show based on config
    save_path = getattr(cfg, 'save_path', None)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Attribution map saved to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    main()