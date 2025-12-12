import os, SimpleITK as sitk
import nnunet_config_paths as cfg
from deep_psma_utils import resample_to_target
import torch
import numpy as np
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.network_architecture.network_architecture import MultiDecoderUNet
from typing import Tuple, Union, Optional

def run_inference(psma_pet_path, psma_ct_path, fdg_pet_path, fdg_ct_path, fold, 
                  psma_suv_threshold, fdg_suv_threshold, output_type='psma', return_sitk=False):
    """
    Runs early fusion nnU-Net inference and returns predicted TTB mask and baseline mask.
    
    Args:
        psma_pet_path: Path to PSMA PET image
        psma_ct_path: Path to PSMA CT image
        fdg_pet_path: Path to FDG PET image
        fdg_ct_path: Path to FDG CT image
        fold: Validation fold to use
        psma_suv_threshold: SUV threshold for PSMA
        fdg_suv_threshold: SUV threshold for FDG
        output_type: Which output head to use ('psma', 'fdg', or 'both')
        return_sitk: Whether to return SimpleITK images (True) or numpy arrays (False)
    
    Returns:
        If output_type is 'psma' or 'fdg': (predicted_mask, baseline_mask)
        If output_type is 'both': (psma_predicted_mask, fdg_predicted_mask, psma_baseline_mask, fdg_baseline_mask)
    """
    # Prepare input directory with all 4 channels
    tmp_dir = f"inference_input_{output_type}_fold{fold}"
    os.makedirs(tmp_dir, exist_ok=True)
    
    # Create case directory
    case = os.path.basename(psma_pet_path).replace('.nii.gz', '')
    case_dir = os.path.join(tmp_dir, case)
    os.makedirs(case_dir, exist_ok=True)
    
    # Read images
    psma_pet = sitk.ReadImage(psma_pet_path)
    psma_ct = sitk.ReadImage(psma_ct_path)
    fdg_pet = sitk.ReadImage(fdg_pet_path)
    fdg_ct = sitk.ReadImage(fdg_ct_path)
    
    # Rescale and resample
    psma_pet_rs = sitk.Cast(psma_pet / psma_suv_threshold, sitk.sitkFloat32)
    fdg_pet_rs = sitk.Cast(fdg_pet / fdg_suv_threshold, sitk.sitkFloat32)
    psma_ct_rs = resample_to_target(psma_ct, psma_pet, default=-1000)
    fdg_ct_rs = resample_to_target(fdg_ct, psma_pet, default=-1000)
    
    # Save preprocessed images
    sitk.WriteImage(psma_pet_rs, os.path.join(case_dir, f"{case}_0000.nii.gz"))
    sitk.WriteImage(fdg_pet_rs, os.path.join(case_dir, f"{case}_0001.nii.gz"))
    sitk.WriteImage(psma_ct_rs, os.path.join(case_dir, f"{case}_0002.nii.gz"))
    sitk.WriteImage(fdg_ct_rs, os.path.join(case_dir, f"{case}_0003.nii.gz"))
    
    # Set up predictor
    task_id = cfg.dataset_dictionary['EarlyFusion']
    model_folder = os.path.join(cfg.nn_results_dir, f"Dataset{task_id}_EarlyFusion/nnUNetTrainer__nnUNetPlans__3d_fullres")
    
    # Use custom predictor to access both decoder outputs
    if output_type == 'both':
        # Initialize a predictor
        predictor = MultiHeadPredictor(model_folder, folds=fold)
        
        # Run prediction for both heads
        psma_pred, fdg_pred = predictor.predict_case_with_dual_output(
            data_dir=tmp_dir,
            output_dir="predictions_earlyfusion_fold{fold}",
            case_id=case,
            return_both=True
        )
        
        # Compute baseline threshold masks
        psma_bs = psma_pet / psma_suv_threshold >= 1.0
        fdg_bs = fdg_pet / fdg_suv_threshold >= 1.0
        
        psma_bs_img = sitk.Cast(psma_bs, sitk.sitkUInt8)
        fdg_bs_img = sitk.Cast(fdg_bs, sitk.sitkUInt8)
        
        if return_sitk:
            return psma_pred, fdg_pred, psma_bs_img, fdg_bs_img
        
        return (
            sitk.GetArrayFromImage(psma_pred), 
            sitk.GetArrayFromImage(fdg_pred),
            sitk.GetArrayFromImage(psma_bs_img), 
            sitk.GetArrayFromImage(fdg_bs_img)
        )
    else:
        # Use standard nnUNet predictor for single output
        output_dir = f"predictions_earlyfusion_fold{fold}"
        os.makedirs(output_dir, exist_ok=True)
        
        command = f"nnUNetv2_predict -d {task_id} -i {tmp_dir} -o {output_dir} -f {fold}"
        os.system(command)
        
        # Load prediction
        pred = sitk.ReadImage(f"{output_dir}/{case}.nii.gz")
        
        # Compute baseline threshold mask based on requested output type
        if output_type == 'psma':
            bs = psma_pet / psma_suv_threshold >= 1.0
        else:
            bs = fdg_pet / fdg_suv_threshold >= 1.0
        
        bs_img = sitk.Cast(bs, sitk.sitkUInt8)
        
        if return_sitk:
            return pred, bs_img
        
        return sitk.GetArrayFromImage(pred), sitk.GetArrayFromImage(bs_img)


class MultiHeadPredictor(nnUNetPredictor):
    """Extension of nnUNetPredictor to support dual-head output from MultiDecoderUNet"""
    
    def predict_case_with_dual_output(self, data_dir: str, output_dir: str, case_id: str, 
                                     return_both: bool = True) -> Tuple[sitk.Image, Optional[sitk.Image]]:
        """
        Run prediction using the MultiDecoderUNet and get outputs from both decoder heads
        
        Args:
            data_dir: Directory containing the input case data
            output_dir: Directory to save prediction results
            case_id: Case identifier
            return_both: Whether to return outputs from both heads (if False, returns only PSMA)
            
        Returns:
            Tuple of (PSMA prediction, FDG prediction) as SimpleITK images
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Load input data (4 channels)
        input_files = [
            os.path.join(data_dir, case_id, f"{case_id}_0000.nii.gz"),
            os.path.join(data_dir, case_id, f"{case_id}_0001.nii.gz"),
            os.path.join(data_dir, case_id, f"{case_id}_0002.nii.gz"),
            os.path.join(data_dir, case_id, f"{case_id}_0003.nii.gz")
        ]
        
        # Run prediction but don't save output
        data, data_properties = self.preprocess_input_data(input_files)
        
        # Make sure network is in eval mode
        self.network.eval()
        
        # Disable gradient computation
        with torch.no_grad():
            # Forward pass through the network
            prediction = self.predict_logits_from_preprocessed_data(data)
            
            # Since we're using a MultiDecoderUNet, prediction should be a list with two elements
            # Each element contains predictions for one segmentation head
            if isinstance(prediction, list) and len(prediction) > 1:
                # Process and save PSMA prediction (first head)
                psma_pred = prediction[0]
                psma_segmentation = self.convert_logits_to_segmentation(psma_pred)
                psma_output = self.postprocess_prediction(psma_segmentation, data_properties)
                sitk.WriteImage(psma_output, os.path.join(output_dir, f"{case_id}_psma.nii.gz"))
                
                # Process and save FDG prediction (second head)
                if return_both:
                    fdg_pred = prediction[1]
                    fdg_segmentation = self.convert_logits_to_segmentation(fdg_pred)
                    fdg_output = self.postprocess_prediction(fdg_segmentation, data_properties)
                    sitk.WriteImage(fdg_output, os.path.join(output_dir, f"{case_id}_fdg.nii.gz"))
                    
                    return psma_output, fdg_output
                
                return psma_output, None
            else:
                # Fall back to standard processing if not dual head output
                segmentation = self.convert_logits_to_segmentation(prediction)
                output = self.postprocess_prediction(segmentation, data_properties)
                sitk.WriteImage(output, os.path.join(output_dir, f"{case_id}.nii.gz"))
                
                return output, None