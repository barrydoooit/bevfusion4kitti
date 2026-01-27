import argparse
from collections import OrderedDict
import torch
import os
from typing import List, Dict



class CheckpointMerger:
    def __init__(self, checkpoint1_path: str, checkpoint2_path: str, output_path: str, exclude_keys: List[str] = None):
        """
        Initialize the CheckpointMerger with paths to the checkpoint files.

        :param checkpoint1_path: Path to the first checkpoint file.
        :param checkpoint2_path: Path to the second checkpoint file.
        :param output_path: Path where the merged checkpoint will be saved.
        :param exclude_keys: List of parameter keys to exclude from the final checkpoint.
        """
        self.checkpoint1_path = checkpoint1_path
        self.checkpoint2_path = checkpoint2_path
        self.output_path = output_path
        self.exclude_keys = exclude_keys if exclude_keys is not None else []

    def load_checkpoint(self, path: str) -> Dict[str, torch.Tensor]:
        """
        Load a PyTorch checkpoint.

        :param path: Path to the checkpoint file.
        :return: The state dictionary from the checkpoint.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        
        return torch.load(path, map_location='cpu')

    def merge_checkpoints(self) -> Dict[str, torch.Tensor]:
        """
        Merge two checkpoints into one, following the first-come-first-use strategy and filtering out unwanted parameters.

        :return: The merged state dictionary.
        """
        # Load the checkpoints
        state_dict1 = self.load_checkpoint(self.checkpoint1_path)["state_dict"]
        state_dict2 = self.load_checkpoint(self.checkpoint2_path)["state_dict"]

        # Initialize the merged state dict with the first checkpoint
        merged_state_dict = state_dict1.copy()

        # Add parameters from the second checkpoint, only if they don't exist in the first
        for key, value in state_dict2.items():
            if all([not key.startswith(exclude_key) for exclude_key in self.exclude_keys]) and key not in merged_state_dict:
                merged_state_dict[key] = value

        # Filter out any keys that need to be excluded from the final checkpoint
        for key in self.exclude_keys:
            if key in merged_state_dict:
                del merged_state_dict[key]

        for key in merged_state_dict.keys():
            if (key == 'pts_middle_encoder.conv_input.0.weight') or (key == 'pts_middle_encoder.conv_out.0.weight') or (key == 'pts_middle_encoder.encoder_layers.encoder_layer3.2.0.weight') or (key == 'pts_middle_encoder.encoder_layers.encoder_layer2.2.0.weight') or (key == 'pts_middle_encoder.encoder_layers.encoder_layer1.2.0.weight') or (('pts_middle_encoder.encoder_layers.encoder_layer' in key) and ('conv' in key)):
                merged_state_dict[key] = torch.transpose(merged_state_dict[key],0,1)
                merged_state_dict[key] = torch.transpose(merged_state_dict[key],1,2)
                merged_state_dict[key] = torch.transpose(merged_state_dict[key],2,3)
                merged_state_dict[key] = torch.transpose(merged_state_dict[key],3,4)
        return merged_state_dict

    def save_checkpoint(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """
        Save the merged checkpoint to the specified output path.

        :param state_dict: The merged state dictionary.
        """
        torch.save(state_dict, self.output_path)
        print(f"Checkpoint successfully saved to {self.output_path}")

    def execute(self) -> None:
        """
        Execute the merging process: load, merge, and save the checkpoint.
        """
        print("Starting the checkpoint merging process...")
        merged_state_dict = OrderedDict()
        merged_state_dict['state_dict'] = self.merge_checkpoints()
        self.save_checkpoint(merged_state_dict)
        print("Checkpoint merging process completed.")

def main():
    parser = argparse.ArgumentParser(description="Merge two checkpoints into one.")
    parser.add_argument('--ckpt1', type=str, required=True, help="Path to the first checkpoint file.")
    parser.add_argument('--ckpt2', type=str, required=True, help="Path to the second checkpoint file.")
    parser.add_argument('--outfile', type=str, required=True, help="Path to save the merged checkpoint.")
    parser.add_argument('--exclude', type=str, nargs='*', default=[], help="Keys to exclude from the final checkpoint.")
    
    args = parser.parse_args()
    
    merger = CheckpointMerger(args.ckpt1, args.ckpt2, args.outfile, args.exclude)
    merger.execute()
    
if __name__ == "__main__":
    # checkpoint1_path = "/workspace/work_dirs/[k01s3]20240812bevfusion_lidar_kitti-3d-ring_voxel0075_aug_q128_box-weight/epoch_20.pth"
    # checkpoint2_path = "/workspace/work_dirs/[k02cs1]bevfusion_cam_kitti-3d-ring_q200_box-weight/epoch_40.pth"
    # output_path = "/workspace/checkpoints/k01s3_k02cs1.pth"
    # exclude_keys = ["fusion_layer"]
    # merger = CheckpointMerger(checkpoint1_path, checkpoint2_path, output_path, exclude_keys)
    # merger.execute()

    main()