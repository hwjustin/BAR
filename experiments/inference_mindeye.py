import numpy as np
import os
import torch
from datetime import datetime
import argparse
from models.brain import BrainNetwork, BrainDiffusionPriorOld

# Argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_dir', type=str, default='train_logs',
                    help='directory to load checkpoints from')
parser.add_argument('--out_dim', type=int, default=768,
                    help='output dimension of the voxel2clip model')
parser.add_argument('--subj', type=int, required=True,
                    help='subject number', choices=[1, 2, 5, 7])
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

# Create global variables without the args prefix
for attribute_name in vars(args).keys():
    globals()[attribute_name] = getattr(args, attribute_name)

# Set up device and seed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

# We only focus on the specified subject
voxels_per_subj = {1: 15724, 2: 14278, 5: 13039, 7: 12682}
num_voxels = voxels_per_subj.get(subj)
print("subj", subj, "num_voxels", num_voxels)

# CLS model
voxel2clip_kwargs = dict(in_dim=num_voxels, out_dim=out_dim)
voxel2clip_cls = BrainNetwork(**voxel2clip_kwargs)
voxel2clip_cls.requires_grad_(False)
voxel2clip_cls.eval()

diffusion_prior_cls = BrainDiffusionPriorOld.from_pretrained(
    # kwargs for DiffusionPriorNetwork
    dict(),
    # kwargs for DiffusionNetwork
    dict(
        condition_on_text_encodings=False,
        timesteps=1000,
        voxel2clip=voxel2clip_cls,
    ),
    voxel2clip_path=None,
)

outdir = os.path.join(ckpt_dir, f'prior_1x768_final_subj0{subj}_bimixco_softclip_byol')
ckpt_path = os.path.join(outdir, f'last.pth')

print("ckpt_path", ckpt_path)
checkpoint = torch.load(ckpt_path, map_location=device)
state_dict = checkpoint['model_state_dict']
print("EPOCH: ", checkpoint['epoch'])
diffusion_prior_cls.load_state_dict(state_dict, strict=False)
diffusion_prior_cls.eval().to(device)

print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

# Define directory for voxel data
voxel_dir = f'concept_subj0{subj}/voxel'
feature_dir = os.path.join(f'concept_subj0{subj}', 'feature_Mindeye')
os.makedirs(feature_dir, exist_ok=True)

# Process each voxel file in the directory
for voxel_file in os.listdir(voxel_dir):
    if voxel_file.endswith('.npy'):
        voxel_path = os.path.join(voxel_dir, voxel_file)
        
        # Load voxel data
        print(f'Loading voxel data from {voxel_file}...')
        voxels = np.load(voxel_path)
        voxels = torch.from_numpy(voxels).float()
        voxels = torch.mean(voxels, axis=1)  # Take mean across repetitions

        # Generate embeddings
        print('Generating embeddings...')
        embeddings = []
        batch_size = 1  # You can adjust this based on your GPU memory

        with torch.no_grad():
            for i in range(0, len(voxels), batch_size):
                batch = voxels[i:i + batch_size].to(device)
                _, cls_embeddings = diffusion_prior_cls.voxel2clip(batch.float())
                embeddings.append(cls_embeddings.cpu())

        # Concatenate all embeddings
        embeddings = torch.cat(embeddings, dim=0)

        # Save embeddings with the same name in the feature directory, but with .pt extension
        feature_file = os.path.join(feature_dir, voxel_file.replace('voxels', 'features').replace('.npy', '.pt'))
        print(f'Saving embeddings to {feature_file}...')
        torch.save(embeddings, feature_file)

        print(f'Features saved to {feature_file}')
        print(f'Feature shape: {embeddings.shape}')

print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))