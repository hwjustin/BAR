import numpy as np
import os
import torch
from datetime import datetime
from tqdm import tqdm
import webdataset as wds

from models.brain import BrainNetwork, BrainDiffusionPriorOld

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# We only focus on subj 1 for right now
subj = 1
num_voxels = 15724
print("subj",subj,"num_voxels",num_voxels)


# Dataset Setup
voxels = np.load('fake_concept/positive_voxels.npy')  # Load the voxel data directly
voxels = torch.from_numpy(voxels)  # Convert to torch tensor
print("Loaded voxels shape:", voxels.shape)

# CLS model
out_dim = 768
voxel2clip_kwargs = dict(in_dim=num_voxels,out_dim=out_dim)
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

outdir = f'train_logs/prior_1x768_final_subj01_bimixco_softclip_byol'
ckpt_path = os.path.join(outdir, f'last.pth')

print("ckpt_path",ckpt_path)
checkpoint = torch.load(ckpt_path, map_location=device)
state_dict = checkpoint['model_state_dict']
print("EPOCH: ",checkpoint['epoch'])
diffusion_prior_cls.load_state_dict(state_dict,strict=False)
diffusion_prior_cls.eval().to(device)
pass

print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

retrieve = True
plotting = False
saving = True
verbose = False
imsize = 512

all_brain_recons = None
ind_include = np.arange(len(voxels))
all_cls_embeddings = []

# Process all voxels in batches
batch_size = 1
for i in tqdm(range(0, len(voxels), batch_size)):
    batch_voxels = voxels[i:i+batch_size].to(device)
    batch_voxels = torch.mean(batch_voxels,axis=1).to(device)
    
    with torch.no_grad():
        _, cls_embeddings = diffusion_prior_cls.voxel2clip(batch_voxels.float())
        all_cls_embeddings.append(cls_embeddings.cpu())

print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

all_cls_embeddings = torch.cat(all_cls_embeddings, dim=0)
torch.save(all_cls_embeddings, 'cls_embeddings.pt')
print(f"CLS embeddings saved to cls_embeddings.pt with shape: {all_cls_embeddings.shape}")