import os
import json
import numpy as np
import argparse
import torch
from models.umbrae.model import BrainX, BrainXS

parser = argparse.ArgumentParser()
parser.add_argument('--brainx_path', default='train_logs/training_demo/best.pth',
                    help='path to the trained brain encoder model')
parser.add_argument('--voxel_path', type=str, required=True,
                    help='path to the voxel data (.npy file)')
parser.add_argument('--fmri_encoder', type=str, default='brainx',
                    help='type of brainnet', choices=['brainx', 'brainxs'])
parser.add_argument('--use_norm', type=bool, default=False,
                    help='whether to use norm layer in the model')
parser.add_argument('--use_token', type=bool, default=False,
                    help='whether to use learnable token in the model')
parser.add_argument('--feat_dim', type=int, default=1024,
                    help='output dimension of the fmri encoder', choices=[1024, 4096])
parser.add_argument('--save_path', type=str, default='concept',
                    help='path to save features')
parser.add_argument('--subj', type=int, required=True,
                    help='subject number', choices=[1, 2, 5, 7])
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

# Create global variables without the args prefix
for attribute_name in vars(args).keys():
    globals()[attribute_name] = getattr(args, attribute_name)

# Set up device and seed
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

# Create save directory
os.makedirs(save_path, exist_ok=True)

# Save config
args_dict = vars(args)
with open(os.path.join(save_path, 'config.json'), 'w') as file:
    json.dump(args_dict, file, indent=4)

# Load voxel data
print('Loading voxel data...')
voxels = np.load(voxel_path)
voxels = torch.from_numpy(voxels).float()
voxels = torch.mean(voxels, axis=1)  # Take mean across repetitions

# Initialize model
voxels_per_subj = {1: 15724, 2: 14278, 3: 15226, 4: 13153, 5: 13039, 6: 17907, 7: 12682, 8: 14386}
num_voxels = voxels_per_subj.get(subj)

kwargs = {'hidden_dim': 1024, 'out_dim': feat_dim, 'num_latents': 256, 
          'use_norm': use_norm, 'use_token': use_token}

if fmri_encoder == 'brainx':
    voxel2emb = BrainX(**kwargs)
elif fmri_encoder == 'brainxs':
    voxel2emb = BrainXS(in_dim=num_voxels, hidden_dim=1024, 
                        out_dim=feat_dim, num_latents=256)
else:
    raise ValueError("The fmri encoder is not implemented.")

voxel2emb.to(device)

# Load checkpoint
print('Loading checkpoint...')
checkpoint = torch.load(brainx_path, map_location='cpu', weights_only=False)
voxel2emb.load_state_dict(checkpoint['model_state_dict'], strict=False)
voxel2emb.eval()

# Generate embeddings
print('Generating embeddings...')
embeddings = []
batch_size = 1  # You can adjust this based on your GPU memory

with torch.no_grad():
    for i in range(0, len(voxels), batch_size):
        batch = voxels[i:i + batch_size].to(device)
        with torch.cuda.amp.autocast():
            if fmri_encoder == 'brainx':
                emb = voxel2emb(batch, modal=f'fmri{subj}')
            else:
                emb = voxel2emb(batch)
            embeddings.append(emb.cpu())

# Concatenate all embeddings
embeddings = torch.cat(embeddings, dim=0)

# Save embeddings
print('Saving embeddings...')
save_file = os.path.join(save_path, f'features_sub{subj}_dim{feat_dim}.pt')
torch.save(embeddings, save_file)

print(f'Features saved to {save_file}')
print(f'Feature shape: {embeddings.shape}')
