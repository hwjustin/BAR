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
val_url = f"webdataset_avg_split/test/test_subj0{subj}_" + "{0..1}.tar"
meta_url = f"webdataset_avg_split/metadata_subj0{subj}.json"
num_train = 8559 + 300
num_val = 982
batch_size = val_batch_size = 1
voxels_key = 'nsdgeneral.npy' # 1d inputs

val_data = wds.WebDataset(val_url, resampled=False)\
    .decode("torch")\
    .rename(images="jpg;png", voxels=voxels_key, trial="trial.npy", coco="coco73k.npy", reps="num_uniques.npy")\
    .to_tuple("voxels", "images", "coco")\
    .batched(val_batch_size, partial=False)

val_dl = torch.utils.data.DataLoader(val_data, batch_size=None, shuffle=False)

# check that your data loader is working
for val_i, (voxel, img_input, coco) in enumerate(val_dl):
    print("idx",val_i)
    print("voxel.shape",voxel.shape)
    print("img_input.shape",img_input.shape)
    break

# Generate 10 images and saved them, print their coco value as well
# for val_i, (voxel, img_input, coco) in enumerate(val_dl):
#     print("idx", val_i)
#     print('coco', coco)
    
#     save_path = f"generated_images/sample_{val_i}.png"
#     os.makedirs("generated_images", exist_ok=True)
    
#     # Convert tensor to PIL image and save
#     img_np = img_input[0].permute(1,2,0).cpu().numpy()
#     img_np = (img_np * 255).astype(np.uint8)
#     img = Image.fromarray(img_np)
#     img.save(save_path)
    
#     if val_i >= 9:  # Generate 10 images (0-9)
#         break


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
ind_include = np.arange(num_val)
all_cls_embeddings = []

for val_i, (voxel, img, coco) in enumerate(tqdm(val_dl,total=len(ind_include))):
    if val_i<np.min(ind_include):
        continue
    voxel = torch.mean(voxel,axis=1).to(device)
 
    with torch.no_grad():
        n_samples_save = 1
        brain_recons = None
       

        voxel=voxel[:n_samples_save]
        # image=image[:n_samples_save]

        generator = torch.Generator(device=device)
        generator.manual_seed(42)

        _, cls_embeddings = diffusion_prior_cls.voxel2clip(voxel.to(device).float())

        # Append the embeddings to the list
        all_cls_embeddings.append(cls_embeddings.cpu())  # Move to CPU to avoid memory issues

        # grid, brain_recons, laion_best_picks, recon_img = utils.reconstruction(
        #     img, voxel,
        #     clip_extractor,
        #     voxel2clip_cls = diffusion_prior_cls.voxel2clip,
        #     diffusion_priors = diffusion_priors,
        #     text_token = None,
        #     n_samples_save = batch_size,-
        #     recons_per_sample = 0,
        #     seed = seed,
        #     retrieve = retrieve,
        #     plotting = plotting,
        #     verbose = verbose,
        #     num_retrieved=16,
        # )
            
        # if plotting:
        #     plt.show()
        #     # grid.savefig(f'evals/{model_name}_{val_i}.png')
        #     # plt.close()
            
        # brain_recons = brain_recons[laion_best_picks.astype(np.int8)]

        # if all_brain_recons is None:
        #     all_brain_recons = brain_recons
        #     all_images = img
        # else:
        #     all_brain_recons = torch.vstack((all_brain_recons,brain_recons))
        #     all_images = torch.vstack((all_images,img))
            
    if val_i>=np.max(ind_include):
        break

# all_brain_recons = all_brain_recons.view(-1,3,imsize,imsize)
print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

all_cls_embeddings = torch.cat(all_cls_embeddings, dim=0)
torch.save(all_cls_embeddings, 'cls_embeddings.pt')
print(f"CLS embeddings saved to cls_embeddings.pt with shape: {all_cls_embeddings.shape}")