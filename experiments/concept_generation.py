import os
import random
import pandas as pd
import json
import webdataset as wds
import torch
import numpy as np
import argparse
from coco.PythonAPI.pycocotools.coco import COCO

def generate_concept_samples(csv_path, annotations_dir, webdataset_dir, output_base_dir, categories, subj, sample_size=200, seed=42):
    
    random.seed(seed)
    
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file '{csv_path}' not found.")
        return

    all_url = f"{webdataset_dir}/train/train_subj0{subj}_" + "{0..17}.tar"

    dataset = wds.WebDataset(all_url, resampled=False)\
        .decode("torch")\
        .rename(coco="coco73k.npy", voxels="nsdgeneral.npy")\
        .to_tuple("coco", "voxels")

    valid_nsd_ids = {}  
    for batch in dataset:
        nsd_id = batch[0].item()
        voxels = batch[1]
        valid_nsd_ids[nsd_id] = voxels

    df = df[df['nsdId'].isin(valid_nsd_ids.keys())].copy()
    
    coco_instances = {}
    categories_dict = {}
    
    splits = df['cocoSplit'].unique()
    
    for split in splits:
        ann_file = os.path.join(annotations_dir, f'instances_{split}.json')
        if not os.path.isfile(ann_file):
            print(f"Error: Annotation file '{ann_file}' not found.")
            return
        coco_instances[split] = COCO(ann_file)
        cats = coco_instances[split].loadCats(coco_instances[split].getCatIds())
        categories_dict[split] = {cat['name']: cat['id'] for cat in cats}
    
    for concept in categories:
        target_cat_id = None
        for split in splits:
            if concept in categories_dict[split]:
                target_cat_id = categories_dict[split][concept]
                break
        
        if target_cat_id is None:
            print(f"Error: Concept '{concept}' not found in any COCO split.")
            continue
        
        df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        
        positive_samples = []
        negative_samples = []
        positive_voxels = []
        negative_voxels = []
        
        for idx, row in df_shuffled.iterrows():
            if len(positive_samples) >= sample_size and len(negative_samples) >= sample_size:
                break
                
            coco_id = row['cocoId']
            split = row['cocoSplit']
            nsd_id = row['nsdId']
            
            if split not in coco_instances:
                continue
                
            ann_ids = coco_instances[split].getAnnIds(imgIds=coco_id, catIds=[target_cat_id], iscrowd=None)
            if ann_ids:
                if len(positive_samples) < sample_size:
                    positive_samples.append({
                        'cocoId': coco_id,
                        'cocoSplit': split,
                        'nsdId': nsd_id,
                        'concept': concept,
                        'label': 1
                    })
                    positive_voxels.append(valid_nsd_ids[nsd_id])
            else:
                if len(negative_samples) < sample_size:
                    negative_samples.append({
                        'cocoId': coco_id,
                        'cocoSplit': split,
                        'nsdId': nsd_id,
                        'concept': concept,
                        'label': 0
                    })
                    negative_voxels.append(valid_nsd_ids[nsd_id])
        
        if len(positive_samples) < sample_size or len(negative_samples) < sample_size:
            print(f"Error: Not enough samples to generate the desired number of positive and negative examples for '{concept}'.")
            continue
        
        combined_samples = positive_samples + negative_samples
        output_df = pd.DataFrame(combined_samples)

        os.makedirs(os.path.join(output_base_dir, 'csv'), exist_ok=True)
        os.makedirs(os.path.join(output_base_dir, 'voxel'), exist_ok=True)
        
        output_csv_path = os.path.join(output_base_dir, 'csv', f'{concept}_samples.csv')
        output_pos_voxels_path = os.path.join(output_base_dir, 'voxel', f'{concept}_positive_voxels.npy')
        output_neg_voxels_path = os.path.join(output_base_dir, 'voxel', f'{concept}_negative_voxels.npy')
        
        output_df.to_csv(output_csv_path, index=False)
        
        positive_voxels = np.stack(positive_voxels)
        negative_voxels = np.stack(negative_voxels)
        np.save(output_pos_voxels_path, positive_voxels)
        np.save(output_neg_voxels_path, negative_voxels)
        
        print(f"Successfully generated {sample_size} positive and {sample_size} negative samples for '{concept}'.")
        print(f"Output CSV saved to '{output_csv_path}'")
        print(f"Positive voxels saved to '{output_pos_voxels_path}' with shape {positive_voxels.shape}")
        print(f"Negative voxels saved to '{output_neg_voxels_path}' with shape {negative_voxels.shape}")

if __name__ == "__main__":
    CSV_PATH = './data/nsd_stim_info_merged.csv'
    ANNOTATIONS_DIR = './coco/annotations'
    WEBDATASET_DIR = './data/webdataset_avg_split'

    parser = argparse.ArgumentParser(description="Train and evaluate concept classifier for Bar dataset.")
    parser.add_argument("--random_seed", type=int, default=57, 
                       help="Random seed for reproducibility.")
    parser.add_argument("--sample_size", type=int, default=200,
                       help="Sample size for each concept")
    parser.add_argument("--subj", type=int, default=1,
                       help="Subject number", choices=[1, 2, 5, 7])
    args = parser.parse_args()
    
    CATEGORIES_SUBJ01 = [
        'person', 'chair', 'car', 'dining table', 'cup', 'bottle', 'bowl', 'handbag', 'truck', 'bench',
        'book', 'backpack', 'sink', 'clock', 'dog', 'sports ball', 'cat', 'potted plant', 'cell phone',
        'surfboard', 'knife', 'tie', 'skis', 'bus', 'traffic light', 'tv', 'bed', 'train', 'umbrella',
        'toilet', 'tennis racket', 'spoon', 'couch', 'bird', 'skateboard', 'airplane', 'motorcycle',
        'boat', 'vase', 'bicycle', 'fork', 'pizza', 'oven', 'giraffe', 'laptop'
    ]
    
    CATEGORIES_SUBJ02 = [
        'person', 'car', 'chair', 'dining table', 'cup', 'bottle', 'bowl', 'handbag', 'bench', 'truck',
        'backpack', 'book', 'clock', 'sink', 'traffic light', 'cell phone', 'cat', 'sports ball', 'dog',
        'potted plant', 'surfboard', 'bus', 'train', 'umbrella', 'knife', 'toilet', 'tv', 'tennis racket',
        'couch', 'bed', 'airplane', 'bicycle', 'skis', 'spoon', 'fork', 'vase', 'skateboard', 'bird',
        'motorcycle', 'laptop', 'boat', 'pizza', 'horse', 'oven', 'tie'
    ]
    
    CATEGORIES_SUBJ05 = [
        'person', 'car', 'chair', 'dining table', 'cup', 'bottle', 'bowl', 'truck', 'handbag', 'bench',
        'backpack', 'clock', 'book', 'sink', 'dog', 'sports ball', 'cell phone', 'surfboard', 'knife', 'cat',
        'toilet', 'traffic light', 'umbrella', 'tv', 'potted plant', 'train', 'bed', 'couch', 'bird', 'spoon',
        'bus', 'fork', 'motorcycle', 'skateboard', 'pizza', 'tennis racket', 'vase', 'laptop', 'airplane',
        'bicycle', 'cake', 'tie', 'boat', 'giraffe', 'skis', 'oven'
    ]
    
    CATEGORIES_SUBJ07 = [
        'person', 'chair', 'car', 'dining table', 'cup', 'bottle', 'bowl', 'handbag', 'truck', 'bench',
        'clock', 'backpack', 'sink', 'book', 'sports ball', 'dog', 'traffic light', 'toilet', 'knife',
        'potted plant', 'surfboard', 'train', 'bus', 'cell phone', 'airplane', 'cat', 'tv', 'vase', 'bird',
        'skis', 'tennis racket', 'pizza', 'umbrella', 'tie', 'couch', 'spoon', 'bicycle', 'fork', 'bed',
        'horse', 'laptop', 'skateboard', 'motorcycle', 'boat', 'oven', 'giraffe'
    ]

    categories_map = {
        1: CATEGORIES_SUBJ01,
        2: CATEGORIES_SUBJ02,
        5: CATEGORIES_SUBJ05,
        7: CATEGORIES_SUBJ07
    }


    generate_concept_samples(
        csv_path=CSV_PATH,
        annotations_dir=ANNOTATIONS_DIR,
        webdataset_dir=WEBDATASET_DIR,
        output_base_dir=f'./concept_subj0{args.subj}',
        categories=categories_map[args.subj],
        subj=args.subj,
        sample_size=args.sample_size,
        seed=args.random_seed
    )
