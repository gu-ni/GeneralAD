import os
import argparse
import re
import numpy as np
from tqdm import tqdm
import pandas as pd
import json

import torch
import torch
import torch.nn.functional as F
import gc
from pytorch_lightning import Trainer

from src.general_ad import GeneralADForTest
from src.load_data import prepare_loader_from_json_by_chunk


def parse_args():
    parser = argparse.ArgumentParser(description="Train an Anomaly Detection algorithm")

    # Add arguments
    parser.add_argument("--normal_class", type=str, default=0, help="Normal class for training")
    parser.add_argument("--seed", type=int, default=0, help="Seed for random number generators")
    parser.add_argument("--dataset_name", default="cifar10", choices=['cifar10', 'mvtec-loco-ad', 'mvtec-ad', 'fgvc-aircraft', 'cifar100', 'stanford-cars', 'fmnist', 'catsvdogs', 'view', 'mpdd', 'visa'], help="Name of the dataset")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--log_every_n_steps", type=int, default=5, help="Log frequency")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers for data loading")
    parser.add_argument("--embed_dim", type=int, default=768, help="Embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=2048, help="Hidden dimension")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of heads")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--patch_size", type=int, default=14, help="Size of each patch")
    parser.add_argument("--num_channels", type=int, default=3, help="Number of channels")
    parser.add_argument("--num_patches", type=int, default=256, help="Number of patches")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--lr_decay_factor", type=float, default=0.2, help="Learning rate decay factor for cosine annealing")
    parser.add_argument("--lr_adaptor", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--hf_path", type=str, default='vit_large_patch14_dinov2.lvd142m', help="Huggingface model path")
    parser.add_argument("--milestones", type=str, default="5", help="Scheduler milestones as a comma-separated string")
    parser.add_argument("--gamma", type=float, default=0.2, help="Scheduler gamma value")
    parser.add_argument('--data_dir', type=str, default='data/', help='Data directory where to store/find the dataset.')
    parser.add_argument("--run_type", default="general_ad", choices=['kdad', 'test_data', 'simplenet', 'viz_attn', 'general_ad', 'viz_segmentation'], help="The files that have to be run.")
    parser.add_argument("--model_type", default="ViT", choices=['ViT', 'MLP'], help="The type of model to be trained for KDAD.")
    parser.add_argument("--image_size", type=int, default=336, help="Input size of ViT images")
    parser.add_argument("--layers_to_extract_from", type=str, default="24", help="Layers to extract from as a comma-separated string")
    parser.add_argument("--wd", type=float, default=0.00001, help="Weight decay for the discriminator")
    parser.add_argument("--dsc_layers", type=int, default=1, help="Number of layers for the discriminator")
    parser.add_argument("--dsc_heads", type=int, default=4, help="Number of heads for the discriminator")
    parser.add_argument("--dsc_dropout", type=float, default=0.1, help="Dropout rate for the discriminator")
    parser.add_argument("--noise_std", type=float, default=0.25, help="Standard deviation of the noise to create fake samples for the discriminator")
    parser.add_argument("--dsc_type", default="mlp", choices=['mlp', 'transformer'], help="The type of model you want for the discriminator.")
    parser.add_argument('--no_avg_pooling', action='store_false', help='Set to disable average pooling. Defaults to True.')
    parser.add_argument("--pool_size", type=int, default=3, help="Size of local neighboorhood to aggregate over.")
    parser.add_argument("--num_fake_patches", type=int, default=-1, help="Number of fake patches for the transformer discriminator")
    parser.add_argument('--load_checkpoint', action='store_true', help='Load the model from a checkpoint instead of training from scratch. Defaults to False.')
    parser.add_argument("--checkpoint_dir", type=str, default="lightning_logs/dir", help="The directory in which the model checkpoints are stored, printed after a training run.")
    parser.add_argument("--blob_size_factor", type=float, default=1.0, help="Size of the blob")
    parser.add_argument("--sigma_blob_noise", type=float, default=0.4, help="magnitude of the standard deviation for the probability distribution over the grid for creating the starting patch of the blob.")
    parser.add_argument("--fake_feature_type", type=str, default="random", help="The type of fake featuers to create for general ad.")
    parser.add_argument("--top_k", type=int, default=10, help="number of patches to use to determine if an image is anomalous.")
    parser.add_argument("--smoothing_sigma", type=float, default=16, help="Standard deviation of the smoothing to create the segmentation map.")
    parser.add_argument("--smoothing_radius", type=int, default=18, help="Standard deviation of the smoothing to create the segmentation map.")
    parser.add_argument("--shots", type=int, default=-1, help="number of shots for few-shot setting.")
    parser.add_argument("--val_monitor", default="image_auroc", choices=['image_auroc', 'pixel_auroc'], help="Validate based on image level score or pixel level score.")
    parser.add_argument("--log_pixel_metrics", type=int, default=1, choices=[0, 1], help="If the dataset includes segmentation masks than 1 else 0.")
    parser.add_argument("--wandb_project", type=str, default="continual-general-ad", help="WandB project name")
    
    parser.add_argument("--json_path", type=str, default="5classes_tasks", help="Name of the task JSON file (e.g., 5classes_tasks)")
    parser.add_argument('--score_dir', default='/workspace/MegaInspection/GeneralAD/scores', type=str,)
    parser.add_argument('--scenario', default='scenario_1', type=str)
    parser.add_argument('--case', default='5_classes_tasks', type=str)

    # Parse arguments
    args = parser.parse_args()
    
    return args

if __name__ == '__main__':
    args = parse_args()
    
    model_weight_path = "/workspace/MegaInspection/GeneralAD/outputs"
    args.milestones = [int(x) for x in args.milestones.split(',')]
    args.layers_to_extract_from = [int(x) for x in args.layers_to_extract_from.split(',')]
    
    json_path = os.path.join("/workspace/meta_files", f"{args.json_path}.json")
    with open(json_path, 'r') as f:
        data_dict = json.load(f)
    data_dict = data_dict['test']

    weights_list = os.listdir(os.path.join(model_weight_path, args.scenario, args.case))
    weights_list = sorted(weights_list, key=lambda x: int(x.split(".")[0][4:]))
    last_model = weights_list[-1]
    last_model_num = int(last_model.split(".")[0][4:])
    
    pretrained_path = os.path.join(model_weight_path, 
                                        args.scenario, 
                                        args.case, 
                                        last_model)
    final_score_path = os.path.join(args.score_dir,
                                    args.scenario, 
                                    args.case, 
                                    args.json_path,
                                    f"model{last_model_num}.json")

    args.final_score_path = final_score_path
    model = GeneralADForTest(
            lr=args.lr,
            lr_decay_factor=args.lr_decay_factor,
            hf_path=args.hf_path,
            layers_to_extract_from=args.layers_to_extract_from,
            hidden_dim=args.hidden_dim,
            wd=args.wd,
            epochs=args.epochs,
            noise_std=args.noise_std,
            dsc_layers=args.dsc_layers,
            dsc_heads=args.dsc_heads,
            dsc_dropout=args.dsc_dropout,
            pool_size=args.pool_size,
            image_size=args.image_size,
            num_fake_patches=args.num_fake_patches,
            fake_feature_type=args.fake_feature_type,
            top_k=args.top_k,
            log_pixel_metrics=args.log_pixel_metrics,
            smoothing_sigma=args.smoothing_sigma,
            smoothing_radius=args.smoothing_radius,
            final_score_path = args.final_score_path,
        )

    model.load_state_dict(torch.load(pretrained_path))
    
    os.makedirs(os.path.dirname(final_score_path), exist_ok=True)
    
    for i, (cls_name, samples) in enumerate(data_dict.items()):
        len_samples = len(samples)
        print(f"[{i+1}/{len(data_dict)}] {cls_name}")
        print("length of samples:", len_samples)
        print()
        # if len_samples > 1500:
        #     print(f"Sample size {len_samples} is larger than 1500, passing...")
        #     continue
        
        if os.path.exists(final_score_path):
            with open(final_score_path, 'r') as f:
                cumm_score = json.load(f)
            if cls_name in cumm_score:
                print(f"json_path: {json_path}")
                print(f"Already evaluated {cls_name} class")
                continue
        else:
            cumm_score = {}
        
        sub_data_dict = {}
        sub_data_dict[cls_name] = samples
        
        test_loader = prepare_loader_from_json_by_chunk(
            sub_data_dict,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            train=False,
            zero_shot_category=None,
        )
        
        trainer = Trainer(
            accelerator="gpu",
            devices=1,
            enable_progress_bar=True,
        )
        
        with torch.no_grad():
            results = trainer.test(model, dataloaders=test_loader)
        results = results[0] if isinstance(results, list) else results
        
        cumm_score[cls_name] = results
        
        with open(final_score_path, 'w') as f:
            json.dump(cumm_score, f, indent=4)
        
        del test_loader
        torch.cuda.empty_cache()
        gc.collect()