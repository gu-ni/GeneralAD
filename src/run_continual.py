import os
import re
import torch
import sys
import json

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from .general_ad import General_AD
from .load_data import prepare_loader_from_json
import wandb

def run(args):
    # 기본 설정
    # wandb.login()
    
    # 디바이스 확인
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        sys.exit()
    device = torch.device("cuda:0")
    print("Device:", device)

    # seed 설정
    seed_everything(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.phase == "base":
        print("[BASE] Training base model...")
        train_loader = prepare_loader_from_json(
            json_path=args.json_path,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            task_id=None,
            train=True,
        )

        model = General_AD(
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
            smoothing_radius=args.smoothing_radius
        )

        # wandb.init(project=args.wandb_project, name="base_training")
        # wandb_logger = WandbLogger()

        trainer = Trainer(
            log_every_n_steps=args.log_every_n_steps,
            # logger=wandb_logger,
            accelerator="gpu",
            devices=1,
            max_epochs=args.epochs,
            callbacks=[
                ModelCheckpoint(
                    save_top_k=0,
                    save_last=True
                ),
                LearningRateMonitor("epoch")
            ],
            enable_progress_bar=True
        )
        trainer.fit(model, train_loader)
        
        save_path = os.path.join(args.output_dir, "base.ckpt")
        os.makedirs(args.output_dir, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print("[BASE] Training completed. Exiting.")
        return

    elif args.phase == "continual":
        print("[CONTINUAL] Loading saved base model checkpoint...")
        model = General_AD(
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
            smoothing_radius=args.smoothing_radius
        )
        
        pretrained_path = (
            f"/workspace/MegaInspection/GeneralAD/outputs/{args.scenario}/base/base.ckpt" if args.task_id == 1
            else os.path.join(args.output_dir, f"task{args.task_id - 1}.ckpt")
        )
        model.load_state_dict(torch.load(pretrained_path))

        # 인크리멘탈 클래스 학습 루프
        if "except_continual_ad" in args.json_path:
            num_all_tasks = 30
        else:
            num_all_tasks = 60
        num_classes = re.match(r'\d+', args.json_path).group()
        num_tasks = num_all_tasks // int(num_classes)
        print(f"[CONTINUAL] Number of classes: {num_classes}")
        print(f"[CONTINUAL] Number of all tasks: {num_all_tasks}")
        print(f"[CONTINUAL] Number of each tasks: {num_tasks}")
        
        train_loader = prepare_loader_from_json(
            json_path=args.json_path,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            task_id=args.task_id,
            train=True,
        )

        # wandb.init(project=args.wandb_project, name=f"task_{args.task_id+1}", reinit=True)
        # wandb_logger = WandbLogger()

        # model.hparams.epochs = args.epochs

        trainer = Trainer(
            log_every_n_steps=args.log_every_n_steps,
            # logger=wandb_logger,
            accelerator="gpu",
            devices=1,
            max_epochs=args.epochs,
            callbacks=[
                ModelCheckpoint(
                    save_top_k=0,
                    save_last=True
                ),
                LearningRateMonitor("epoch")
            ],
            enable_progress_bar=True
        )

        trainer.fit(model, train_loader)
        
        save_path = os.path.join(args.output_dir, f"task{args.task_id}.ckpt")
        os.makedirs(args.output_dir, exist_ok=True)
        torch.save(model.state_dict(), save_path)

        # wandb.finish()

    else:
        print("Invalid phase selected. Use --phase base or --phase continual")
        sys.exit()
