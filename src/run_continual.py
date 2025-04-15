import os
import re
import torch
import sys
import json

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from .general_ad import General_AD  # General_AD 모델
from .load_data import prepare_loader_from_json  # JSON 기반 데이터 로더 구현 필요
import wandb

def run(args):
    # 기본 설정
    wandb.login()
    
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

    base_ckpt_path = os.path.join(args.data_dir, "base_checkpoint.ckpt")

    if args.phase == "base":
        print("[BASE] Training base model...")
        base_json = os.path.join("/workspace/meta_files", "base_classes.json")
        train_loader, val_loader = prepare_loader_from_json(
            json_path=base_json,
            image_size=args.image_size,
            batch_size=args.batch_size,
            test_batch_size=args.test_batch_size,
            num_workers=args.num_workers
        )

        model = General_AD(
            lr=args.lr,
            lr_decay_factor=args.lr_decay_factor,
            hf_path=args.hf_path,
            layers_to_extract_from=args.layers_to_extract_from,
            hidden_dim=args.hidden_dim,
            wd=args.wd,
            epochs=args.base_epochs,
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

        wandb.init(project=args.wandb_project, name="base_training")
        wandb_logger = WandbLogger()

        trainer = Trainer(
            log_every_n_steps=args.log_every_n_steps,
            logger=wandb_logger,
            accelerator="gpu",
            devices=1,
            max_epochs=args.base_epochs,
            callbacks=[
                ModelCheckpoint(
                    save_weights_only=True,
                    mode="max", 
                    monitor=f"val_{args.val_monitor}", 
                    dirpath=args.data_dir, 
                    filename="base_checkpoint",
                ),
                LearningRateMonitor("epoch")
            ],
            enable_progress_bar=True
        )
        trainer.fit(model, train_loader, val_loader)
        torch.save(model.state_dict(), base_ckpt_path)
        print("[BASE] Training completed. Exiting.")
        return

    elif args.phase == "continual":
        print("[BASE] Loading saved base model checkpoint...")
        model = General_AD(
            lr=args.lr,
            lr_decay_factor=args.lr_decay_factor,
            hf_path=args.hf_path,
            layers_to_extract_from=args.layers_to_extract_from,
            hidden_dim=args.hidden_dim,
            wd=args.wd,
            epochs=args.inc_epochs,  # 바로 inc_epochs 설정
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
        model.load_state_dict(torch.load(base_ckpt_path))

        # 인크리멘탈 클래스 학습 루프
        if "except_continual_ad" in args.task_json_name:
            num_all_tasks = 30
        else:
            num_all_tasks = 60
        num_classes = re.match(r'\d+', args.task_json_name).group()
        num_tasks = num_all_tasks // int(num_classes)
        print(f"[INCREMENTAL] Number of classes: {num_classes}")
        print(f"[INCREMENTAL] Number of all tasks: {num_all_tasks}")
        print(f"[INCREMENTAL] Number of each tasks: {num_tasks}")
        
        for task_idx in range(num_tasks):
            task_json = os.path.join("/workspace/meta_files", f"{args.task_json_name}.json")
            print(f"[TASK {task_idx+1}] Loading from {task_json}")
            train_loader, val_loader = prepare_loader_from_json(
                json_path=task_json,
                image_size=args.image_size,
                batch_size=args.batch_size,
                test_batch_size=args.test_batch_size,
                num_workers=args.num_workers,
                task_idx=task_idx,
            )

            wandb.init(project=args.wandb_project, name=f"task_{task_idx+1}", reinit=True)
            wandb_logger = WandbLogger()

            model.hparams.epochs = args.inc_epochs

            trainer = Trainer(
                log_every_n_steps=args.log_every_n_steps,
                logger=wandb_logger,
                accelerator="gpu",
                devices=1,
                max_epochs=args.inc_epochs,
                callbacks=[
                    ModelCheckpoint(
                        save_weights_only=True, 
                        mode="max", 
                        monitor=f"val_{args.val_monitor}",
                        dirpath=os.path.join(args.data_dir, "continual_checkpoints"),
                        save_top_k=0,
                        save_last=True,
                    ),
                    LearningRateMonitor("epoch")
                ],
                enable_progress_bar=True
            )

            trainer.fit(model, train_loader, val_loader)
            trainer.test(model, val_loader)

        wandb.finish()

    else:
        print("Invalid phase selected. Use --phase base or --phase continual")
        sys.exit()
