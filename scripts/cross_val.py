import logging
import os
import sys

lib_path = os.path.abspath("").replace("scripts", "src")
sys.path.append(lib_path)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

import argparse
import csv
import datetime
import glob as glob_module
import pickle
import random
import tempfile

import numpy as np
import torch
from torch import optim

import trainer as Trainer
from configs.base import Config
from data.dataloader import BaseDataset
from eval import eval as evaluate_model
from models import losses, networks, optims
from utils.configs import get_options
from utils.torch.callbacks import CheckpointsCallback
from torch.utils.data import DataLoader

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_SESSIONS = 5


def load_session_data(data_root: str) -> dict:
    session_data = {}
    for sid in range(1, NUM_SESSIONS + 1):
        path = os.path.join(data_root, f"session_{sid}.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Session file not found: {path}\n"
                "Run: python preprocess.py -ds IEMOCAP -dr <data_root> --cross_val"
            )
        with open(path, "rb") as f:
            session_data[sid] = pickle.load(f)
        logging.info(f"Session {sid}: {len(session_data[sid])} samples")
    return session_data


def write_fold_pkls(session_data: dict, test_session: int, fold_dir: str, val_ratio: float = 0.1):
    all_train = []
    for sid, samples in session_data.items():
        if sid != test_session:
            all_train.extend(samples)
    test_samples = session_data[test_session]

    # Random split of training sessions into train / val
    random.Random(SEED).shuffle(all_train)
    n_val = max(1, int(len(all_train) * val_ratio))
    val_samples = all_train[:n_val]
    train_samples = all_train[n_val:]

    with open(os.path.join(fold_dir, "train.pkl"), "wb") as f:
        pickle.dump(train_samples, f)
    with open(os.path.join(fold_dir, "val.pkl"), "wb") as f:
        pickle.dump(val_samples, f)
    with open(os.path.join(fold_dir, "test.pkl"), "wb") as f:
        pickle.dump(test_samples, f)

    return len(train_samples), len(val_samples), len(test_samples)


def train_fold(cfg: Config, fold_idx: int, base_checkpoint_dir: str):
    # Re-initialize model weights for each fold
    try:
        network = getattr(networks, cfg.model_type)(cfg)
        network.to(device)
    except AttributeError:
        raise NotImplementedError(f"Model {cfg.model_type} is not implemented")

    # Fold-specific checkpoint directory
    cfg.checkpoint_dir = checkpoint_dir = os.path.join(
        base_checkpoint_dir,
        cfg.name,
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
        f"fold_{fold_idx}",
    )
    weight_dir = os.path.join(checkpoint_dir, "weights")
    os.makedirs(os.path.join(checkpoint_dir, "logs"), exist_ok=True)
    os.makedirs(weight_dir, exist_ok=True)
    cfg.save(cfg)

    try:
        criterion = getattr(losses, cfg.loss_type)(cfg)
        criterion.to(device)
    except AttributeError:
        raise NotImplementedError(f"Loss {cfg.loss_type} is not implemented")

    try:
        fold_trainer = getattr(Trainer, cfg.trainer)(
            cfg=cfg,
            network=network,
            criterion=criterion,
            log_dir=cfg.checkpoint_dir,
        )
    except AttributeError:
        raise NotImplementedError(f"Trainer {cfg.trainer} is not implemented")

    train_dl = DataLoader(
        BaseDataset(cfg, data_mode="train.pkl"),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    val_dl = DataLoader(
        BaseDataset(cfg, data_mode="val.pkl"),
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    optimizer = optims.get_optim(cfg, network)
    lr_scheduler = None
    if cfg.learning_rate_step_size is not None:
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.learning_rate_step_size,
            gamma=cfg.learning_rate_gamma,
        )

    ckpt_callback = CheckpointsCallback(
        checkpoint_dir=weight_dir,
        save_freq=cfg.save_freq,
        max_to_keep=cfg.max_to_keep,
        save_best_val=cfg.save_best_val,
        save_all_states=cfg.save_all_states,
    )

    fold_trainer.compile(optimizer=optimizer, scheduler=lr_scheduler)
    fold_trainer.fit(train_dl, cfg.num_epochs, val_dl, callbacks=[ckpt_callback])

    # Locate best checkpoint
    best_ckpt = os.path.join(weight_dir, "best_acc", "checkpoint_0_0.pt")
    all_state_dict = True
    if not os.path.exists(best_ckpt):
        best_ckpt = os.path.join(weight_dir, "best_acc", "checkpoint_0.pth")
        all_state_dict = False

    return best_ckpt, all_state_dict


def main(cfg: Config, base_checkpoint_dir: str):
    session_data = load_session_data(cfg.data_root)

    results = []
    fields = ["Fold", "BACC", "ACC", "MACRO_F1", "WEIGHTED_F1"]
    csv_path = os.path.join(
        base_checkpoint_dir,
        f"cross_val_{cfg.name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.csv",
    )
    os.makedirs(base_checkpoint_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as fold_data_dir:
        original_data_root = cfg.data_root
        original_data_valid = cfg.data_valid
        cfg.data_root = fold_data_dir
        cfg.data_valid = "test.pkl"

        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fields)
            writer.writeheader()

            for test_session in range(1, NUM_SESSIONS + 1):
                logging.info(
                    f"\n{'='*60}\nFold {test_session}/{NUM_SESSIONS} - test session: Session{test_session}\n{'='*60}"
                )

                n_train, n_val, n_test = write_fold_pkls(session_data, test_session, fold_data_dir)
                logging.info(f"Train: {n_train} | Val: {n_val} | Test: {n_test} samples")

                best_ckpt, all_state_dict = train_fold(
                    cfg,
                    fold_idx=test_session,
                    base_checkpoint_dir=base_checkpoint_dir,
                )

                # Restore data_root after train_fold may have modified cfg.checkpoint_dir
                cfg.data_root = fold_data_dir
                cfg.data_valid = "test.pkl"

                bacc, acc, macro_f1, weighted_f1 = evaluate_model(
                    cfg, best_ckpt, all_state_dict=all_state_dict
                )

                row = {
                    "Fold": test_session,
                    "BACC": round(bacc * 100, 2),
                    "ACC": round(acc * 100, 2),
                    "MACRO_F1": round(macro_f1 * 100, 2),
                    "WEIGHTED_F1": round(weighted_f1 * 100, 2),
                }
                results.append(row)
                writer.writerow(row)
                csvfile.flush()

                logging.info(
                    f"Fold {test_session} → BACC: {row['BACC']:.2f} | ACC: {row['ACC']:.2f} | "
                    f"Macro-F1: {row['MACRO_F1']:.2f} | Weighted-F1: {row['WEIGHTED_F1']:.2f}"
                )

        cfg.data_root = original_data_root
        cfg.data_valid = original_data_valid

    # Summary
    baccs = [r["BACC"] for r in results]
    accs = [r["ACC"] for r in results]
    macro_f1s = [r["MACRO_F1"] for r in results]
    weighted_f1s = [r["WEIGHTED_F1"] for r in results]

    logging.info("\n" + "=" * 60)
    logging.info("Cross-Validation Summary (mean ± std)")
    logging.info(f"BACC: {np.mean(baccs):.2f} ± {np.std(baccs):.2f}")
    logging.info(f"ACC: {np.mean(accs):.2f} ± {np.std(accs):.2f}")
    logging.info(f"Macro-F1: {np.mean(macro_f1s):.2f} ± {np.std(macro_f1s):.2f}")
    logging.info(f"Weighted-F1: {np.mean(weighted_f1s):.2f} ± {np.std(weighted_f1s):.2f}")
    logging.info(f"Results saved to: {csv_path}")


def arg_parser():
    parser = argparse.ArgumentParser(
        description="Leave-One-Session-Out cross-validation for IEMOCAP"
    )
    parser.add_argument(
        "-cfg", "--config", type=str, default="../src/configs/hubert_base.py",
        help="Path to config file",
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, default=None,
        help="Base directory for checkpoints and results (defaults to cfg.checkpoint_dir)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    cfg: Config = get_options(args.config)

    base_checkpoint_dir = args.output_dir if args.output_dir else os.path.abspath(cfg.checkpoint_dir)

    logging.info(f"Starting 5-fold LOSO cross-validation for IEMOCAP")
    logging.info(f"Data root: {cfg.data_root}")
    logging.info(f"Checkpoints: {base_checkpoint_dir}")

    main(cfg, base_checkpoint_dir)