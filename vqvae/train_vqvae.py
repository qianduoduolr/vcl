import cv2
import sys
import json
import time
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from dataset import ImageFolderLMDB
from vqvae import VQVAE


def train(epoch, loader, model, optimizer, scheduler, device):
    model.train()
    loader = tqdm(loader)

    criterion = nn.MSELoss()  # Reconstruction criterion

    commit_loss_list = []
    recon_loss_list = []
    for i, x in enumerate(loader):
        x = x.to(device)
        optimizer.zero_grad()

        out, commit_loss = model(x)
        recon_loss = criterion(out, x)
        loss = commit_loss + recon_loss
        loss.backward()

        optimizer.step()

        loader.set_description(
            (
                f'epoch: {epoch + 1}; '
                f'commit: {commit_loss.item():.5f}; '
                f'recon: {recon_loss.item():.5f}'
            )
        )

        commit_loss_list.append(commit_loss.item())
        recon_loss_list.append(recon_loss.item())

    avg_commit_loss = np.mean(commit_loss_list)
    avg_recon_loss = np.mean(recon_loss_list)

    if scheduler is not None:
        scheduler.step()

    return avg_commit_loss, avg_recon_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/gdata2/pengjl/imagenet-lmdb/train")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--train_epoch", type=int, default=2)
    parser.add_argument("--downsample", type=int, default=1)
    parser.add_argument("--n_embed", type=int, default=4096)
    parser.add_argument("--save_path", type=str, default="/gdata/lirui/models/vqvae/vqvae_d1_n4096.pth")
    parser.add_argument("--pretrained_model", type=str, default="")

    args = parser.parse_args()
    print("input args:\n", json.dumps(vars(args), indent=4, separators=(",", ":")))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = ImageFolderLMDB(args.data_path, args.img_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    model = VQVAE(downsample=args.downsample, n_embed=args.n_embed)

    if args.pretrained_model:
        model.load_state_dict(torch.load(args.pretrained_model))

    model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = None

    # Start to train
    for i in range(args.train_epoch):
        # Training stage
        print(f"Training epoch {i + 1}")
        avg_commit_loss, avg_recon_loss = train(i, train_loader, model, optimizer, scheduler, device)
        print(f"Training epoch {i + 1}; commit: {avg_commit_loss:.5f}; recon: {avg_recon_loss:.5f}")
        
        # Save model
        torch.save(model.state_dict(), args.save_path)


if __name__ == "__main__":
    main()
