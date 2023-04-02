"""
based on https://github.com/timesler/facenet-pytorch/blob/master/examples/lfw_evaluate.ipynb
"""

import os
from argparse import ArgumentParser

import torch
from facenet_pytorch import MTCNN, training
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('batched', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()

    batch_size = 16 if args.batched else 1
    workers = 0 if os.name == 'nt' else 8

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))

    mtcnn = MTCNN(
        image_size=160,
        margin=14,
        device=device,
        selection_method='center_weighted_size'
    )

    # Define the data loader for the input set of images
    orig_img_ds = datasets.ImageFolder(args.data_dir, transform=None)

    # overwrites class labels in dataset with path so path can be used for saving output in mtcnn batches
    orig_img_ds.samples = [
        (p, p)
        for p, _ in orig_img_ds.samples
    ]

    loader = DataLoader(
        orig_img_ds,
        num_workers=workers,
        batch_size=batch_size,
        collate_fn=training.collate_pil
    )

    crop_paths = []

    for x, b_paths in tqdm(loader):
        crops = [p.replace(args.data_dir, args.data_dir + '_cropped') for p in b_paths]
        mtcnn(x, save_path=crops)
        crop_paths.extend(crops)


if __name__ == '__main__':
    main()
