import argparse
import os

import torch
import torchvision
from torch import nn
from torch.utils import make_grid
from torch.utis.tensorboard import SummaryWriter
from torchvision import models, transforms

from dataset import EuroSAT, random_split


def main(args):
    save = torch.load(args.model, map_location=args.device)
    normalization = save['normalization']
    model = models.resnet50()
    model = save['model_state']

    dataset = EuroSAT()
    trainval, test = random_split(dataset, 0.9, random_state=42)
    test.transforms = transforms.Normalize(**normalization)
    test_dl = torch.utils.data.DataLoader(
        test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True
    )

    model.eval()
    preds = []
    labels = []
    for images, l in test_dl:
        p = model(images)
        preds += p
        labels += l

    print(preds)
    print(labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-j',
        '--workers',
        default=4,
        type=int,
        metavar='N',
        help="Number of workers for the DataLoader",
    )
    parser.add_argument('-m', '--model', default='weights/best.pt', type=str, help="Model to use")
    args = parser.parse_args()

    args.device = torch.device('gpu' if torch.cuda.is_available() else 'cpu')
    main(args)
