import argparse
import os
from collections import namedtuple

import torch
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import models, transforms
from tqdm import tqdm

from dataset import EuroSAT, random_split, ImageFiles

# to be sure that we don't mix them
TestResult = namedtuple('TestResult', 'truth predictions')


def predict(model: nn.Module, dl: torch.utils.data.DataLoader, paths=None, show_progress=True):
    """
    Run the model on the specified data.
    Automatically moves the samples to the same device as the model.
    """
    if show_progress:
        dl = tqdm(dl, "Predict", unit="batch")
    device = next(model.parameters()).device

    model.eval()
    preds = []
    truth = []
    i = 0
    for images, labels in dl:
        images = images.to(device, non_blocking=True)
        p = model(images).argmax(1).tolist()
        preds += p
        truth += labels.tolist()

        if paths:
            for pred in p:
                print(f"{paths[i]!r}, {pred}")
                i += 1

    return TestResult(truth=torch.as_tensor(truth), predictions=torch.as_tensor(preds))


def main(args):
    save = torch.load(args.model, map_location=args.device)
    normalization = save['normalization']
    model = models.resnet50(num_classes=save['model_state']['fc.bias'].numel())
    model.load_state_dict(save['model_state'])

    tr = transforms.Compose([transforms.ToTensor(), transforms.Normalize(**normalization)])
    if args.files:
        test = ImageFiles(args.files, transform=tr)
    else:
        print("EUROSAT")
        dataset = EuroSAT(transform=tr)
        trainval, test = random_split(dataset, 0.9, random_state=42)

    test_dl = torch.utils.data.DataLoader(
        test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True
    )
    result = predict(model, test_dl, paths=args.files)

    if not args.files:
        # this is the test, so we need to analyze results
        # TODO

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model', default='weights/best.pt', type=str, help="Model to use")
    parser.add_argument(
        '-j',
        '--workers',
        default=4,
        type=int,
        metavar='N',
        help="Number of workers for the DataLoader",
    )
    parser.add_argument('-b', '--batch-size', default=8, type=int, metavar='N')
    parser.add_argument('files', nargs='*', help="Files to run prediction on")
    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main(args)
