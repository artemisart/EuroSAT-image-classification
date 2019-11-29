#!/bin/env python3

import argparse
import os
from collections import namedtuple

import torch
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import models, transforms
from tqdm import tqdm

from dataset import EuroSAT, ImageFiles, random_split

# to be sure that we don't mix them, use this instead of a tuple
TestResult = namedtuple('TestResult', 'truth predictions')


@torch.no_grad()
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


def report(result: TestResult, label_names):
    from sklearn.metrics import classification_report, confusion_matrix

    cr = classification_report(result.truth, result.predictions, target_names=label_names, digits=3)
    confusion = confusion_matrix(result.truth, result.predictions)

    try:  # add names if pandas is installed, otherwise don't bother but don't crash
        import pandas as pd

        # keep only initial for columns (or it's too wide when printed)
        confusion = pd.DataFrame(confusion, index=label_names, columns=[s[:3] for s in label_names])
    except ImportError:
        pass

    print("Classification report")
    print(cr)
    print("Confusion matrix")
    print(confusion)


def main(args):
    save = torch.load(args.model, map_location=args.device)
    normalization = save['normalization']
    model = models.resnet50(num_classes=save['model_state']['fc.bias'].numel())
    model.load_state_dict(save['model_state'])

    tr = transforms.Compose([transforms.ToTensor(), transforms.Normalize(**normalization)])
    if args.files:
        test = ImageFiles(args.files, transform=tr)
    else:
        dataset = EuroSAT(transform=tr)
        trainval, test = random_split(dataset, 0.9, random_state=42)

    test_dl = torch.utils.data.DataLoader(
        test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True
    )
    result = predict(model, test_dl, paths=args.files)

    if not args.files:  # this is the test, so we need to analyze results
        report(result, dataset.classes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""Predict the label on the specified files and outputs the results in csv format.
            If no file is specified, then run on the test set of EuroSAT and produce a report.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '-m', '--model', default='weights/best.pt', type=str, help="Model to use for prediction"
    )
    parser.add_argument(
        '-j',
        '--workers',
        default=4,
        type=int,
        metavar='N',
        help="Number of workers for the DataLoader",
    )
    parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N')
    parser.add_argument('files', nargs='*', help="Files to run prediction on")
    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main(args)
