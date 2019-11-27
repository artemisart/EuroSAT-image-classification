import os

from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

URL = "http://madm.dfki.de/files/sentinel/EuroSAT.zip"
MD5 = "c8fa014336c82ac7804f0398fcb19387"
SUBDIR = '2750'


class EuroSAT(ImageFolder):
    def __init__(self, root='data', transform=None, target_transform=None):
        self.download(root)
        root = os.path.join(root, SUBDIR)
        super().__init__(root, transform=transform, target_transform=target_transform)

    def download(self, root):
        if not check_integrity(os.path.join(root, "EuroSAT.zip")):
            download_and_extract_archive(URL, root, md5=MD5)
