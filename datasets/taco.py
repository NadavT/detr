from PIL import Image, ExifTags
from copy import deepcopy
import os
from pycocotools.coco import COCO
import torch
import torchvision

class TacoDataset(torchvision.datasets.CocoDetection):
    def __init__(self, root, transforms, categories, annotations, single_class=False, toy=False):
        super(TacoDataset, self).__init__(root, annotations)
        self.root = root
        self.transforms = transforms
        self.single_class = single_class

        self.coco = COCO(annotations)
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                self.orientation = orientation
                break
        imgIds = []
        if categories == "all":
            self.imgs = self.coco.loadImgs(self.coco.getImgIds())
            self.catIds = self.coco.getCatIds()
        else:
            self.catIds = self.coco.getCatIds(catNms=categories)
            if self.catIds:
                # Get all images containing an instance of the chosen category
                imgIds = self.coco.getImgIds(catIds=self.catIds)
            else:
                # Get all images containing an instance of the chosen super category
                self.catIds = self.coco.getCatIds(supNms=categories)
                for catId in self.catIds:
                    imgIds += (self.coco.getImgIds(catIds=catId))
                imgIds = list(set(imgIds))
            self.imgs = self.coco.loadImgs(imgIds)
        self.imgs = [
            img for img in self.imgs if img['file_name'].startswith('batch')]
        if toy:
            self.imgs = self.imgs[:10]
        self.catMap = {catId: i + 1 for i, catId in enumerate(self.catIds)}

    def __getitem__(self, idx):
        # load images and masks
        img_info = self.imgs[idx]
        img = Image.open(os.path.join(self.root, img_info['file_name']))
        annIds = self.coco.getAnnIds(
            imgIds=img_info['id'], catIds=self.catIds, iscrowd=None)
        anns_sel = deepcopy(self.coco.loadAnns(annIds))

        for ann in anns_sel:
            ann['category_id'] = self.catMap[ann['category_id']]

        if img._getexif():
            exif = dict(img._getexif().items())
            # Rotate portrait and upside down images if necessary
            if self.orientation in exif:
                if exif[self.orientation] == 3:
                    img = img.rotate(180, expand=True)
                if exif[self.orientation] == 6:
                    img = img.rotate(270, expand=True)
                if exif[self.orientation] == 8:
                    img = img.rotate(90, expand=True)
        img = img.convert("RGB")

        target = {}
        target['image_id'] = torch.tensor([img_info['id']])
        target['annotations'] = anns_sel

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        # if self.single_class:
        #     target['labels'] = torch.ones_like(target['labels'])

        return img, target

    def __len__(self):
        return len(self.imgs)
