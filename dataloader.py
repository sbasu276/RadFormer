# encoding: utf-8

"""
Read images and corresponding labels.
"""
import cv2
import json
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image, ImageFilter
import os


class GbFeatureDataSet(Dataset):
    def __init__(self, data_dir, file_list):
        feature_names = []
        labels = []
        self.data_dir = data_dir
        with open(file_list, "r") as f:
            for line in f:
                items = line.split(",")
                fname = items[0][:-4]+".pt"
                label = int(items[1])
                feature_names.append(fname)
                labels.append(label)
        self.feature_names = feature_names
        self.labels = labels

    def __len__(self):
        return len(self.feature_names)

    def __getitem__(self, index):
        fname = self.feature_names[index]
        feature = torch.load(os.path.join(self.data_dir, fname))
        label = self.labels[index]
        size = feature.size()[0]
        label = label*torch.ones(size)
        label = torch.as_tensor(label, dtype=torch.int64)
        return feature, label, fname


class GbUsgDataSet(Dataset):
    def __init__(self, data_dir, image_list_file, to_blur=False, sigma=0, transform=None):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        image_names = []
        labels = []
        with open(image_list_file, "r") as f:
            for line in f:
                items = line.split(",")
                image_name= items[0]
                label = int(items[1])
                image_name = os.path.join(data_dir, image_name)
                image_names.append(image_name)
                labels.append(label)
        self.image_names = image_names
        self.labels = labels
        self.transform = transform
        self.to_blur = to_blur
        self.sigma = sigma

    def __getitem__(self, index):
        """
        Args:
            index: the index of item

        Returns:
            image and its labels
        """
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        #image = cv2.imread(image_name)
        if self.to_blur:
            image = image.filter(ImageFilter.GaussianBlur(self.sigma))
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        label = torch.as_tensor(label, dtype=torch.int64)
        return image, label, image_name

    def __len__(self):
        return len(self.image_names)


def crop_image(image, box, p=0.1):
    x1, y1, x2, y2 = box
    x1 = (1-p)*x1
    y1 = (1-p)*y1
    x2 = (1+p)*x2
    y2 = (1+p)*y2
    cropped_img = image.crop((x1,y1,x2,y2))
    return cropped_img


class GbDataSet(Dataset):
    def __init__(self, data_dir, image_list_file, df, p=0.1, train=True, transform=None):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        file_names = []
        image_names = []
        labels = []
        with open(image_list_file, "r") as f:
            for line in f:
                items = line.split(",")
                image_name= items[0]
                label = int(items[1])
                file_names.append(image_name)
                image_name = os.path.join(data_dir, image_name)
                image_names.append(image_name)
                labels.append(label)
        self.image_names = image_names
        self.file_names = file_names
        self.labels = labels
        self.transform = transform
        self.df = df
        self.p = p
        self.train = train

    def __getitem__(self, index):
        """
        Args:
            index: the index of item

        Returns:
            image and its labels
        """
        image_name = self.image_names[index]
        file_name = self.file_names[index]
        image = Image.open(image_name).convert('RGB')
        if self.transform is not None:
            x = self.transform(image)
        label = self.labels[index]
        z, label = self.get_crop_image(image, label, file_name)
        return x, z, label, file_name

    def get_crop_image(self, image, label, file_name):
        """ Get ROI cropped images
        """
        label = torch.as_tensor(label, dtype=torch.int64)
        if self.train:
            image = crop_image(image, self.df[file_name]["Gold"], self.p)
            if self.transform is not None:
                img = self.transform(image)
        else:
            if self.transform is not None:
                orig = self.transform(image)
            num_objs = len(self.df[file_name]["Boxes"])
            imgs = []
            labels = []
            for i in range(num_objs):
                bbs = self.df[file_name]["Boxes"][i]
                crop_img = crop_image(image, bbs, self.p)
                if self.transform:
                    crop_img = self.transform(crop_img)
                imgs.append(crop_img)
                labels.append(label)
            if num_objs == 0:
                img = orig.unsqueeze(0)
                label = label.unsqueeze(0)
            else:
                img = torch.stack(imgs, 0)
                label = torch.stack(labels, 0)
        return img, label

    def __len__(self):
        return len(self.image_names)


class GbUsgRoiTrainDataSet(Dataset):
    def __init__(self, data_dir, image_list_file, df, to_blur=False, sigma=0, transform=None):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        image_names = []
        labels = []
        file_names = []
        with open(image_list_file, "r") as f:
            for line in f:
                items = line.split(",")
                image_name= items[0]
                label = int(items[1])
                file_names.append(image_name)
                image_name = os.path.join(data_dir, image_name)
                image_names.append(image_name)
                labels.append(label)
        self.image_names = image_names
        self.file_names = file_names
        self.labels = labels
        self.transform = transform
        self.df = df
        self.to_blur = to_blur
        self.sigma = sigma

    def __getitem__(self, index):
        """
        Args:
            index: the index of item

        Returns:
            image and its labels
        """
        image_name = self.image_names[index]
        file_name = self.file_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.to_blur:
            image = image.filter(ImageFilter.GaussianBlur(self.sigma))
        image = crop_image(image, self.df[file_name]["Gold"])
        if self.transform is not None:
            image = self.transform(image)
        label = torch.as_tensor(label, dtype=torch.int64)
        return image, label, image_name

    def __len__(self):
        return len(self.image_names)


class GbUsgRoiTestDataSet(Dataset):
    def __init__(self, data_dir, image_list_file, df, to_blur=False, sigma=0, transform=None):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        image_names = []
        labels = []
        file_names = []
        with open(image_list_file, "r") as f:
            for line in f:
                items = line.split(",")
                image_name= items[0]
                label = int(items[1])
                file_names.append(image_name)
                image_name = os.path.join(data_dir, image_name)
                image_names.append(image_name)
                labels.append(label)
        self.image_names = image_names
        self.file_names = file_names
        self.labels = labels
        self.transform = transform
        self.df = df
        self.to_blur = to_blur
        self.sigma = sigma

    def __getitem__(self, index):
        """
        Args:
            index: the index of item

        Returns:
            image and its labels
        """
        image_name = self.image_names[index]
        file_name = self.file_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.to_blur:
            image = image.filter(ImageFilter.GaussianBlur(self.sigma))
        #orig = crop_image(image, self.df[file_name]["Gold"])
        if self.transform is not None:
            orig = self.transform(image)
        label = torch.as_tensor(label, dtype=torch.int64)

        num_objs = len(self.df[file_name]["Boxes"])
        imgs = []
        labels = []
        for i in range(num_objs):
            bbs = self.df[file_name]["Boxes"][i]
            crop_img = crop_image(image, bbs, 0.1)
            if self.transform:
                crop_img = self.transform(crop_img)
            imgs.append(crop_img)
            labels.append(label)
        if num_objs == 0:
            img = orig.unsqueeze(0)
            label = label.unsqueeze(0)
        else:
            img = torch.stack(imgs, 0)
            label = torch.stack(labels, 0)
        return img, label, image_name

    def __len__(self):
        return len(self.image_names)


if __name__ == "__main__":
    with open("data/res_new.json", "r") as f:
        df = json.load(f)
    ds = GbUsgRoiTrainDataSet(data_dir="data/gb_imgs", image_list_file="data/cls_split/val.txt",\
            df=df, transform = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor()]))
    dl = DataLoader(dataset=ds, batch_size=4, shuffle=False)
    for img, label, img_name in dl:
        print(label, img.size(),img_name)
    ds = GbUsgRoiTestDataSet(data_dir="data/gb_imgs", image_list_file="data/cls_split/val.txt",\
            df=df, transform = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor()]))
    dl = DataLoader(dataset=ds, batch_size=1, shuffle=False)
    for img, label, img_name in dl:
        print(label, img.size(),img_name)
