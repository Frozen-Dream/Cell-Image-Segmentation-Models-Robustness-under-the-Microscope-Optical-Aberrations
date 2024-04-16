import os
import torch
import numpy as np
import pandas as pd
import torch.utils.data
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class CellDataset_aberration(torch.utils.data.Dataset):
    def __init__(self, root, dataset='train', transform = None):
        self.root = root
        self.dataset = dataset
        self.trasforms =  transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.0], std=[1.0 / 255.0])  # Assuming you want to normalize to [0, 1]
        ])
        self.trans= transform
        assert dataset in ["train", "val", "test_convolved"], 'dataset must be in ["train", "val", "test_convolved]'
        self.data = np.load(os.path.join(root, f"{dataset}.npz"), allow_pickle=True)
        self.X = self.data['convolved_X']
        self.y = self.data['y']
        self.meta = pd.DataFrame(self.data['meta'][1:], columns=self.data['meta'][0])
        # self.imgs = list(sorted(os.listdir(os.path.join(root, dataset, "imgs"))))
        # self.masks = list(sorted(os.listdir(os.path.join(root, dataset, "masks"))))

    def __getitem__(self, idx):
        # load images ad masks
        # img_path = os.path.join(self.root, self.dataset, 'imgs', self.imgs[idx])
        # mask_path = os.path.join(self.root,self.dataset, 'masks', self.masks[idx])
        img = Image.fromarray(self.X[idx, 0, :, :]).convert('RGB')
        mask = Image.fromarray(self.y[idx, :, :, 0])
        # img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance with 0 being background
        # mask = Image.open(mask_path)

        # img = np.array(img)
        mask = np.array(mask)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]  # 对背景0进行去除
        masks = mask == obj_ids[:, None, None]  # 进行数据广播，将obj_ids扩展到三维，此时obj_ids的shape是(30, 1, 1)
        img = self.trasforms(img)
        masks = masks.astype(np.float32)  # 将掩码转换为浮点数
        # 获取每一个mask的bounding box
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            # Check if area is larger than a threshold
            # A = abs((xmax - xmin) * (ymax - ymin))
            # # print(A)
            # if A < 5:
            #     print('Nr before deletion:', num_objs)
            #     obj_ids = np.delete(obj_ids, [i])
            #     # print('Area smaller than 5! Box coordinates:', [xmin, ymin, xmax, ymax])
            #     print('Nr after deletion:', len(obj_ids))
            #     continue
            #     # xmax=xmax+5
            #     # ymax=ymax+5
            #
            # boxes.append([xmin, ymin, xmax, ymax])
            boxes.append([xmin, ymin, xmax, ymax])
            # num_objs = len(obj_ids)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        labels = torch.ones((num_objs,), dtype=torch.int64)  # 这里的label是一个一维的tensor
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        meta_data = self.meta.iloc[idx]

        # for i in self.transforms:
        #   img = i(img)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.trans is not None:
            img, target = self.trans(img, target)

        return img, target

    def __len__(self):
        return len(self.X)
        # return len(self.imgs)

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))
    #
    # def get_annotations(self, idx):
    #     """方便构建COCO()"""
    #     h = int(self.y.shape[1])
    #     w = int(self.y.shape[2])
    #     target = self.__getitem__(idx)
    #     return target, h, w


# dataset = CellDataset_aberration(root='/home/server/Desktop/Project/dataset/DNN_with_abe/Ast_0.05',dataset='test_convolved')
# image = Image.fromarray(dataset.X[1, 0, :, :]).convert('L')
# print(dataset[0])