import os
import json

import torch
from tqdm import tqdm
import numpy as np

import transforms
from backbone import resnet50_fpn_backbone
from network_files import MaskRCNN, AnchorsGenerator
from Cell_Dataset import CellDataset
from backbone import BackboneWithFPN, LastLevelMaxPool
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor
from Cell_DNN_aberration import CellDataset_aberration
from sklearn.metrics import precision_recall_curve

import warnings
warnings.filterwarnings("ignore")


def compute_ap(y_true, y_scores):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap = np.trapz(precision, recall)
    ap = abs(ap) * 100
    return ap

def main(parser_data):
    device = torch.device(parser_data.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    data_transform = {
        "val": transforms.Compose([transforms.ToTensor()])
    }

    # read class_indict
    label_json_path = parser_data.label_json_path
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    with open(label_json_path, 'r') as f:
        category_index = json.load(f)

    data_root = parser_data.data_path


    batch_size = parser_data.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using %g dataloader workers' % nw)

    # load validation data set
    # test_dataset = CellDataset(data_root, 'test', transform=None)
    test_dataset = CellDataset_aberration(data_root, 'test_convolved', transform=None)
    test_dataset_loader = torch.utils.data.DataLoader(test_dataset,
                                                     batch_size=batch_size,
                                                     shuffle=False,
                                                     pin_memory=True,
                                                     num_workers=nw,
                                                     collate_fn=test_dataset.collate_fn)

    # create model

    # --- resnet101 fpn backbone --- #
    backbone = torchvision.models.resnet101(pretrained=False)
    # print(backbone)
    return_layers = {"layer2": "0",  # stride 8
                     "layer3": "1",  # stride 16
                     "layer4": "2"}  # stride 32
    new_backbone = create_feature_extractor(backbone, return_layers)
    in_channels_list = [512, 1024, 2048]
    

    # --- vgg19 fpn backbone --- #
    # backbone = torchvision.models.vgg19_bn(pretrained=False)
    # # print(backbone)
    # return_layers = {"features.38": "0",  # stride 8
    #                  "features.51": "1",  # stride 16
    #                  "features.52": "2"}  # stride 32
    # new_backbone = create_feature_extractor(backbone, return_layers)
    # in_channels_list = [512, 512, 512]
    # img = torch.randn(1, 3, 224, 224)
    # outputs = new_backbone(img)
    # [print(f"{k} shape:{v.shape}") for k, v in outputs.items()]


    backbone_with_fpn = BackboneWithFPN(new_backbone,
                                        return_layers=return_layers,
                                        out_channels=256,
                                        in_channels_list=in_channels_list,
                                        extra_blocks=LastLevelMaxPool(),
                                        re_getter=False)

    anchor_sizes = ((64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorsGenerator(sizes=anchor_sizes,
                                        aspect_ratios=aspect_ratios)

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2'],
                                                    output_size=[7, 7],
                                                    sampling_ratio=2)

    model = MaskRCNN(backbone=backbone_with_fpn,
                     num_classes=args.num_classes + 1,
                     rpn_anchor_generator=anchor_generator,
                     box_roi_pool=roi_pooler)

    weights_path = parser_data.weights_path
    assert os.path.exists(weights_path), "not found {} file.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    # print(model)

    model.to(device)

    ap_scores_list = []
    skipped = 0

    for images, targets in tqdm(test_dataset_loader):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        model.eval()
        output = model(images)

        target_im = targets[0]['masks'][0].cpu().detach().numpy()
        for k in range(len(targets[0]['masks'])):
            target_im2 = targets[0]['masks'][k].cpu().detach().numpy()
            target_im2[target_im2 > 0.5] = 1
            target_im2[target_im2 < 0.5] = 0
            target_im = target_im + target_im2

        target_im[target_im > 0.5] = 1
        target_im[target_im < 0.5] = 0
        target_im = target_im.astype('int64')

        output_im = output[0]['masks'][0][0, :, :].cpu().detach().numpy()
        for k in range(len(output[0]['masks'])):
            output_im2 = output[0]['masks'][k][0, :, :].cpu().detach().numpy()
            output_im2[output_im2 > 0.5] = 1
            output_im2[output_im2 < 0.5] = 0
            output_im = output_im + output_im2

        output_im[output_im > 0.5] = 1
        output_im[output_im < 0.5] = 0
        output_im = output_im.astype('int64')

        if target_im.shape != output_im.shape:
            skipped += 1
            continue

        ap_score = compute_ap(target_im.flatten(), output_im.flatten())
        ap_scores_list.append(ap_score)

    print('Mean Average Precision for test set:', np.mean(ap_scores_list))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    # 使用设备类型
    parser.add_argument('--device', default='cuda', help='device')

    # 检测目标类别数(不包含背景)
    parser.add_argument('--num-classes', type=int, default=1, help='number of classes')

    # 数据集的根目录
    parser.add_argument('--data-path', default='', help='dataset root')


    # 训练好的权重文件
    parser.add_argument('--weights-path', default='', type=str, help='training weights')

    # batch size(set to 1, don't change)
    parser.add_argument('--batch-size', default=1, type=int, metavar='N',
                        help='batch size when validation.')
    # 类别索引和类别名称对应关系
    parser.add_argument('--label-json-path', type=str, default="cell_class.json")

    args = parser.parse_args()

    main(args)
