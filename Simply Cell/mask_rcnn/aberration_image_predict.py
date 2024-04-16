import os
import time
import json

import numpy as np
from PIL import Image

import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('TkAgg')

import torch
from torchvision import transforms

from network_files import MaskRCNN
from backbone import resnet50_fpn_backbone
from draw_box_utils import draw_objs

import torchvision
from torchvision.models.feature_extraction import create_feature_extractor

from network_files import MaskRCNN, AnchorsGenerator
from backbone import BackboneWithFPN, LastLevelMaxPool

from swin_transformer import swin_t, Swin_T_Weights, swin_s, Swin_S_Weights

import warnings
warnings.filterwarnings("ignore")


def create_r50_model(num_classes, box_thresh=0.5):
    backbone = resnet50_fpn_backbone()
    model = MaskRCNN(backbone,
                     num_classes=num_classes,
                     rpn_score_thresh=box_thresh,
                     box_score_thresh=box_thresh)

    return model


def create_r50_witout_fpn_model(num_classes, box_thresh=0.5):
    backbone = torchvision.models.resnet50(pretrained=True)
    backbone = create_feature_extractor(backbone, return_nodes={"layer3": "0"})
    backbone.out_channels = 1024
    model = MaskRCNN(backbone,
                     num_classes=num_classes,
                     rpn_score_thresh=box_thresh,
                     box_score_thresh=box_thresh)
    return model


def create_r101_model(num_classes, box_thresh=0.5):
    backbone = torchvision.models.resnet101(pretrained=False)
    # print(backbone)
    return_layers = {"layer2": "0",  # stride 8
                     "layer3": "1",  # stride 16
                     "layer4": "2"}  # stride 32
    new_backbone = create_feature_extractor(backbone, return_layers)
    in_channels_list = [512, 1024, 2048]
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
                     num_classes=num_classes,
                     rpn_anchor_generator=anchor_generator,
                     box_roi_pool=roi_pooler,
                     rpn_score_thresh=box_thresh,
                     box_score_thresh=box_thresh)
    return model

def create_vgg19_model(num_classes, box_thresh=0.5):
    backbone = torchvision.models.vgg19_bn(pretrained=False)
    # print(backbone)
    return_layers = {"features.38": "0",  # stride 8
                     "features.51": "1",  # stride 16
                     "features.52": "2"}  # stride 32
    new_backbone = create_feature_extractor(backbone, return_layers)
    in_channels_list = [512, 512, 512]
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
                     num_classes=num_classes,
                     rpn_anchor_generator=anchor_generator,
                     box_roi_pool=roi_pooler,
                     rpn_score_thresh=box_thresh,
                     box_score_thresh=box_thresh)
    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    num_classes = 1  # 不包含背景
    box_thresh = 0.5
    idx = 7
    amp = 1.0
    abe = 'Tre'
    weights_path = ''

    img_path = f''
    data = np.load(os.path.join(img_path, 'test_convolved.npz'), allow_pickle=True)
    test_data = data['convolved_X']

    label_json_path = './cell_class.json'

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = create_vgg19_model(num_classes=num_classes+1, box_thresh=box_thresh)

    # load train weights
    assert os.path.exists(weights_path), "{} file dose not exist.".format(weights_path)
    weights_dict = torch.load(weights_path, map_location='cpu')
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    model.load_state_dict(weights_dict)
    model.to(device)

    # read class_indict
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    with open(label_json_path, 'r') as json_file:
        category_index = json.load(json_file)

    # load image
    assert os.path.exists(img_path), f"{img_path} does not exits."
    # original_img = Image.open(img_path).convert('RGB')
    # original_img = Image.fromarray(test_data[idx, :, :, 0]).convert('RGB')
    original_img = Image.fromarray(test_data[idx, 0, :, :]).convert('RGB')

    # from pil image to tensor, do not normalize image
    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    model.eval()  # 进入验证模式
    with torch.no_grad():
        # init
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        predictions = model(img.to(device))[0]
        t_end = time_synchronized()
        # print("inference+NMS time: {}".format(t_end - t_start))

        predict_boxes = predictions["boxes"].to("cpu").numpy()
        predict_classes = predictions["labels"].to("cpu").numpy()
        predict_scores = predictions["scores"].to("cpu").numpy()
        predict_mask = predictions["masks"].to("cpu").numpy()
        predict_mask = np.squeeze(predict_mask, axis=1)  # [batch, 1, h, w] -> [batch, h, w]

        if len(predict_boxes) == 0:
            print("没有检测到任何目标!")
            return

        plot_img = draw_objs(original_img,
                             boxes=predict_boxes,
                             classes=predict_classes,
                             scores=predict_scores,
                             masks=predict_mask,
                             category_index=category_index,
                             line_thickness=3,
                             font='arial.ttf',
                             font_size=20)
        plt.imshow(plot_img)
        plt.show()
        # 保存预测的图片结果
        # plot_img.save(f'test_result_{idx}.jpg')


if __name__ == '__main__':
    main()

