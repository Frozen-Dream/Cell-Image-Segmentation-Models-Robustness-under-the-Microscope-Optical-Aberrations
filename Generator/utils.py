import os
import math
import torch
import random
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim_skimage
from skimage.feature import local_binary_pattern

def get_phaseZ_multiple(opt=None, batch_size=1, device=torch.device('cpu')):
    """
    opt: default = {'idx_start': 4, 'num_idx': 11, 'mode': 'gaussian', 'std': 0.125, 'bound': 1.0}
    """
    if opt is None:
        opt = {'idx_start': [4], 'num_idx': [15], 'mode': 'gaussian', 'std': 0.125, 'bound': [1.0]}
    phaseZ = torch.zeros(size=(batch_size, 25))
    if opt['mode'] == 'gaussian':
        for i, n, b in zip(opt['idx_start'], opt['num_idx'], opt['bound']):
            phaseZ[:, i:i + n] = torch.normal(mean=0.0, std=opt['std'], size=(batch_size, n))
            phaseZ[:, i:i + n] = torch.clamp(phaseZ[:, i:i + n], min=0, max=b)
    elif opt['mode'] == 'uniform':
        for i, n, b in zip(opt['idx_start'], opt['num_idx'], opt['bound']):
            phaseZ[:, i:i + n] = torch.rand(size=(batch_size, n)) * 2.0 * b - b
    else:
        raise NotImplementedError
    return phaseZ.to(device)


def get_custom_phaseZ_single(opt, batch_size=1, device=torch.device('cpu')):
    """
    opt: {'mode': 'Astigmatism', 'amplitude': 0.5}
    """
    phaseZ = torch.zeros(size=(batch_size, 25))

    mode = opt['mode']
    amplitude = opt['amplitude']

    if mode == 'Astigmatism':
        # Set Z4 and Z5 amplitudes
        phaseZ[:, 4] = amplitude
        phaseZ[:, 5] = amplitude
    elif mode == 'Coma':
        # Set Z6 and Z7 amplitudes
        phaseZ[:, 6] = amplitude
        phaseZ[:, 7] = amplitude
    elif mode == 'Spherical':
        # Set Z8 amplitude
        phaseZ[:, 8] = amplitude
    elif mode == 'Trefoil':
        # Set Z9 and Z10 amplitudes
        phaseZ[:, 9] = amplitude
        phaseZ[:, 10] = amplitude
    else:
        raise ValueError("Unsupported mode. Supported modes are: 'Astigmatism', 'Coma', 'Spherical', 'Trefoil'")

    return phaseZ.to(device)


def add_poisson_gaussian_noise(img, level=1000.0):
    if torch.max(img) == 0.0:
        poisson = torch.poisson(torch.zeros(*img.shape)).to(img.device)
    else:
        poisson = torch.poisson(img / torch.max(img) * level).to(img.device)
    gaussian = torch.normal(mean=torch.ones(*img.shape) * 100.0, std=torch.ones(*img.shape) * 4.5).to(img.device)
    img_noised = poisson + gaussian
    assert torch.max(img_noised) - torch.min(img_noised) != 0.0
    img_noised = (img_noised - torch.min(img_noised)) / (torch.max(img_noised) - torch.min(img_noised))
    if torch.max(img) != 0.0:
        img_noised = img_noised * (torch.max(img) - torch.min(img)) + torch.min(img)
    else:
        # raise RuntimeWarning('occur purely dark img')
        print('occur purely dark img')
    return img_noised


def convolve_and_add_noise(hr, kernel, **kwargs):
    """
    将输入图像与PSF卷积，并添加噪声

    Parameters:
        hr (torch.Tensor): 输入图像
        kernel (torch.Tensor): PSF
        padding_mode (str): 卷积时的填充模式，默认为"circular"
        padding_value (float): 填充的值，默认为0.0
        img_signal_range (tuple): 图像信号范围，默认为(1000.0, 2000.0)

    Returns:
        torch.Tensor: 添加噪声后的图像
    """

    padding_mode = kwargs.get('padding_mode', "circular")
    padding_value = kwargs.get('padding_value', 0.0)
    # img_signal_range = kwargs.get('img_signal_range', (1000.0, 2000.0))

    # 卷积
    # pad = (kernel.shape[-2] // 2, kernel.shape[-1] // 2)
    pad = (kernel.shape[-2] // 2,) * 2 + (kernel.shape[-1] // 2,) * 2
    if padding_mode == "circular":
        lr = F.conv2d(F.pad(hr.unsqueeze(0), pad=pad, mode=padding_mode), kernel.unsqueeze(0)).squeeze(0)
    else:
        lr = F.conv2d(F.pad(hr.unsqueeze(0), pad=pad, mode=padding_mode, value=padding_value),
                      kernel.unsqueeze(0)).squeeze(0)
    # else:
    #     lr = F.conv2d(F.pad(hr.unsqueeze(0), pad=pad, mode='constant', value=0),
    #                 kernel.unsqueeze(0)).squeeze(0)

    # 添加噪声
    # img_signal = 10.0 ** random.uniform(math.log10(img_signal_range[0]), math.log10(img_signal_range[1]))
    # lr_noised = add_poisson_gaussian_noise(lr, level=img_signal)

    # return lr_noised
    return lr

def read_DNN_data(data_root, idx, mode):
    root = data_root
    if mode == 'normal':
        dataset = 'test'
    else:
        dataset = 'test_convolved'
    data = np.load(os.path.join(root, f"{dataset}.npz"), allow_pickle=True)
    # trasforms = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5], std=[0.5 / 255.0])  # Assuming you want to normalize to [0, 1]
    # ])
    transforms_pipeline = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5 / 255.0])  # Assuming you want to normalize to [0, 1]
    ])
    if mode == 'normal':
        img = Image.fromarray(data['X'][idx, :, :, 0]).convert('L')
    else:
        img = Image.fromarray(data['convolved_X'][idx, 0, :, :]).convert('L')
    # img = trasforms(img).convert('RGB')
    # img_array = np.array(img)
    # img_tensor = torch.from_numpy(img_array).float() / 255.0  # 归一化到 [0, 1]
    img = transforms_pipeline(img)
    return img

def read_livecell_data(cell_type, name):
    root = '/home/server/Desktop/Project/dataset/livecell/livecell_test_images'
    image_path = os.path.join(root, f"{cell_type}", f"{name}")
    img = Image.open(image_path).convert('L')
    transforms_pipeline = transforms.Compose([
         transforms.ToTensor(),
         # transforms.Normalize(mean=[0.5], std=[0.5 / 255.0])  # Assuming you want to normalize to [0, 1]
     ])
    img = transforms_pipeline(img)
    return img


def compute_PSNR(original_image, distorted_image):
    original_image = original_image.squeeze().cpu().numpy()
    distorted_image = distorted_image.squeeze().cpu().numpy()
    # Ensure the images are in the range [0, 1]
    original_image = np.clip(original_image, 0, 1)
    distorted_image = np.clip(distorted_image, 0, 1)
    psnr_value = psnr(original_image, distorted_image)

    return psnr_value


def compute_SSIM(original_image, distorted_image):
    """
    计算两个图像的结构相似性指数（SSIM）。

    参数：
    - original_image：表示原始图像的NumPy数组或PyTorch张量。
    - distorted_image：表示失真图像的NumPy数组或PyTorch张量。

    返回：
    - ssim_value：计算得到的SSIM值。
    """

    # 确保图像在[0, 1]范围内
    original_image = np.clip(original_image, 0, 1)
    distorted_image = np.clip(distorted_image, 0, 1)

    # 将PyTorch张量转换为NumPy数组
    if isinstance(original_image, torch.Tensor):
        original_image = original_image.cpu().numpy()
    if isinstance(distorted_image, torch.Tensor):
        distorted_image = distorted_image.cpu().numpy()

    # 计算SSIM
    ssim_value, _ = ssim(original_image, distorted_image, full=True)

    return ssim_value


def save_image(img, path):
    # Convert tensor to PIL Image and save
    img = transforms.ToPILImage()(img)
    img.save(path, format='TIFF')

