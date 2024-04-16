import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from zernike_psf import ZernikePSFGenerator
from utils import get_custom_phaseZ_single, get_phaseZ_multiple, \
    convolve_and_add_noise, read_DNN_data, read_livecell_data, compute_PSNR, save_image

device = torch.device('cpu')

# nMed：介质中的折射率
# NA：数值孔径
# Lambda：波长
# RefractiveIndex：折射率

opt_para_DNN = {'device': device,
                'kernel_size': 33,
                'NA': 0.75,
                'Lambda': 0.35,
                'RefractiveIndex': 1,
                'SigmaX': 2.0,
                'SigmaY': 2.0,
                'Pixelsize': 0.25,
                'nMed': 1}

psf_gen = ZernikePSFGenerator(opt=opt_para_DNN)


def livecell_process_and_save_images(input_folder, output_folder, cell_types, amp_opt):
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device('cpu')
    phaseZ = get_custom_phaseZ_single(opt=amp_opt, batch_size=1).to(device)
    # phaseZ = get_phaseZ_multiple(opt=opt_muti_amp).to(device)

    for cell_type in cell_types:
        cell_folder = os.path.join(input_folder, cell_type)
        output_cell_folder = os.path.join(output_folder, cell_type)

        # Ensure output subfolder exists for each cell type
        if not os.path.exists(output_cell_folder):
            os.makedirs(output_cell_folder)

        filenames = [filename for filename in os.listdir(cell_folder) if filename.endswith(".tif")]
        for filename in tqdm(filenames, desc=f"Processing {cell_type} images"):
            input_image = read_livecell_data(cell_type=cell_type, name=filename)
            psf = psf_gen.generate_PSF(phaseZ=phaseZ)

            convolved_image = convolve_and_add_noise(input_image, psf, padding_mode="circular", padding_value=0,
                                                             img_signal_range=(1000, 2000))

            # Save convolved image with the same filename
            output_path = os.path.join(output_cell_folder, filename)
            save_image(convolved_image, output_path)


def DNN_convolve_and_save_images(input_file, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 从输入的npz文件中加载数据
    data = np.load(input_file, allow_pickle=True)
    data_X = data['X']

    # 准备卷积相关的组件
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device('cpu')
    phaseZ = get_custom_phaseZ_single(opt=amp_opt, batch_size=1).to(device)
    # phaseZ = get_phaseZ_multiple(opt=opt_muti_amp).to(device)

    # 定义变换管道
    transforms_pipeline = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5 / 255.0])
    ])

    # 初始化列表以存储卷积后的图像
    convolved_images = []

    # 遍历数据集中的每个图像
    for idx in tqdm(range(data_X.shape[0]), desc="Convolution Progress"):
        # if idx in skip_indices:
        #     print(f"Skipping idx {idx}.")
        #     continue
        img_array = data_X[idx, :, :, 0]
        img = Image.fromarray(img_array).convert('L')
        img = transforms_pipeline(img)

        # Perform convolution and add noise
        psf = psf_gen.generate_PSF(phaseZ=phaseZ)
        convolved_image = convolve_and_add_noise(img, psf, padding_mode="circular", padding_value=0,
                                                 img_signal_range=(1000, 2000))

        # Append convolved image to the list
        convolved_images.append(convolved_image.numpy())

        # Convert the list to a numpy array
    convolved_images = np.array(convolved_images)

    # Create a new npz file with convolved images
    output_file = os.path.join(output_folder, 'test_convolved.npz')
    np.savez(output_file, convolved_X=convolved_images, **data)
