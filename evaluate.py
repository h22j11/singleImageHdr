import cv2
import skimage
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np


def crop_to_same_size(img1, img2):
    # 获取最小的宽度和高度
    min_height = min(img1.shape[0], img2.shape[0])
    min_width = min(img1.shape[1], img2.shape[1])

    # 裁剪图像
    cropped_img1 = img1[:min_height, :min_width]
    cropped_img2 = img2[:min_height, :min_width]

    return cropped_img1, cropped_img2


def evaluate_hdr_image(hdr_image_path, reference_hdr_image_path):
    # 读取HDR图像
    hdr_img = cv2.imread(hdr_image_path, cv2.IMREAD_ANYDEPTH)
    ref_hdr_img = cv2.imread(reference_hdr_image_path, cv2.IMREAD_ANYDEPTH)

    # 裁剪图像到相同的尺寸
    hdr_img, ref_hdr_img = crop_to_same_size(hdr_img, ref_hdr_img)

    # 计算PSNR
    psnr_value = peak_signal_noise_ratio(ref_hdr_img, hdr_img)

    # 计算SSIM
    ssim_value, _ = structural_similarity(ref_hdr_img, hdr_img, full=True, multichannel=True)

    # Calculate MSSIM
    mssim_value = skimage.metrics.structural_similarity(ref_hdr_img, hdr_img, multichannel=True, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, K1=0.01, K2=0.03)


    return psnr_value, ssim_value,mssim_value


# 示例调用
psnr, ssim, mssim = evaluate_hdr_image('C.png', 'raw.jpg')
print(f"PSNR: {psnr}, SSIM: {ssim}, MSSIM: {mssim}")
