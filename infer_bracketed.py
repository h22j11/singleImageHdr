import os
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob
from shutil import copyfile

import torch
from PIL import Image
from tqdm import tqdm

from src.datamodules.datamodule import DrTMODataModule
from src.models.ednet import EDNet

import cv2
import numpy as np

torch.autograd.set_grad_enabled(False)
torch.autograd.profiler.profile(enabled=False)
torch.autograd.profiler.emit_nvtx(enabled=False)
torch.autograd.set_detect_anomaly(mode=False)


def getDisplayableImg(img):
    # [0, 255]
    img = (img * 255).clamp(min=0, max=255).round()
    # uint8
    img = img.type(torch.uint8)
    # [1, 3, H, W] -> [H, W, 3]
    img = img.squeeze(0).permute(1, 2, 0)
    # To numpy
    img = img.cpu().numpy()
    return img


def inference(model, img, mask, exp, isUpExposed, name, out_dir):
    pred, _ = model(img, mask, exp, isUpExposed=isUpExposed)
    file_dir, filename = os.path.split(name)
    _, folder_dir = os.path.split(file_dir)

    # Ensure the output directory exists
    output_folder_path = os.path.join(out_dir, folder_dir)
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    pred = Image.fromarray(getDisplayableImg(pred))
    pred.save(os.path.join(output_folder_path, filename))


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--out_dir")
    parser.add_argument("--ckpt")
    parser.add_argument("--in_name")
    parser.add_argument("--cpu", action="store_true")

    parser = DrTMODataModule.add_dataset_specific_args(parser)

    args = parser.parse_args()

    device = torch.device("cpu") if args.cpu else torch.device("cuda")

    datamodule = DrTMODataModule(
        num_workers=args.num_workers,
        batch_size=1,
        test_batch_size=1,
        pin_memory=True,
        image_size=512,
        clip_threshold=0.95,
        test_dir=args.test_dir,
        test_label_path=args.test_label_path,
        test_hdr=True,
    )
    datamodule.setup()
    test_loader = datamodule.test_dataloader()

    model = EDNet.load_from_checkpoint(args.ckpt).to(device=device)
    model.eval()

    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)

    out_folders = [
        os.path.join(args.out_dir, folder)
        for folder in os.listdir(args.test_dir)
        if "label" not in folder
    ]

    if not len(os.listdir(args.out_dir)):
        for folder in out_folders:
            os.mkdir(folder)

    for path, out_folder in zip(
        glob(f"{args.test_dir}/*/*{args.in_name}"), out_folders
    ):
        _, filename = os.path.split(path)
        # Ensure the output directory exists
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        copyfile(path, f"{out_folder}/{filename}")

    for batch in tqdm(test_loader):
        x_name = batch["x_name"]
        y_name = batch["y_name"]
        x = batch["x"].to(device=device)
        y = batch["y"].to(device=device)
        x_mask = batch["x_mask"].to(device=device)
        y_mask = batch["y_mask"].to(device=device)
        x_exp = batch["x_exp"].view(-1, 1, 1, 1).to(device=device)
        y_exp = batch["y_exp"].view(-1, 1, 1, 1).to(device=device)
        x_wrt_y_exp = x_exp / y_exp
        y_wrt_x_exp = y_exp / x_exp

        if args.in_name in x_name[0]:
            inference(model, x, x_mask, y_wrt_x_exp, True, y_name[0], args.out_dir)
        else:
            inference(model, y, y_mask, x_wrt_y_exp, False, x_name[0], args.out_dir)

    pathes = "results/HDR001_1800x600_80_10_0_gamma+1.4/"
    img_paths = ['a','b','c','d','e','f','g','h','i']

    images = [cv2.imread(pathes+img_path+'.png') for img_path in img_paths]

    # 检查是否成功读取所有图像
    if any(img is None for img in images):
        print("有一个或多个图像未能正确读取，请检查图像路径。")
        exit()

    # 曝光时间
    exposure_times = np.array([1 / 16.0, 1 / 8.0, 1 / 4.0, 1 / 2.0, 1.0, 2.0, 4.0, 8.0, 16.0], dtype=np.float32)

    # 合成HDR图像
    merge_debevec = cv2.createMergeDebevec()
    hdr = merge_debevec.process(images, times=exposure_times.copy())

    # 使用Reinhard色调映射
    tonemap_reinhard = cv2.createTonemapReinhard(gamma=0.84, intensity=0, light_adapt=1, color_adapt=0)
    ldr_reinhard = tonemap_reinhard.process(hdr)

    ldr_reinhard = np.clip(ldr_reinhard * 255, 0, 255).astype('uint8')

    # 保存和显示图像
    cv2.imwrite('ldr_image_reinhard.png', ldr_reinhard)
    cv2.imshow('LDR', ldr_reinhard)
    cv2.waitKey(0)
    cv2.destroyAllWindows()