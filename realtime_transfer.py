import argparse
import os
import torch
from PIL import Image
from torchvision import transforms
import torchvision.utils as utils
from utils.utils import img_resize, load_segment
import numpy as np
from utils.dataset import make_dataset
import cv2
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='photorealistic')
parser.add_argument('--ckpoint', type=str, default='checkpoints/photo_video.pt')

# data
parser.add_argument('--video', type=str, default='data/content/03.avi')
parser.add_argument('--style', type=str, default='data/style/03.jpeg')

parser.add_argument('--out_dir', type=str, default="output")
parser.add_argument('--max_size', type=int, default=1280)
parser.add_argument('--alpha_c', type=float, default=None)



args = parser.parse_args()


# Reversible Network
from models.RevResNet import RevResNet
if args.mode.lower() == "photorealistic":
    RevNetwork = RevResNet(nBlocks=[10, 10, 10], nStrides=[1, 2, 2], nChannels=[16, 64, 256], in_channel=3, mult=4, hidden_dim=16, sp_steps=2)
elif args.mode.lower() == "artistic":
    RevNetwork = RevResNet(nBlocks=[10, 10, 10], nStrides=[1, 2, 2], nChannels=[16, 64, 256], in_channel=3, mult=4, hidden_dim=64, sp_steps=1)
else:
    raise NotImplementedError()

state_dict = torch.load(args.ckpoint)
RevNetwork.load_state_dict(state_dict['state_dict'])
RevNetwork = RevNetwork.to(device)
RevNetwork.eval()


# Transfer module
from models.cWCT import cWCT
cwct = cWCT()



cap = cv2.VideoCapture(0)


# # Load style image
style = Image.open(args.style).convert('RGB')
style = img_resize(style, args.max_size, down_scale=RevNetwork.down_scale)
style_seg = None
style = transforms.ToTensor()(style).unsqueeze(0).to(device)


while True:

    ret, frame = cap.read()
    if not ret:
        break

    video_height, video_width = np.array(frame).shape[:2]
    if max(video_width, video_height) > args.max_size:
        video_width = int(1.0 * video_width / max(video_width, video_height) * args.max_size)
        video_height = int(1.0 * video_height / max(video_width, video_height) * args.max_size)
    transform_video_size = transforms.Resize((video_height, video_width), interpolation=Image.BICUBIC)
    content = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    content = img_resize(content, args.max_size, down_scale=RevNetwork.down_scale)
    content = transforms.ToTensor()(content).unsqueeze(0).to(device)

    with torch.no_grad():
        # Forward inference
        z_c = RevNetwork(content, forward=True)
        z_s = RevNetwork(style, forward=True)

        # Transfer
        if args.alpha_c is not None:
            assert 0.0 <= args.alpha_c <= 1.0
            z_cs = cwct.interpolation(z_c, styl_feat_list=[z_s], alpha_s_list=[1.0], alpha_c=args.alpha_c)
        else:
            z_cs = cwct.transfer(z_c, z_s, None, None)

        # Backward inference
        stylized = RevNetwork(z_cs, forward=False)


    stylized = transform_video_size(stylized)
    grid = utils.make_grid(stylized.data, nrow=1, padding=0)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    cv2.imshow('Stylized Frame', ndarr[..., ::-1])


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
