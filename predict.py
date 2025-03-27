from transformers import SamProcessor
from transformers import SamModel
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import argparse
import glob
import metrics

import warnings
warnings.filterwarnings("ignore")

# gpu training
if torch.cuda.is_available():
    device = "cuda" 
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else: 
    device = "cpu"

print('Model on: ', device)

def show_mask(mask, ax, random_color=False):
            if random_color:
                color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
            else:
                color = np.array([30/255, 144/255, 255/255, 0.6])
            h, w = mask.shape[-2:]
            mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
            ax.imshow(mask_image)


TEST_FOLDER = 'FTW/test'
TEST_IMG = 'g212_00016_4.jpg'

if __name__=='__main__':

    parser = argparse.ArgumentParser(description="Script for checking arguments.")
    
    # Add the expected arguments
    parser.add_argument('-w', type=bool, help="Use custom weights", default=False)
    parser.add_argument('-dir', type=str, help="Directory path", default='')
    parser.add_argument('-f', type=str, help="Image to segment", default='')

    # Parse the arguments
    args = parser.parse_args()
    if args.f == '':
        test_img = Image.open(os.path.join(TEST_FOLDER, 'img', TEST_IMG))
    else:
        TEST_IMG = args.f.split('/')[-1]
        if TEST_IMG.split('.')[-1] == 'tif':
            print('loaded tif file')

            import rasterio
            from rasterio.plot import show
            from PIL import Image as im
            import cv2 as cv
            from shapely.geometry import Polygon

            src = rasterio.open(args.f)
            test_img = src.read()[:3,:,:]
            test_img = test_img.transpose(1, 2, 0) / 3000  # Normalizing 
                
            test_img = im.fromarray((test_img * 255).astype(np.uint8))  



    # Initialize the processor
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    # Load the model
    model = SamModel.from_pretrained("facebook/sam-vit-base")

    # make sure we only compute gradients for mask decoder
    for name, param in model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)

    # use custom weights
    if args.w:
        print('load custom weights...', end = '')
        weights_files = glob.glob('./*.pt')
        epoch = np.array([int(file.split('_')[-1].split('.')[0]) for file in weights_files])
        last = np.where(epoch == np.max(epoch))[0][0]

        model.load_state_dict(torch.load(weights_files[last], weights_only=True))
        print('loaded')
    
    model.to(device)

    inputs = processor(test_img, input_text='field', return_tensors="pt").to(device)
    model.eval()

    # forward pass
    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)
    
    # apply sigmoid
    medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
    # convert soft mask to hard mask
    medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
    mask = (medsam_seg_prob > 0.5).astype(np.uint8)

    if args.f == '':
        test_gt = Image.open(os.path.join(TEST_FOLDER, 'gt', TEST_IMG))
        fn = lambda x : 255 if x > 70 else 0
        gt = test_gt.convert('L').point(fn, mode = '1')

        fig, axes = plt.subplots(1,3, )

        axes[0].imshow(test_img)
        axes[0].title.set_text(f"Original Image")
        axes[0].axis("off")

        #axes[1].imshow(np.array(test_img))
        show_mask(mask, axes[1])
        axes[1].title.set_text(f"Predicted mask")
        axes[1].axis("off")

        axes[2].imshow(gt)
        axes[2].title.set_text(f"Ground Thruth")
        axes[2].axis("off")

        # print results
        print('Mean IoU:', metrics.pixel_accuracy(mask, np.asarray(gt)))
        print('Mean Pixel Accuracy:', metrics.IoU(mask, np.asarray(gt)))
        print('Mean Dice:', metrics.dice(mask, np.asarray(gt)))

        plt.show()
    
    else:
        fig, axes = plt.subplots(1,2, )

        axes[0].imshow(test_img)
        axes[0].title.set_text(f"Original Image")
        axes[0].axis("off")

        #axes[1].imshow(np.array(test_img))
        show_mask(mask, axes[1])
        axes[1].title.set_text(f"Predicted mask")
        axes[1].axis("off")

        plt.show()