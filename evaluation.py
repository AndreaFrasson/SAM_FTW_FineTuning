import metrics
import torch
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import glob
import numpy as np
from transformers import SamProcessor
from transformers import SamModel
import torch
import argparse
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

DATA_FOLDER = 'FTW/'

if __name__=='__main__':

    parser = argparse.ArgumentParser(description="Script for checking arguments.")
    
    # Add the expected arguments
    parser.add_argument('-w', type=bool, help="Use custom weights", default=False)
    parser.add_argument('-dir', type=str, help="Directory path", default='')

    # Parse the arguments
    args = parser.parse_args()

    if len(args.dir) > 0: 
        # check directory???
        DATA_FOLDER = args.dir

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

    TEST = os.path.join(DATA_FOLDER, 'test')

    # delete empty masks as they may cause issues later on during evaluation
    test_files = os.listdir(os.path.join(TEST, 'gt'))

    valid_instances = []
    for gt in test_files:
        img = Image.open(os.path.join(os.path.join(TEST, 'gt', gt)))
        fn = lambda x : 255 if x > 70 else 0
        img = img.convert('L').point(fn, mode = '1')
        if np.max(img) > 0:
            valid_instances.append(gt)

    # array with the metrics
    iou = []
    pixel_acc = []
    dice = []

    print('start evaluation of the test set')
    # evaluate every valid instance in the test set
    for filename in tqdm(valid_instances):

        # open and process mask ground truth
        test_gt = Image.open(os.path.join(TEST, 'gt', filename))
        fn = lambda x : 255 if x > 70 else 0
        test_gt = test_gt.convert('L').point(fn, mode = '1')

        # open test image
        test_img = Image.open(os.path.join(TEST, 'img', filename))

        # predict mask
        inputs = processor(test_img, input_text='field', return_tensors="pt").to(device)

        model.eval()

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs, multimask_output=False)
        
        # apply sigmoid
        medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
        # convert soft mask to hard mask
        medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
        predicted_masks = (medsam_seg_prob > 0.5).astype(np.uint8)

        # calculate metrics on the test image
        iou.append(metrics.IoU(predicted_masks, np.asarray(test_gt)))
        pixel_acc.append(metrics.pixel_accuracy(predicted_masks, np.asarray(test_gt)))
        dice.append(metrics.dice(predicted_masks, np.asarray(test_gt)))


    # print results
    print('Mean IoU:', np.mean(iou))
    print('Mean Pixel Accuracy:', np.mean(pixel_acc))
    print('Mean Dice:', np.mean(dice))