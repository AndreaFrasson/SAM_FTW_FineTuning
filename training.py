import numpy as np
import os
from patchify import patchify  #Only to handle large images
from PIL import Image
from datasets import Dataset
from SamDataset import SAMDataset
from transformers import SamProcessor
from torch.utils.data import DataLoader
from torch.optim import Adam
import monai
from transformers import SamModel
from tqdm import tqdm
from statistics import mean
import torch
from torch.nn.functional import threshold, normalize
import argparse
import os
import sys
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

# gpu training
if torch.cuda.is_available():
    device = "cuda" 
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else: 
    device = "cpu"

print('Training model on: ', device)

DATA_FOLDER = 'FTW/'
TRAINING = os.path.join(DATA_FOLDER, 'training')

# script to train sam with a custom dataset
# every 10 epochs, the model weights are saved
# to train the model for 100 epochs, run the following command:
# python training.py -epochs 100
# to train the model with another dataset, add at the end -dir /path/to/dataset

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Script for checking arguments.")
    
    # Add the expected arguments
    parser.add_argument('-epochs', type=int, help="Number of epochs", default=10)
    parser.add_argument('-dir', type=str, help="Directory path", default='')

    # Parse the arguments
    args = parser.parse_args()

    if len(args.dir) > 0: 
        # check directory???
        DATA_FOLDER = args.dir

    # delete empty masks as they may cause issues later on during training
    train_gt = os.listdir(os.path.join(TRAINING, 'gt'))

    valid_instances = []
    for gt in train_gt:
        img = Image.open(os.path.join(os.path.join(TRAINING, 'gt', gt)))
        fn = lambda x : 255 if x > 70 else 0
        img = img.convert('L').point(fn, mode = '1')
        if np.max(img) > 0:
            valid_instances.append(gt)
    

    data_dict = {}

    for img in valid_instances:
        data_dict['img'] = data_dict.get('img', []) + [Image.open(os.path.join(TRAINING, 'img', img))]

        gt = Image.open(os.path.join(TRAINING, 'gt', img))
        fn = lambda x : 255 if x > 70 else 0
        gt = gt.convert('L').point(fn, mode = '1')
        data_dict['gt'] = data_dict.get('gt', []) + [gt]

    # Create the dataset using the datasets.Dataset class
    dataset = Dataset.from_dict(data_dict)

    # Initialize the processor
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    train_dataset = SAMDataset(dataset=dataset, processor=processor)
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    #Load Model, once the model is trained, other weights can be loaded
    model = SamModel.from_pretrained("facebook/sam-vit-base")

    # only compute gradients for mask decoder
    for name, param in model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)

    # Initialize the optimizer and the loss function
    optimizer = Adam(model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
    #Try DiceFocalLoss, FocalLoss, DiceCELoss
    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    #Training loop
    model.to(device)
    losses = []

    model.train()
    for epoch in range(args.epochs):
        epoch_losses = []
        for batch in tqdm(train_dataloader):
        # forward pass
            outputs = model(pixel_values=batch["pixel_values"].to(device),
                        multimask_output=False)

        # compute loss
        predicted_masks = outputs.pred_masks.squeeze(1)
        ground_truth_masks = batch["ground_truth_mask"].float().to(device)
        loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1)).to(device)

        # backward pass (compute gradients of parameters w.r.t. loss)
        optimizer.zero_grad()
        loss.backward()

        # optimize
        optimizer.step()
        epoch_losses.append(loss.item())

        print(f'EPOCH: {epoch}')
        print(f'Mean loss: {mean(epoch_losses)}')
        losses.append(mean(epoch_losses))

        # save checkpoint
        if epoch+1 % 10 == 0:
            torch.save(model.state_dict(), 'sam_model_'+str(epoch+1)+'.pt')
    
    # save final model
    torch.save(model.state_dict(), 'sam_model_final.pt')

    # save plot of the loss
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.set(style='darkgrid')
    sns.lineplot(losses,  markers=True, linewidth = 3.5)

    ax.xaxis.set_major_locator(plt.MaxNLocator(10))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))

    plt.yticks(size = 20)
    plt.xticks(size = 20)

    plt.ylabel('Loss', size = 35)
    plt.xlabel('Epochs', size = 35)

    plt.savefig('loss_plot.png')

