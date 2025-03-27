from torch.utils.data import Dataset
import numpy as np

class SAMDataset(Dataset):
  def __init__(self, dataset, processor):
    self.dataset = dataset
    self.processor = processor

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    item = self.dataset[idx]
    image = item["img"]
    ground_truth_mask = np.array(item["gt"])

    # get bounding box prompt
    prompt = 'field'

    # prepare image and prompt for the model
    inputs = self.processor(image, input_text=prompt, return_tensors="pt")

    # remove batch dimension which the processor adds by default
    inputs = {k:v.squeeze(0) for k,v in inputs.items()}

    # add ground truth segmentation
    inputs["ground_truth_mask"] = ground_truth_mask

    return inputs