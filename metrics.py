import numpy as np

# Dice Coefficient
# comparing the similarity of two samples. The
# Dice Coefficient is twice the area of overlap of the two segmentations
# divided by the total number of pixels in both images
def dice(mask1, mask2):
    # 2*TP
    tp = 2*np.count_nonzero(mask1 == mask2)
    # FP
    fp = np.sum((mask1 == 1) & (mask2 == 0))
    # FN
    fn = np.sum((mask1 == 0) & (mask2 == 1))
    return tp / (tp+fp+fn)


# Intersection Over Union
# area of overlap divided by the area of
# the union of the predicted and ground truth segmentation.
def IoU(mask1, mask2):
    mask1_area = np.count_nonzero(mask1 == 1)       
    mask2_area = np.count_nonzero(mask2 == 1)
    intersection = np.count_nonzero( np.logical_and( mask1, mask2) )
    iou = intersection/(mask1_area+mask2_area-intersection)
    return iou



# Pixel Accuracy
# simplest used metric and it measures the
# percentage of pixels that were accurately classified
def pixel_accuracy(mask1, mask2):
    correct = np.count_nonzero(mask1 == mask2)
    total = mask1.shape[0] * mask1.shape[1]
    return correct / total