import glob
import cv2 
import numpy as np
import os
import pprint
import random

path_data = "..\\data\\classification\\train1000\\*\\*.*"

def generating_pairs(path):

    all_address = [address for address in glob.glob(path)]
    image_categories = np.array([category.split("\\")[-2] for category in glob.glob(path)])
    all_current_samples = []
    all_current_pair_samples = []
    all_current_labels = []
    for image_address in all_address:

        image_category = image_address.split("\\")[-2]
        current_img = cv2.imread(image_address)
        same_imgs_indexs = np.where(image_categories == image_category)[0]
        same_img_index = random.choice(same_imgs_indexs)
        same_img = cv2.imread(all_address[same_img_index])
        same_label = 1
        all_current_samples.append(current_img)
        all_current_pair_samples.append(same_img)
        all_current_labels.append(same_label)

        different_imgs_indexs = np.where(image_categories != image_category)[0]
        different_img_index = random.choice(different_imgs_indexs)
        different_img = cv2.imread(all_address[different_img_index])
        different_label = 0
        all_current_samples.append(current_img)
        all_current_pair_samples.append(different_img)
        all_current_labels.append(different_label)

    all_current_samples = np.array(all_current_samples)
    all_current_pair_samples = np.array(all_current_pair_samples)
    all_current_labels = np.array(all_current_labels)

    return all_current_samples, all_current_pair_samples, all_current_labels

all_current_samples, all_current_pair_samples, all_current_labels = generating_pairs(path_data)


for i in range(2):

    cv2.imshow("image", all_current_samples[i])
    cv2.waitKey(0)

    cv2.imshow("image", all_current_pair_samples[i])
    cv2.waitKey(0)

    print(all_current_labels[i])

cv2.destroyAllWindows()







