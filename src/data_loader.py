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
    all_samples = []

    for image_address in all_address:

        image_category = image_address.split("\\")[-2]
        current_img = cv2.imread(image_address)
        same_imgs_indexs = np.where(image_categories == image_category)[0]
        same_img_index = random.choice(same_imgs_indexs)
        same_img = cv2.imread(all_address[same_img_index])
        same_sample_data = np.concatenate((current_img, same_img), axis=1)
        same_label = 1
        all_samples.append((same_sample_data, same_label))

        different_imgs_indexs = np.where(image_categories != image_category)[0]
        different_img_index = random.choice(different_imgs_indexs)
        different_img = cv2.imread(all_address[different_img_index])
        different_sample_data = np.concatenate((current_img, different_img), axis=1)
        different_label = 0
        all_samples.append((different_sample_data, different_label))

        all_samples = np.array(all_samples)

    return all_samples










