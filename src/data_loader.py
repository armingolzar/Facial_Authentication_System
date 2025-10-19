import glob
import cv2 
import numpy as np
import os
import pprint
import random


data_folders = os.listdir('..\\data\\classification\\train1000')

# address_dict = {}

# for address in glob.glob("..\\data\\classification\\train1000\\*\\*.*"):

#     label = address.split("\\")[-2]
#     if label in address_dict:
#         address_dict[label].append(address)

#     else:
#         address_dict[label] = [address]

all_address = [address for address in glob.glob("..\\data\\classification\\train1000\\*\\*.*")]
image_categories = [category.split("\\")[-2] for category in glob.glob("..\\data\\classification\\train1000\\*\\*.*")]

all_samples = []

for image_address in all_address:

    image_category = image_address.split("\\")[-2]
    current_img = cv2.imread(image_address)
    same_imgs_indexs = np.where(image_categories == image_category)
    print(same_imgs_indexs)
    print("\n")
    same_img_address = random.choice(same_imgs_indexs)
    print(same_img_address)
    print("\n")
    same_img = cv2.imread(all_address[same_img_address])
    sample_data = np.concatenate((current_img, same_img), axis=1)
    label = 1

    all_samples.append((sample_data, label))
    cv2.imshow('image', current_img)
    cv2.waitKey(0)
    break
print(all_samples)




