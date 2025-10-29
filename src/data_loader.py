import tensorflow as tf
import glob 
import numpy as np
import random


# The old version of preparing data for training
####################################################################################################

# path_data = "..\\data\\classification\\train1000\\*\\*.*"

# def generating_pairs(path):

#     all_address = [address for address in glob.glob(path)]
#     image_categories = np.array([category.split("\\")[-2] for category in glob.glob(path)])
#     all_current_samples = []
#     all_current_pair_samples = []
#     all_current_labels = []
#     for image_address in all_address:

#         image_category = image_address.split("\\")[-2]
#         current_img = cv2.imread(image_address)
#         same_imgs_indexs = np.where(image_categories == image_category)[0]
#         same_img_index = random.choice(same_imgs_indexs)
#         same_img = cv2.imread(all_address[same_img_index])
#         same_label = 1
#         all_current_samples.append(current_img.astype("float32")/255.0)
#         all_current_pair_samples.append(same_img.astype("float32")/255.0)
#         all_current_labels.append(same_label)

#         different_imgs_indexs = np.where(image_categories != image_category)[0]
#         different_img_index = random.choice(different_imgs_indexs)
#         different_img = cv2.imread(all_address[different_img_index])
#         different_label = 0
#         all_current_samples.append(current_img.astype("float32")/255.0)
#         all_current_pair_samples.append(different_img.astype("float32")/255.0)
#         all_current_labels.append(different_label)

#     all_current_samples = np.array(all_current_samples)
#     all_current_pair_samples = np.array(all_current_pair_samples)
#     all_current_labels = np.array(all_current_labels)

#     return all_current_samples, all_current_pair_samples, all_current_labels
#######################################################################################################

# The new version with GPU 
#######################################################################################################

def get_image_path_and_categories(path):

    addrs = glob.glob(path)
    cats = [category.split("\\")[-2] for category in addrs]

    return addrs, cats

def full_epoch_data_generator(addrs, cats, batch_size):

    while True: 
        
        indices = np.arange(len(addrs))
        np.random.shuffle(indices)
        samples_batch_needded = batch_size // 2

        for index in range(0, len(indices), samples_batch_needded):
            batch_indices = indices[index : index + samples_batch_needded]
            batch_samples, batch_pairs, batch_labels = [], [], []


            for sample_idx in batch_indices:
                sample_path = addrs[sample_idx]
                sample_label = cats[sample_idx]

                # positive pair
                pos_candidates = [i for i, itrator in enumerate(cats) if itrator == sample_label and i != sample_idx]
                pos_idx = random.choice(pos_candidates) if pos_candidates else sample_idx
                pos_path = addrs[pos_idx]


                # negative pair
                neg_candidates = [i2 for i2, itrator2 in enumerate(cats) if itrator2 != sample_label]
                neg_idx = random.choice(neg_candidates)
                neg_path = addrs[neg_idx]


                # appending pairs
                batch_samples.extend([sample_path, sample_path])
                batch_pairs.extend([pos_path, neg_path])
                batch_labels.extend([1.0, 0.0])

            yield (batch_samples, batch_pairs), batch_labels

def preprocess_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = tf.cast(img, tf.float32) / 255.0

    return img


def create_dataset(addrs, cats, batch_size):

    dataset = tf.data.Dataset.from_generator(
                lambda: full_epoch_data_generator(addrs, cats, batch_size),
                output_signature=((tf.TensorSpec(shape=(None,), dtype=tf.string),
                                   tf.TensorSpec(shape=(None,), dtype=tf.string)),
                                   tf.TensorSpec(shape=(None,), dtype=tf.float32))
    )


    def map_paths_to_images(sample_paths, pair_paths):
        sample_imgs = tf.map_fn(preprocess_image, sample_paths, fn_output_signature=tf.float32)
        pair_imgs = tf.map_fn(preprocess_image, pair_paths, fn_output_signature=tf.float32)

        return (sample_imgs, pair_imgs)

    dataset = dataset.map(lambda x, y: (map_paths_to_images(*x), y), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset









