import math

import pandas as pd
import multiprocess as mp
import gc
import tensorflow as tf
import cv2
from sklearn.utils import shuffle


def get_number_by_percentage(percentage, whole):
    return math.ceil(float(percentage) / 100 * float(whole))


def selecting_images_preprocessing(images_path_array, limit_image_to_train="MAX", composition={}):
    """Get images for training by filtering the images based on determined composition.

    Args:
        images_path_array (Array<String>): list of absolute path of image file.
        limit_image_to_train (String): limit of images that will be proceed.
        composition (Dict<String, Int>): percentage of multiple images that need to be used based on the std value.
         (top - High std value, mid - Medium std value, bottom - low std value.)

    Returns:
        final_image_path (List<String>): list of filtered image path
        final_label (List<Int>): list of filtered image's label
    """
    final_image_path = []
    final_label = []

    def processing_image(img_data):
        img_path = img_data[0]
        label = img_data[1]
        image = cv2.imread(img_path)

        data_row = {
            "image_path": img_path,
            "mean": image.mean(),
            "std": image.std(),
            "class": label
        }
        # print(data_row)
        return data_row

    print("processed number of data: ", len(images_path_array))
    if limit_image_to_train == "MAX":
        limit_image_to_train = len(images_path_array)

    df_analysis = pd.DataFrame(columns=['image_path', 'mean', 'std', 'class'])

    # multiple processing for calculating std

    pool = mp.Pool(5)
    data_rows = pool.map(processing_image, images_path_array)

    df_analysis = df_analysis.append(data_rows, ignore_index=True)

    final_df = df_analysis.sort_values(['std', 'mean'], ascending=[True, False])

    if composition == {}:
        shuffle(final_df)
        final_image_path = final_df['image_path'].head(limit_image_to_train).tolist()
        final_label = final_df['class'].head(limit_image_to_train).tolist()
    else:
        counter_available_no_data = limit_image_to_train
        if composition.get('top') != 0:
            num_rows = get_number_by_percentage(composition.get('top'), limit_image_to_train)
            if counter_available_no_data <= num_rows:
                num_rows = counter_available_no_data
            counter_available_no_data = counter_available_no_data - num_rows

            print(composition.get('top'), num_rows, counter_available_no_data)

            # get top data
            final_image_path = final_image_path + final_df['image_path'].head(num_rows).tolist()
            final_label = final_label + final_df['class'].head(num_rows).tolist()

        if composition.get('mid') != 0:
            num_rows = get_number_by_percentage(composition.get('mid'), limit_image_to_train)
            if counter_available_no_data <= num_rows:
                num_rows = counter_available_no_data
            counter_available_no_data = counter_available_no_data - num_rows

            print(composition.get('mid'), num_rows, counter_available_no_data)

            # top & mid
            n = len(final_df.index)
            mid_n = round(n / 2)
            mid_k = round(num_rows / 2)

            start = mid_n - mid_k
            end = mid_n + mid_k

            final = final_df.iloc[start:end]
            final_image_path = final_image_path + final['image_path'].head(num_rows).tolist()
            final_label = final_label + final['class'].head(num_rows).tolist()

        if composition.get('bottom') != 0:
            num_rows = get_number_by_percentage(composition.get('bottom'), limit_image_to_train)
            if counter_available_no_data <= num_rows:
                num_rows = counter_available_no_data
            counter_available_no_data = counter_available_no_data - num_rows

            print(composition.get('bottom'), num_rows, counter_available_no_data)

            # get bottom data
            final_image_path = final_image_path + final_df['image_path'].tail(num_rows).tolist()
            final_label = final_label + final_df['class'].tail(num_rows).tolist()

    # clear zombies memory
    del [[final_df, df_analysis]]
    gc.collect()

    print("selecting_images_preprocessing Done.")
    return final_image_path, final_label


def get_ori_size(img):
    """Get the original (Height x Width) size of image.
    Args:
        img (Tensor): an image

    Returns:
        oriSize (List<int, int>): Height and width of image
    """
    oriSize = tf.shape(img).numpy()
    return oriSize


def sliding_crop_and_select_one(img, stepSize=20, windowSize=(256, 256), oriSize=(271, 481), train=True):
    """Scanning the image to find the desirable part of the image based on std value

    Args:
        img (tensor): an image
        stepSize (Int): size of step in pixel for moving the window
        windowSize (List<Int. Int>): size of sliding window
        oriSize(List<Int. Int>): the original size of image
        train: The flag for the purpose of the function

    Returns:
        current_image (tensor): a desired image [cropped]

    """
    current_std = 0
    current_image = None

    y_end_crop, x_end_crop = False, False
    for y in range(0, oriSize[0], stepSize):

        y_end_crop = False

        for x in range(0, oriSize[1], stepSize):

            x_end_crop = False

            crop_y = y
            if (y + windowSize[0]) > oriSize[0]:
                crop_y = oriSize[0] - windowSize[0]
                y_end_crop = True

            crop_x = x
            if (x + windowSize[1]) > oriSize[1]:
                crop_x = oriSize[1] - windowSize[1]
                x_end_crop = True

            image = tf.image.crop_to_bounding_box(img, crop_y, crop_x, windowSize[0], windowSize[1])
            std_image = tf.math.reduce_std(tf.cast(image, dtype=tf.float32))

            if train:
                if current_std == 0 or std_image < current_std:
                    current_std = std_image
                    current_image = image
            else:
                if current_std == 0 or std_image > current_std:
                    current_std = std_image
                    current_image = image

            if x_end_crop:
                break

        if x_end_crop and y_end_crop:
            break

    return current_image


def enhance_image(image, beta=0.1):
    """enhancing the contrast of the image

    Args:
        image (tensor): an image
        beta (float): a coefficient whose value is between 0 and 1 to adjust the image’s contrast

    Returns:
        image (tensor): an enhanced image
    """
    image = tf.cast(image, tf.float64)
    image = ((1 + beta) * image) + (-beta * tf.math.reduce_mean(image))

    return image


def crop_left_right(img, oriSize=(271, 481)):
    """generating the left and right part of image.

    Args:
        img (tensor): an image
        oriSize (List<Int. Int>): the original size of image

    Returns:
        img_left (tensor): the left part of image [cropped]
        img_right (tensor): the right part of image [cropped]
    """
    win_size = min(oriSize)

    img_left = tf.image.crop_to_bounding_box(img, 0, 0, win_size, win_size)

    img_right = tf.image.crop_to_bounding_box(img, oriSize[0] - win_size, oriSize[1] - win_size, win_size, win_size)

    return img_left, img_right


def data_augmentation_layers():
    """for enriching the data type.

    Returns:
        data_augmentation (tf.Layers): layer for data augmentation during training to enrich the data.
    """
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomTranslation(height_factor=0.10, width_factor=0.10),
        tf.keras.layers.RandomZoom(height_factor=0.10, width_factor=0.10),
    ])
    return data_augmentation
