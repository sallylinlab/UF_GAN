import os
import natsort
import numpy as np
import pandas as pd
import math
from tqdm import tqdm
from sklearn.utils import shuffle
from datetime import datetime
from models.data_augmentation import selecting_images_preprocessing
from glob import glob
import random
import tensorflow as tf
import tensorflow_io as tfio


def read_data_with_labels(filepath, class_names, training=True, limit="100",
                          number_images_selected=1000, no_dataset=0,
                          percentage_composition=None):
    if percentage_composition is None:
        percentage_composition = {"top": 70, "mid": 10, "bottom": 20}

    image_list = []
    label_list = []
    for class_n in class_names:  # do dogs and cats
        path = os.path.join(filepath, class_n)  # create path to dogs and cats
        class_num = class_names.index(class_n)  # get the classification  (0 or a 1). 0=dog 1=cat
        path_list = []
        class_list = []

        list_path = natsort.natsorted(os.listdir(path))

        if training:
            # print("total number of dataset", len(list_path))

            newarr_list_path = np.array_split(list_path, math.ceil(len(list_path) / number_images_selected))

            # print("number of sub dataset", len(newarr_list_path))

            list_path = newarr_list_path[no_dataset]

            # print("data taken from dataset", len(list_path))

        for img in tqdm(list_path, desc='selecting images'):
            if ".DS_Store" != img:
                # print(img)
                filpath = os.path.join(path, img)

                path_list.append(filpath)
                class_list.append(class_num)

        n_samples = len(path_list)

        if limit != "MAX" and n_samples > int(limit):
            n_samples = int(limit)

        if training:
            ''' 
            selecting by attribute of image
            '''
            combined = np.transpose((path_list, class_list))
            # print(combined)
            path_list, class_list = selecting_images_preprocessing(combined, limit_image_to_train=n_samples,
                                                                   composition=percentage_composition)
        else:
            ''' 
            random selecting
            '''
            path_list, class_list = shuffle(path_list, class_list, n_samples=n_samples,
                                            random_state=int(round(datetime.now().timestamp())))

        image_list = image_list + path_list
        label_list = label_list + class_list
    # print(image_list, label_list)
    return image_list, label_list


def get_filename(filepath, decoder=True):
    if decoder:
        filepath = filepath.decode('utf-8')
    head, tail = os.path.split(filepath)
    return tail.strip()


def read_data_with_labels_basic(filepath, class_names, limit="100"):
    image_list = []
    label_list = []
    for class_n in class_names:  # do dogs and cats
        path = os.path.join(filepath, class_n)  # create path to dogs and cats
        class_num = class_names.index(class_n)  # get the classification  (0 or a 1). 0=dog 1=cat
        path_list = []
        class_list = []

        list_path = natsort.natsorted(os.listdir(path))

        for img in tqdm(list_path, desc='selecting images basic'):

            if ".DS_Store" != img:
                # print(img)
                filpath = os.path.join(path, img)
                path_list.append(filpath)
                class_list.append(class_num)

        n_samples = len(path_list)

        if limit != "MAX" and n_samples > int(limit):
            n_samples = int(limit)

        path_list, class_list = shuffle(path_list, class_list, n_samples=n_samples,
                                        random_state=int(round(datetime.now().timestamp())))

        image_list = image_list + path_list
        label_list = label_list + class_list
    # print(image_list, label_list)
    return image_list, label_list


def get_ori_size_from_image(fpath, img_c):
    ext = "png"
    normal_image = random.choice(glob(f"{fpath}/normal/*.{ext}"))
    img = tf.io.read_file(normal_image)

    img = tf.io.decode_png(img, channels=img_c)

    height = tf.shape(img)[0]
    width = tf.shape(img)[1]

    return height.numpy(), width.numpy()


def filter_by_substr(string, substr):
    return [str for str in string if
            all(sub in str for sub in substr)]


def remove_path(models):
    output = []
    for filepath in models:
        _, tail = os.path.split(filepath)
        output.append(tail)
    return output


def get_the_best_model(models, size):
    if len(models) == 1:
        return models[0]

    models = filter_by_substr(models, ["_best_"])
    df = pd.DataFrame(columns=['name', 'epoch', 'value'])
    for model in models:

        name_model = os.path.splitext(model)[0]
        name_model = name_model.split("_")
        epoch = name_model[-2]
        value = name_model[-1]
        if size == int(name_model[0]):
            data = {
                "name": model,
                "epoch": int(epoch),
                "value": float(value),
            }

            # print(data)
            if data.get("epoch") > 300:
                df = pd.concat([df, pd.DataFrame.from_records([data])])

    try:
        first_row = \
        df.sort_values(by=['value', 'epoch'], ascending=[False, False]).head(1).to_dict(orient='records')[0]['name']
        return first_row
    except:
        return None


def get_the_best_weight(file_model_path, size=64):
    models = glob(file_model_path + "/*.h5")

    g_models = remove_path(models)
    g_models = filter_by_substr(g_models, ["_g_"])
    final_g_model = get_the_best_model(g_models, size)

    d_models = remove_path(models)
    d_models = filter_by_substr(d_models, ["_d_"])
    final_d_model = get_the_best_model(d_models, size)

    return final_g_model, final_d_model
