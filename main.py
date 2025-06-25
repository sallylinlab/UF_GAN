#!/usr/bin/env python
# coding: utf-8

__author__ = "Ratu, Adityano Widiata. Lin, Chia-Yu."
__copyright__ = "Copyright 2023, The Yuan Ze University"
__version__ = "1.0.0"
__maintainer__ = "Ratu, Adityano Widiata"
__email__ = "ratu.adityano@gmail.com, sallylin0121@ncu.edu.tw"
__status__ = "Experimental"

"""
Hi, This is Adit.
Congratulations, you can access the code based on Few-shot GAN.
This is the first few-shot model to be applied to anomaly detection, especially mura detection.
This code based consists of 3 main stages:
    1. Parameters initialization - @parameter-declaration
    2. Training stage - @training-here
    3. Testing stage - @testing-here
To make exploration easier, you can copy @stage-name (ex: @parameter-declaration) 
than paste it into the find section to go to the stage you want.
"""
import tensorflow as tf

from tqdm import tqdm
import numpy as np
import random
import gc

from sklearn.metrics import f1_score

from datetime import datetime
import pprint

from models.resnext import build_resnext50_unet
from models.discriminator import build_discriminator
from models.loss_func import MultiFeatureLoss, CharbonnierLoss
from models.data_augmentation import data_augmentation_layers, \
    crop_left_right, enhance_image, sliding_crop_and_select_one
from models.eval_test import checking_gen_disc

from utility.drawer_plot import roc, plot_confusion_matrix, plot_epoch_result, plot_anomaly_score, \
    write_data_analytics, write_settings, get_tnr_tpr_custom, write_result_dict, write_main_result
from utility.config_args import get_arguments
from utility.dataloaders import read_data_with_labels, get_filename, get_ori_size_from_image, get_the_best_weight

"""
@parameter-declaration
Global parameters initialization.
"""

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        if len(gpus) > 1:
            tf.config.set_visible_devices(gpus[1], 'GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)

parser = get_arguments()
args = vars(parser.parse_args())

IMG_H = args["SIZE"]
IMG_W = args["SIZE"]
# Change this to 1 for grayscale.
IMG_C = 3

st_size = args["STEP_SLIDING"]

# DATASET_NAME = "mura"
DATASET_NAME = args["DATASET_NAME"]
# NO_DATASET = 0 # 0=0-999 images, 1=1000-1999, 2=2000-2999 so on
NO_DATASET = args["NO_DATASET"]

BETA_CONTRAST = 0.1

KER_SIZE = args["KERNEL_SIZE"]

args["BETA_CONTRAST"] = BETA_CONTRAST

AUTOTUNE = tf.data.AUTOTUNE

LIMIT_EVAL_IMAGES = "100"
LIMIT_TEST_IMAGES = "MAX"
LIMIT_TRAIN_IMAGES = str(args["SHOTS"])

TRAINING_DURATION = None

NUMBER_IMAGES_SELECTED = 1000

evaluation_interval = args["EVAL_INTERVAL"]

# range between 0-1
anomaly_weight = args["ANOMALY_WEIGHT"]

learning_rate = args["LEARNING_RATE"]
meta_step_size = args["META_STEP_SIZE"]

inner_batch_size = args["BATCH_SIZE"]
eval_batch_size = args["BATCH_SIZE"]

meta_iters = args["META_ITERS"]
inner_iters = args["INNER_ITERS"]

train_shots = 20
shots = args["SHOTS"]
classes = 1
n_shots = shots
if shots > 20:
    n_shots = "few"

PERCENTAGE_COMPOSITION_DATASET = {
    "top": 70,
    "mid": 20,
    "bottom": 10
}

mode_colour = str(IMG_H) + "_rgb"
if IMG_C == 1:
    mode_colour = str(IMG_H) + "_gray"

SAVED_MODEL_DIR = args["SAVED_MODEL_DIR"]
name_model = f"{mode_colour}_{DATASET_NAME}_{NO_DATASET}_" \
             f"resnext50_{n_shots}_shots_mura_detection_{str(meta_iters)}"
g_model_path = f"{SAVED_MODEL_DIR}/{name_model}_g_model.h5"
d_model_path = f"{SAVED_MODEL_DIR}/{name_model}_d_model.h5"
result_folder = args["RESULT_DIR"]

TRAIN = args["MODE"]

TRAIN_DATA_DIR = args["TRAIN_DATA"]
EVAL_DATA_DIR = args["EVAL_DATA"]
TEST_DATA_DIR = args["TEST_DATA"]

ROOT_DATA_FOLDER = args["ROOT_DATA_DIR"]
train_data_path = f"{ROOT_DATA_FOLDER}/{DATASET_NAME}/{TRAIN_DATA_DIR}"
eval_data_path = f"{ROOT_DATA_FOLDER}/{DATASET_NAME}/{EVAL_DATA_DIR}"
test_data_path = f"{ROOT_DATA_FOLDER}/{DATASET_NAME}/{TEST_DATA_DIR}"

ori_size = get_ori_size_from_image(train_data_path, IMG_C)
print("Ori size of images: ", ori_size)
ws = args["WIN_SIZE"]
win_size = (ws, ws)

write_settings(args, name_model, result_folder)
pprint.pprint(args, depth=1)

"""Declare all loss function that we will use"""

# L1 Loss
mae = tf.keras.losses.MeanAbsoluteError()
# L2 Loss
mse = tf.keras.losses.MeanSquaredError()

# multi feature loss
multimse = MultiFeatureLoss()
# charbonnier loss
charbonnier = CharbonnierLoss()


def prep_stage(x):
    """apply a image processing method

    Args:
        x (tensor): an image.

    Returns:
        x (tensor): an updated image.
    """
    # enhance the contrast
    x = enhance_image(x, BETA_CONTRAST)
    return x


def post_stage(x):
    """apply data augmentation method

        Args:
            x (tensor): an image.

        Returns:
            x (tensor): a final updated image.
        """
    x = tf.image.resize(x, (IMG_H, IMG_W))

    x = tf.cast(x, tf.float32)

    ''' normalize to the range -1,1 '''
    x = (x - 127.5) / 127.5

    ''' normalize to the range 0-1 '''
    # x /= 255.0
    return x


@tf.function
def extraction(image, label):
    """load the images for training dataset

    Args:
        image (String): the path of the image.
        label (int64): numeric class.

    Returns:
        img (tensor): the image
        label (label): numeric class.
    """
    # This function will shrink the Omniglot images to the desired size,
    # scale pixel values and convert the RGB image to grayscale
    img = tf.io.read_file(image)
    img = tf.io.decode_png(img, channels=IMG_C)
    img = prep_stage(img)

    img = sliding_crop_and_select_one(img, stepSize=st_size, windowSize=win_size, oriSize=ori_size)

    img = post_stage(img)

    return img, label


@tf.function
def extraction_test(image, label):
    """load the images for testing dataset
    Args:
        image (String): the path of the image.
        label (int64): numeric class.

    Returns:
        l_img (tensor): the left part of the image
        r_img (tensor): the right part of the image
        label (label): numeric class.
        image (label): the path of the image.
    """
    # This function will shrink the Omniglot images to the desired size,
    # scale pixel values and convert the RGB image to grayscale
    img = tf.io.read_file(image)
    img = tf.io.decode_png(img, channels=IMG_C)
    img = prep_stage(img)

    l_img, r_img = crop_left_right(img, oriSize=ori_size)

    l_img = post_stage(l_img)
    r_img = post_stage(r_img)

    return l_img, r_img, label, image


class Dataset:
    """load the images from the folder than producing the dataset."""

    def __init__(self, path_file, training=True, limit="100"):
        """
        Args:
            path_file (String): The path of training images folder.
            training (boolean): The flag for the purpose of the function
            limit (NumericString): The limit of proceed images.
        """

        ds_start_time = datetime.now()
        self.data = {}
        class_names = ["normal"] if training else ["normal", "defect"]
        filenames, labels = read_data_with_labels(path_file, class_names, training, limit,
                                                  number_images_selected=NUMBER_IMAGES_SELECTED,
                                                  no_dataset=NO_DATASET,
                                                  percentage_composition=PERCENTAGE_COMPOSITION_DATASET)

        ds = tf.data.Dataset.from_tensor_slices((filenames, labels))
        self.ds = ds.shuffle(buffer_size=1024, seed=random.randint(123, 10000))

        if training:
            for image, label in ds.map(extraction):
                image = image.numpy()
                label = str(label.numpy())
                if label not in self.data:
                    self.data[label] = []
                self.data[label].append(image)
            self.labels = list(self.data.keys())

        ds_end_time = datetime.now()

        print("Loading Dataset and Preprocessing. ".center(100, "="))
        print("Classes: ", class_names)
        print(f"Duration of counting std and mean of images: {ds_end_time - ds_start_time}".center(100, "="))

    def get_mini_dataset(
            self, batch_size, repetitions, shots, num_classes
    ):
        """Get mini dataset for training dataset.
        Args:
            batch_size (int64): batch size
            repetitions (int64): the number of duplication of each image
            shots (int64): number of training images.
            num_classes (int64): number of classes

        Returns:
            dataset (Dataset<tensor>): dataset for training.
        """

        temp_labels = np.zeros(shape=(num_classes * shots))
        temp_images = np.zeros(shape=(num_classes * shots, IMG_H, IMG_W, IMG_C))

        # Get a random subset of labels from the entire label set.
        label_subset = random.choices(self.labels, k=num_classes)

        for class_idx, class_obj in enumerate(label_subset):
            # Use enumerated index value as a temporary label for mini-batch in
            # few shot learning.
            temp_labels[class_idx * shots: (class_idx + 1) * shots] = class_idx
            # For each index in the randomly selected label_subset, sample the
            # necessary number of images.
            temp_images[
            class_idx * shots: (class_idx + 1) * shots
            ] = random.choices(self.data[label_subset[class_idx]], k=shots)

        dataset = tf.data.Dataset.from_tensor_slices(
            (temp_images.astype(np.float32), temp_labels.astype(np.int32))
        )
        dataset = dataset.shuffle(shots, seed=int(round(datetime.now().timestamp()))).batch(batch_size).repeat(
            repetitions)

        return dataset

    def get_dataset(self, batch_size):
        """produce the dataset for testing

        Args:
            batch_size (int64): batch size

        Returns:
            ds (Dataset<tensor>): dataset for testing.
        """
        ds = self.ds.map(extraction_test, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return ds


"""Load the models and dataset."""

input_shape = (IMG_H, IMG_W, IMG_C)
inputs = tf.keras.layers.Input(input_shape, name="input_1")
data_aug = data_augmentation_layers()

# load the model
d_model = build_discriminator(inputs, IMG_H)
g_model = build_resnext50_unet(input_shape, IMG_H, IMG_C, kersize=KER_SIZE)

d_model.compile()
g_model.compile()

# load dataset
train_dataset = Dataset(train_data_path, training=True, limit=LIMIT_TRAIN_IMAGES)

eval_dataset = Dataset(eval_data_path, training=False, limit=LIMIT_EVAL_IMAGES)
eval_ds = eval_dataset.get_dataset(1)

"""Setup optimizers."""

g_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5, beta_2=0.999)
d_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5, beta_2=0.999)

print(f"Loading Datasets: {name_model} ".center(100, "="))

ADV_REG_RATE_LF = 1
REC_REG_RATE_LF = 50
FEAT_REG_RATE_LF = 1


def calculate_a_score(c_g_model, c_d_model, c_image):
    """Calculating the anomaly score of the image.

    Args:
        c_g_model (Model<Any>): model of generator.
        c_d_model (Model<Any>): model of discriminator.
        c_image (Tensor): Image for testing

    Returns:
        c_score (Float64): Anomaly score of the image
        c_loss_rec (Float64): reconstruction value of the image
        c_loss_rec (Float64): feature score of the image

    """

    reconstructed_images = c_g_model(c_image, training=False)

    feature_real, label_real = c_d_model(c_image, training=False)

    feature_fake, label_fake = c_d_model(reconstructed_images, training=False)

    # Loss 2: RECONSTRUCTION loss (L1)
    c_loss_rec = mae(c_image, reconstructed_images)

    # Loss 3: Multi feature loss (L1)
    c_loss_feat = multimse(feature_real, feature_fake, FEAT_REG_RATE_LF)

    c_score = (anomaly_weight * c_loss_rec) + ((1 - anomaly_weight) * c_loss_feat)

    return c_score, c_loss_rec, c_loss_feat


def testing(g_model_test, d_model_test, g_filepath, d_filepath, test_ds):
    """Run prediction for the model of generator and discriminator.

    Args:
        g_model_test (Model): the model of discriminator.
        d_model_test (Model): the model of discriminator.
        g_filepath (String): the path of generator model weight.
        d_filepath (String): the path of discriminator model weight.
        test_ds (Dataset<tensor>): list of image for testing.
    Produces:
        plots (matplotlib.plot): plots of testing performance in png format.
        text (.txt): output of prediction result in text file format.
    """

    class_names = ["normal", "defect"]  # normal = 0, defect = 1

    print(f"Start Testing: {name_model} ".center(100, "="))

    test_start_time = datetime.now()

    g_model_test.load_weights(g_filepath)
    d_model_test.load_weights(d_filepath)

    test_scores_ano = []
    test_real_label = []
    filepath_list = []

    """" Left-Right Mode """
    for test_left_images, test_right_images, test_labels, filepath \
            in tqdm(test_ds, desc='testing stage, left-right method.'):
        test_l_score, _, _ = calculate_a_score(g_model_test, d_model_test,
                                               test_left_images)
        test_r_score, _, _ = calculate_a_score(g_model_test, d_model_test,
                                               test_right_images)

        test_score = max(
            test_l_score.numpy(), test_r_score.numpy()
        )
        test_scores_ano = np.append(test_scores_ano, test_score)
        test_real_label = np.append(test_real_label, test_labels)
        filen = filepath.numpy()[0]
        filepath_list = np.append(filepath_list, get_filename(filen))

    ''' Scale scores vector between [0, 1]'''
    test_scores_ano = np.where(np.isnan(test_scores_ano), 0, test_scores_ano)
    test_scores_ano = (test_scores_ano - test_scores_ano.min()) / (test_scores_ano.max() - test_scores_ano.min())
    test_auc_out, test_threshold = roc(test_real_label, test_scores_ano, name_model, result_folder=result_folder)
    test_threshold = 0.7
    print("auc: ", test_auc_out)
    # print("threshold: ", test_threshold)

    # histogram distribution of anomaly scores
    plot_anomaly_score(test_scores_ano, test_real_label, "anomaly_score_dist", name_model, result_folder=result_folder)

    scores_ano_final = (test_scores_ano > test_threshold).astype(float)

    test_cm = tf.math.confusion_matrix(
        labels=test_real_label,
        predictions=scores_ano_final
    ).numpy()

    T_TP = test_cm[1][1]
    T_FP = test_cm[0][1]
    T_FN = test_cm[1][0]
    T_TN = test_cm[0][0]

    print(
        "model saved. TP %d:, FP=%d, FN=%d, TN=%d" % (T_TP, T_FP, T_FN, T_TN)
    )

    plot_confusion_matrix(test_cm, class_names, title=name_model, result_folder=result_folder)

    diagonal_sum = test_cm.trace()
    sum_of_all_elements = test_cm.sum()

    test_end_time = datetime.now()
    TESTING_DURATION = test_end_time - test_start_time
    print(f'Duration of Testing: {test_end_time - test_start_time}')

    min_tnr = 0.9
    m_th, m_tnr, m_tpr = get_tnr_tpr_custom(test_real_label, test_scores_ano, min_tnr)

    field_names = ['ModelSpec', 'shot',
                   'AUC', 'Threshold', 'Acc', 'FAR',
                   'TNR', 'PPV', 'TPR', 'NPV', 'F1',
                   'M_Threshold', 'M_TNR', 'M_TPR',
                   'TrainDur', 'TestDur', 'datetime'
                   ]

    arr_result = {
        "ModelSpec": name_model,
        "shot": shots,
        "AUC": round(test_auc_out, 3),
        "Threshold": test_threshold,
        "Acc": round((diagonal_sum / sum_of_all_elements), 3),
        "FAR": round((T_FP / (T_FP + T_TN)), 3),
        "TNR": round((T_TN / (T_FP + T_TN)), 3),
        "PPV": round((T_TP / (T_TP + T_FP)), 3),
        "TPR": round((T_TP / (T_TP + T_FN)), 3),
        "NPV": round((T_TN / (T_FN + T_TN)), 3),
        "F1": round((f1_score(test_real_label, scores_ano_final)), 3),
        "M_Threshold": round(m_th, 3),
        "M_TNR": round(m_tnr, 3),
        "M_TPR": round(m_tpr, 3),
        "TrainDur": TRAINING_DURATION,
        "TestDur": TESTING_DURATION,
        "datetime": test_end_time.strftime("%d/%m/%Y %H:%M:%S")
    }

    [print(key, ':', value) for key, value in arr_result.items()]

    write_result_dict(arr_result, name_model, result_folder=result_folder)

    write_main_result(arr_result, field_names, name_model, result_folder=result_folder)

    final_dict_list = {
        "preds": scores_ano_final,
        "scores": test_scores_ano,
        "path": filepath_list,
        "true": test_real_label
    }

    write_data_analytics(name_model, final_dict_list, result_folder=result_folder)


@tf.function
def train_step(real_images, old_disc):
    """Training the model of generator and discriminator.

        Args:
            old_disc (Model): the second level of discriminator.
            real_images (Dataset<tensor>): list of image for training.
        Produces:
            d_model (Model): Updated model weight of discriminator
            g_model (Model): Updated model weight of generator
        Returns:
            gen_loss (float64): Integer difference in days.
            disc_loss (float64): Integer difference in days.
            train_loss_rec (float64): Integer difference in days.
            train_loss_feat (float64): Integer difference in days.
    """

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        """First level Calculating Loss"""

        # apply augmentation method  
        augmented_images = data_aug(real_images)

        reconstructed_images = g_model(augmented_images, training=True)

        feature_real, label_real = d_model(augmented_images, training=True)

        feature_fake, label_fake = d_model(reconstructed_images, training=True)

        discriminator_fake_average_out = tf.math.reduce_mean(label_fake, axis=0)
        discriminator_real_average_out = tf.math.reduce_mean(label_real, axis=0)
        real_fake_ra_out = label_real - discriminator_fake_average_out
        fake_real_ra_out = label_fake - discriminator_real_average_out
        epsilon = 0.000001

        # Loss 1: 
        # use relativistic average loss binary cross entrophy
        loss_gen_ra = -(
                tf.math.reduce_mean(
                    tf.math.log(
                        tf.math.sigmoid(fake_real_ra_out) + epsilon), axis=0
                ) + tf.math.reduce_mean(tf.math.log(1 - tf.math.sigmoid(real_fake_ra_out) + epsilon), axis=0)
        )

        loss_disc_ra = -(
                tf.math.reduce_mean(
                    tf.math.log(
                        tf.math.sigmoid(real_fake_ra_out) + epsilon), axis=0
                ) + tf.math.reduce_mean(tf.math.log(1 - tf.math.sigmoid(fake_real_ra_out) + epsilon), axis=0)
        )

        # Loss 2: RECONSTRUCTION loss (L1)
        train_loss_rec = mae(augmented_images, reconstructed_images)

        # Loss 3: FEATURE Loss
        train_loss_feat = mse(feature_real[-1], feature_real[-1])

        gen_loss_inner = tf.reduce_mean(
            (loss_gen_ra * ADV_REG_RATE_LF)
            + (train_loss_rec * REC_REG_RATE_LF)
            + (train_loss_feat * FEAT_REG_RATE_LF)
        )

        disc_loss_inner = tf.reduce_mean(
            (loss_disc_ra * ADV_REG_RATE_LF)
            + (train_loss_rec * REC_REG_RATE_LF)
        )

        """Second Level Calculating Loss"""

        feat_real_2nd, lbl_real_2nd = old_disc(augmented_images, training=False)
        # print(generated_images.shape)
        feat_fake_2nd, lbl_fake_2nd = old_disc(reconstructed_images, training=False)

        disc_fake_average_out = tf.math.reduce_mean(lbl_fake_2nd, axis=0)
        disc_real_average_out = tf.math.reduce_mean(lbl_real_2nd, axis=0)
        real_fake_ra_out = lbl_real_2nd - disc_fake_average_out
        fake_real_ra_out = lbl_fake_2nd - disc_real_average_out
        epsilon = 0.000001

        # Loss 1: 
        # use relativistic average loss binary cross entrophy
        loss_gen_ra = -(
                tf.math.reduce_mean(
                    tf.math.log(
                        tf.math.sigmoid(fake_real_ra_out) + epsilon), axis=0
                ) + tf.math.reduce_mean(tf.math.log(1 - tf.math.sigmoid(real_fake_ra_out) + epsilon), axis=0)
        )

        loss_disc_ra = -(
                tf.math.reduce_mean(
                    tf.math.log(
                        tf.math.sigmoid(real_fake_ra_out) + epsilon), axis=0
                ) + tf.math.reduce_mean(tf.math.log(1 - tf.math.sigmoid(fake_real_ra_out) + epsilon), axis=0)
        )

        # Loss 2: RECONSTRUCTION loss (L1)
        train_loss_rec = charbonnier(augmented_images, reconstructed_images)

        # Loss 3: FEATURE Loss
        train_loss_feat = multimse(feat_real_2nd, feat_fake_2nd, FEAT_REG_RATE_LF)

        gen_loss_outer = tf.reduce_mean(
            (loss_gen_ra * ADV_REG_RATE_LF)
            + (train_loss_rec * REC_REG_RATE_LF)
            + train_loss_feat
        )

        disc_loss_outer = tf.reduce_mean(
            (loss_disc_ra * ADV_REG_RATE_LF)
            + (train_loss_rec * REC_REG_RATE_LF)
        )

        """ Apply Final Loss """

        gen_loss = (gen_loss_inner + gen_loss_outer) / 2

        disc_loss = (disc_loss_inner + disc_loss_outer) / 2

    gradients_of_discriminator = disc_tape.gradient(disc_loss, d_model.trainable_variables)
    gradients_of_generator = gen_tape.gradient(gen_loss, g_model.trainable_variables)

    d_optimizer.apply_gradients(zip(gradients_of_discriminator, d_model.trainable_variables))
    g_optimizer.apply_gradients(zip(gradients_of_generator, g_model.trainable_variables))

    return gen_loss, disc_loss, train_loss_rec, train_loss_feat


"""Training stage starts here.
@training-here

Requires:
    train_dataset (Dataset<tensor>): list of image from train dataset.
    eval_ds (Dataset<tensor>): list of image from evaluation dataset.
    g_model (Model<Any>): model of generator.
    d_model (Model<Any>): model of discriminator.

Produces:
    g_model_path (str): path of generator model weight.
    d_model_path (str): path of discriminator model weight.
    plots (matplotlib.plot): plots of training performance.
        
"""

gen_loss_list = []
disc_loss_list = []
iter_list = []
auc_list = []
rec_loss_list = []
feat_loss_list = []

if TRAIN:

    standard_auc = 0.7
    best_auc = standard_auc

    progbar = tf.keras.utils.Progbar(meta_iters)

    print(f" Start Training: {name_model} ".center(100, "="))
    start_time = datetime.now()

    for meta_iter in range(meta_iters):

        frac_done = meta_iter / meta_iters
        cur_meta_step_size = (1 - frac_done) * meta_step_size

        # Temporarily save the weights from the model.
        old_generator = g_model
        old_discriminator = d_model
        d_old_vars = d_model.get_weights()
        g_old_vars = old_generator.get_weights()

        # Get a sample from the full dataset.
        mini_dataset = train_dataset.get_mini_dataset(
            inner_batch_size, inner_iters, train_shots, classes
        )

        gen_loss_out = 0.0
        disc_loss_out = 0.0
        rec_loss_out = 0.0
        feat_loss_out = 0.0

        for images, _ in mini_dataset:
            g_loss, d_loss, rec_loss, feat_loss = train_step(images, old_discriminator)
            gen_loss_out = g_loss
            disc_loss_out = d_loss
            rec_loss_out = rec_loss
            feat_loss_out = feat_loss

        d_new_vars = d_model.get_weights()
        g_new_vars = g_model.get_weights()

        # Perform SGD for the meta step.           
        for var in range(len(d_new_vars)):
            d_new_vars[var] = d_old_vars[var] + (
                    (d_new_vars[var] - d_old_vars[var]) * cur_meta_step_size
            )

        for var in range(len(g_new_vars)):
            g_new_vars[var] = g_old_vars[var] + (
                    (g_new_vars[var] - g_old_vars[var]) * cur_meta_step_size
            )

        # After the meta-learning step, reload the newly-trained weights into the model.
        g_model.set_weights(g_new_vars)
        d_model.set_weights(d_new_vars)

        # Evaluation loop

        meta_iter = meta_iter + 1

        if meta_iter % evaluation_interval == 0:

            eval_g_model = g_model
            eval_d_model = d_model

            scores_ano = []
            real_label = []

            """" Left-Right Mode """
            for left_images, right_images, labels, _ \
                    in tqdm(eval_ds, desc=f'evalution stage at {meta_iter} batch, left-right method.'):
                loss_rec, loss_feat = 0.0, 0.0
                l_score, _, _ = calculate_a_score(eval_g_model, eval_d_model, left_images)
                r_score, _, _ = calculate_a_score(eval_g_model, eval_d_model, right_images)

                score = max(
                    l_score.numpy(), r_score.numpy()
                )

                scores_ano = np.append(scores_ano, score)
                real_label = np.append(real_label, labels)

            """ for analytics """
            iter_list = np.append(iter_list, meta_iter)
            gen_loss_list = np.append(gen_loss_list, gen_loss_out)
            disc_loss_list = np.append(disc_loss_list, disc_loss_out)
            rec_loss_list = np.append(rec_loss_list, rec_loss_out)
            feat_loss_list = np.append(feat_loss_list, feat_loss_out)

            auc_out = 0.0
            threshold = 0.0

            try:
                '''Scale scores vector between [0, 1]'''
                scores_ano = np.where(np.isnan(scores_ano), 0, scores_ano)
                scores_ano = (scores_ano - scores_ano.min()) / (scores_ano.max() - scores_ano.min())
                auc_out, threshold = roc(real_label, scores_ano, name_model, draw_plot=False)
            except Exception as e:
                print("all data is Nan. Model doesnt work.")
                pass

            auc_list = np.append(auc_list, auc_out)
            scores_ano = (scores_ano > threshold).astype(int)

            cm = tf.math.confusion_matrix(labels=real_label, predictions=scores_ano).numpy()
            TP = cm[1][1]
            FP = cm[0][1]
            FN = cm[1][0]
            TN = cm[0][0]

            print(
                f"model saved. batch {meta_iter}:, AUC={auc_out:.3f}, TP={TP}, TN={TN}, FP={FP}, FN={FN},"
                f" Gen Loss={gen_loss_out:.5f}, Disc Loss={disc_loss_out:.5f}"
            )

            if auc_out >= best_auc or auc_out > standard_auc:
                print(
                    f"the best model saved. at batch {meta_iter}: with AUC={auc_out:.3f}"
                )

                best_g_model_path = g_model_path.replace(".h5", f"_best_{meta_iter}_{auc_out:.2f}.h5")
                best_d_model_path = d_model_path.replace(".h5", f"_best_{meta_iter}_{auc_out:.2f}.h5")
                g_model.save(best_g_model_path)
                d_model.save(best_d_model_path)
                best_auc = auc_out

            # save model's weights
            g_model.save(g_model_path)
            d_model.save(d_model_path)

        progbar.update(meta_iter)

    end_time = datetime.now()
    TRAINING_DURATION = end_time - start_time

    print(f'Duration of Training: {end_time - start_time}')

    """
    Train ends and generate plot to show the loss performance.
    """
    plot_epoch_result(iter_list, gen_loss_list, "Generator_Loss", name_model, "g", result_folder=result_folder)
    plot_epoch_result(iter_list, disc_loss_list, "Discriminator_Loss", name_model, "r", result_folder=result_folder)
    plot_epoch_result(iter_list, feat_loss_list, "Feature_Loss", name_model, "y", result_folder=result_folder)
    plot_epoch_result(iter_list, rec_loss_list, "Reconstructed_Loss", name_model, "c", result_folder=result_folder)
    plot_epoch_result(iter_list, auc_list, "AUC_Score", name_model, "b", result_folder=result_folder)

"""Testing stage start here
@testing-here

Requires:
    test_dataset (Dataset<tensor>): list of image from test dataset.
    g_model_path (str): path of generator model weight.
    d_model_path (str): path of discriminator model weight.
    best_g_model (boolean): use the best model (default: true)

Produces:
    plots (matplotlib.plot): plots of testing performance.
    text (.txt): output of prediction result.
    
"""
# load test dataset
test_dataset = Dataset(test_data_path, training=False, limit=LIMIT_TEST_IMAGES)

# load the best weight model
if args["BEST_MODEL_WEIGHT"]:
    best_g_model, best_d_model = get_the_best_weight(SAVED_MODEL_DIR, IMG_H)

    if best_g_model is not None:
        print(f"Choose the best generator model: {best_g_model} ".center(150, "="))
        g_model_path = f"{SAVED_MODEL_DIR}/{best_g_model}"

    if best_d_model is not None:
        print(f"Choose the best discriminator model: {best_d_model} ".center(150, "="))
        d_model_path = f"{SAVED_MODEL_DIR}/{best_d_model}"

# run testing function
testing(g_model, d_model, g_model_path, d_model_path, test_dataset.get_dataset(1))

# produce reconstructed image to verify the quality of generator.
checking_gen_disc(name_model, g_model, d_model, g_model_path, d_model_path, test_data_path, IMG_H=IMG_H, IMG_W=IMG_W,
                  IMG_C=IMG_C, result_folder=result_folder)

# clear the session and cache after training and testing end.
tf.keras.backend.clear_session()
gc.collect()
