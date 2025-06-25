import numpy as np
import tensorflow as tf
from glob import glob
from matplotlib import pyplot as plt


def checking_gen_disc(mode, g_model_inner, d_model_inner, g_filepath, d_filepath, test_data_path, IMG_H=128, IMG_W=128, IMG_C=3,result_folder=""):

    print(" Checking Reconstructed Image. ".center(100, "="))
    
    g_model_inner.load_weights(g_filepath)
    d_model_inner.load_weights(d_filepath)
    
    normal_image = glob(test_data_path+"/normal/*.png")[0]
    defect_image = glob(test_data_path+"/defect/*.png")[0]
    paths = {
        "normal": normal_image,
        "defect": defect_image,
    }
    f = open(result_folder + mode + '_quality_score.txt', 'w+')
    
    for label, img_path in paths.items():
        print(label, img_path)

        rows = 1
        cols = 3
        axes=[]
        fig = plt.figure()

        img = tf.io.read_file(img_path)
        img = tf.io.decode_png(img, channels=IMG_C)
        img = tf.image.resize(img, (IMG_H, IMG_W))

        axes.append( fig.add_subplot(rows, cols, 1) )
        axes[-1].set_title('_original_')
        
        real_image = img.numpy().astype(np.int64)
        plt.imshow(real_image, alpha=1.0)
        plt.axis('off')

        img = tf.cast(img, tf.float32)

        ''' normalize to the range -1,1 '''
        img = (img - 127.5) / 127.5
        # ''' normalize to the range 0,1 '''
        # img /= 255.0
        
        image = tf.reshape(img, (-1, IMG_H, IMG_W, IMG_C))
        reconstructed_images = g_model_inner.predict(image)
        reconstructed_images = reconstructed_images * 127 + 127
        reconstructed_images = tf.reshape(reconstructed_images, (IMG_H, IMG_W, IMG_C))
        axes.append( fig.add_subplot(rows, cols, 3) )
        axes[-1].set_title('_reconstructed_')
        
        fake_image = reconstructed_images.numpy().astype(np.int64)
        plt.imshow(fake_image, alpha=1.0)
        plt.axis('off')

        fig.tight_layout()
        fig.savefig(result_folder + mode + '_'+ label +'.png')
        plt.show()
        plt.clf()
        # L1 Loss
        mae = tf.keras.losses.MeanAbsoluteError()
        # L2 Loss
        mse = tf.keras.losses.MeanSquaredError()
        
        mse_score = mse(real_image, fake_image)
        mae_score = mae(real_image, fake_image)
        ssim_score = tf.image.ssim(real_image, fake_image, max_val=255)

        print(f'label: {label}', file=f)
        print(f'image_path: {img_path}', file=f)
        print(f'mse_score: {mse_score}', file=f)
        print(f'ssim_score: {ssim_score}', file=f)
        print(f'mae_score: {mae_score}', file=f)
        
    f.close()
    