# -*- coding: utf-8 -*-
from tflearn.data_utils import *
from os.path import join
import numpy as np
from skimage import io, transform
from keras.models import load_model
from skimage.color import rgb2lab, lab2rgb
import time
from functools import wraps
import warnings
from tensorflow.python.ops.image_ops import rgb_to_hsv
import tensorflow as tf
from keras import backend as K
warnings.filterwarnings("ignore")


def func_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        res = func(*args, **kwargs)
        end_time = time.time()
        print(func.__name__, end_time - start_time)
        return res
    return wrapper


class DeHaze:
    def __init__(self):
        self.inputImagsPath, self.outputImagsPath = './Test', './TestOutput'
        self.Models = {'model_L': './Model/dehaze_rough_l.h5', 'model_A': './Model/dehaze_rough_a.h5',
                       'model_B': './Model/dehaze_rough_b.h5', 'model_refine': './Model/dehaze_refine_mse_30.h5'}
        self.first_StepResize = (228, 304, 3)
        self.extensionCoef_x, self.extensionCoef_y = 6, 8

    def SSIM_Loss(self, true_y, pred_y):
        hsv_pred_yiction = rgb_to_hsv(pred_y)

        # mae_loss_loss
        mae_loss = K.mean(K.abs(pred_y - true_y), axis=-1)

        # tv_loss
        shape = tf.shape(pred_y)
        height, width = shape[1], shape[2]
        y = tf.slice(pred_y, [0, 0, 0, 0], tf.stack([-1, height - 1, -1, -1])) - tf.slice(pred_y, [0, 1, 0, 0],
                                                                                          [-1, -1, -1, -1])
        x = tf.slice(pred_y, [0, 0, 0, 0], tf.stack([-1, -1, width - 1, -1])) - tf.slice(pred_y, [0, 0, 1, 0],
                                                                                         [-1, -1, -1, -1])
        tv_loss = tf.nn.l2_loss(x) / tf.to_float(tf.size(x)) + tf.nn.l2_loss(y) / tf.to_float(tf.size(y))

        # SSIM_Loss
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        true_y = tf.transpose(true_y, [0, 2, 3, 1])
        pred_y = tf.transpose(pred_y, [0, 2, 3, 1])
        true_patches = tf.extract_image_patches(true_y, [1, 8, 8, 1], [1, 2, 2, 1], [1, 1, 1, 1], "SAME")
        pred_patches = tf.extract_image_patches(pred_y, [1, 8, 8, 1], [1, 2, 2, 1], [1, 1, 1, 1], "SAME")
        # Get mean
        true_u = K.mean(true_patches, axis=-1)
        pred_u = K.mean(pred_patches, axis=-1)
        # Get variance
        true_var = K.var(true_patches, axis=-1)
        pred_var = K.var(pred_patches, axis=-1)
        # Get std dev
        true_std = K.sqrt(true_var)
        pred_std = K.sqrt(pred_var)
        var_pred_cotrue = pred_std * true_std
        ssim = (2 * var_pred_cotrue + c2) * (2 * true_u * pred_u + c1) 
        denom =  (pred_var + true_var + c2) * (K.square(true_u) + K.square(pred_u) + c1) 
        ssim /= denom

        size = tf.size(hsv_pred_yiction)
        loss_light = tf.nn.l2_loss(hsv_pred_yiction[:, :, :, 2]) / tf.to_float(size)

        loss_total = -0.07 * loss_light + 1.0 * mae_loss - 0.0005 * tv_loss
        return loss_total

    ''' Loading dataset　'''
    @func_time
    def loadImages(self):
        self.firstStepInputImages = []
        self.inputImagesList = os.listdir(self.inputImagsPath)
        for pk, pil in enumerate(self.inputImagesList):
            img_prou = transform.resize(io.imread(join(self.inputImagsPath, pil)), self.first_StepResize)
            self.firstStepInputImages.append(rgb2lab(np.uint8(img_prou * 255.0)))
        print("Loading...Done")

    ''' Haze Removal Part '''
    @func_time
    def first_step(self):
        print('#testing images: %s' % (len(self.firstStepInputImages)))
        self.firstStepInputImages = np.reshape(self.firstStepInputImages, [-1]+list(self.first_StepResize))

        l_pres = load_model(self.Models['model_L']).predict(self.firstStepInputImages)
        a_pres = load_model(self.Models['model_A']).predict(self.firstStepInputImages)
        b_pres = load_model(self.Models['model_B']).predict(self.firstStepInputImages)

        predicts = [[l[0], l[1], a[0], a[1], b[0], b[1]] for l, a, b in zip(l_pres, a_pres, b_pres)]
        self.firstOutputImages = [self.restoreCImg(iv, predicts[ik]) for ik, iv in enumerate(self.firstStepInputImages)]

    ''' Texture Refinement Part '''
    @func_time
    def second_step(self):
        self.secondInputImages = []
        for pk, pil in enumerate(self.firstOutputImages):
            imgc = np.pad(np.reshape(pil, newshape=[-1]+list(pil.shape)), [[0, 0], [self.extensionCoef_x, self.extensionCoef_x], [self.extensionCoef_y, self.extensionCoef_y], [0, 0]], mode='reflect')
            self.secondInputImages.append(np.reshape(imgc, newshape=list(imgc.shape[1:4])) / 255.0)

        model = load_model(self.Models['model_refine'], custom_objects={'SSIM_Loss': self.SSIM_Loss})
        img_prous = np.reshape(self.secondInputImages, [-1]+(list(self.secondInputImages[0].shape)))
        self.secondOutputImages = np.clip(model.predict(img_prous), 0, 1)
        [io.imsave(join(self.outputImagsPath, self.inputImagesList[ik]), iv[self.extensionCoef_x:iv.shape[0] - self.extensionCoef_x, self.extensionCoef_y:iv.shape[1] - self.extensionCoef_y, :]) for ik, iv in enumerate(self.secondOutputImages)]

    ''' Color Transfer '''
    def restoreCImg(self, haze_img_lab=None, avg_stds=None):
        pre_img = np.zeros(haze_img_lab.shape)
        avg_clean, std_clean = np.zeros([3]), np.zeros([3])
        avg_haze, std_haze = np.zeros([3]), np.zeros([3])
        for channel in range(3):
            avg_clean[channel], std_clean[channel] = avg_stds[channel * 2], avg_stds[channel * 2 + 1]
            avg_haze[channel], std_haze[channel] = np.mean(haze_img_lab[:, :, channel]), np.std(haze_img_lab[:, :, channel])
            pre_img[:, :, channel] = (haze_img_lab[:, :, channel] - avg_haze[channel]) * (std_clean[channel] / std_haze[channel]) + avg_clean[channel]
        return np.clip(np.uint8(lab2rgb(pre_img) * 255.0), np.uint8(0), np.uint8(255))

    def __del__(self):
        print("Done")

    def run(self):
        self.loadImages()
        self.first_step()
        self.second_step()


if __name__ == '__main__':
    deHaze = DeHaze()
    deHaze.run()
