import os
import shutil
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from glob import glob

from utils import *
from dataset import *
from model import *
from loss import *
from callback import *
from thread import *

os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

set_memory_growth()

route_dataset = '/home/lap14880/hieunmt/faceid/nonmask/dataset'
# route_dataset = '/home/lap14880/hieunmt/faceid/nonmask/download/VN-celeb'

im_size = 160
img_size = (im_size, im_size)
input_shape = (im_size, im_size, 3)
ext = 'jpg'
label_mode = 'cate_int'
BATCH_SIZE = 1024

cate_int = False
if label_mode == 'cate_int':
    cate_int = True

# n_fold = 100
# valid_fold = 0
# test_fold = None
# stratified = True

valid_ratio = 0.001

train_with_labels = True
train_repeat = True
train_shuffle = 8192
train_augment = False

valid_with_labels = True
valid_repeat = False
valid_shuffle = False
valid_augment = False

SEED = 42

base_name = 'ConvNeXtTiny'

final_dropout = 0.1
have_emb_layer = True
emb_dim = 256
use_normdense = True

arcface_label_smoothing = 0
arcface_margin1 = 1.0
arcface_margin2 = 0.5
arcface_margin3 = 0
arc_face_weight = 1.0

sup_con_temperature = 0.3
sup_con_weight = 1.0

monitor = "val_loss"
mode = 'min'

max_lr = 1e-3
min_lr = 1e-5
cycle_epoch = 10
n_cycle = 1
epochs = cycle_epoch * n_cycle
print('epochs:', epochs)

seedEverything(SEED)

print('BATCH_SIZE:', BATCH_SIZE)

# X_train, Y_train, all_class, X_valid, Y_valid = auto_split_data_multiprocessing(route_dataset, n_fold,
#                                                                                 valid_fold, test_fold, 
#                                                                                 stratified, SEED)

X_train, Y_train, all_class, X_valid, Y_valid = auto_split_tran_valid_bigdata_multiprocessing(route_dataset,
                                                                                              valid_ratio,
                                                                                              SEED)

train_n_images = len(Y_train)
train_dataset = build_dataset_from_X_Y(X_train, Y_train, all_class, train_with_labels, label_mode, img_size, ext,
                                       BATCH_SIZE, train_repeat, train_shuffle, train_augment)

valid_n_images = len(Y_valid)
valid_dataset = build_dataset_from_X_Y(X_valid, Y_valid, all_class, valid_with_labels, label_mode, img_size, ext,
                                       BATCH_SIZE, valid_repeat, valid_shuffle, valid_augment)

n_labels = len(all_class)

print('n_labels', n_labels)
print('train_n_images', train_n_images)
print('valid_n_images', valid_n_images)

tf.keras.backend.clear_session()
tf.compat.v1.reset_default_graph()

strategy = auto_select_accelerator()

with strategy.scope():
    base = get_base_model(base_name, input_shape)
    emb_model = create_emb_model(base, final_dropout, have_emb_layer, emb_dim)
    model = create_model(input_shape, emb_model, n_labels, use_normdense, cate_int)
    model.summary()

    losses = {
        'cate_output' : ArcfaceLoss(from_logits=True, 
                                    label_smoothing=arcface_label_smoothing,
                                    margin1=arcface_margin1,
                                    margin2=arcface_margin2,
                                    margin3=arcface_margin3),
        'embedding' : SupervisedContrastiveLoss(temperature=sup_con_temperature),
    }

    loss_weights = {
        'cate_output' : arc_face_weight,
        'embedding' : sup_con_weight,
    }

    metrics = {
        'cate_output' : tf.keras.metrics.CategoricalAccuracy()
    }

model.compile(optimizer=Adam(learning_rate=1e-3),
              loss=losses,
              loss_weights=loss_weights,
              metrics=metrics)

save_path = f'best_model_motor_reid_{base_name}_{im_size}_{n_labels}.h5'

callbacks = get_callbacks(monitor, mode, save_path, max_lr, min_lr, cycle_epoch)

his = model.fit(train_dataset, 
                validation_data=valid_dataset,
                steps_per_epoch = train_n_images//BATCH_SIZE,
                epochs=epochs,
                verbose=1,
                callbacks=callbacks)

# metric = 'loss'
# visual_save_metric(his, metric)

# metric = 'cate_output_categorical_accuracy:'
# visual_save_metric(his, metric)

# # EVALUATE

# valid_eval = model.evaluate(valid_dataset)

# print("valid_eval", valid_eval)

# with open("valid_eval.txt", mode='w') as f:
#     for item in valid_eval:
#         f.write(str(item) + " ")




