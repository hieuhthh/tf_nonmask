import os
from glob import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
import tensorflow as tf
# image dataset from directory

def get_data_from_phrase(route, phrase=None):
    """
    input:
        route to main directory and phrase ("train", "valid", "test")
        or just route to the directory that its subfolder are classes
    output:
        X_path: path to img
        Y_int: int label
        all_class: list of string class name
    """
    X_path = []
    Y_int = []
    phrase_path = os.path.join(route, phrase) if phrase is not None else route
    all_class = sorted(os.listdir(phrase_path))
    for cl in all_class:
        path2cl = os.path.join(phrase_path, cl)
        temp =  glob(path2cl + '/*')
        X_path = X_path + temp
        Y_int = Y_int + len(temp) * [all_class.index(cl)]
    return X_path, Y_int, all_class

def auto_split_data(route, fold, valid_fold=None, test_fold=None, stratified=False, seed=42):
    """
    input:
        route to the directory that its subfolder are classes
        fold: how many folds
    output:
        X, Y, class corresponding
    """
    X_path, Y_int, all_class = get_data_from_phrase(route)

    df = pd.DataFrame({'image':X_path, 'label':Y_int})
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    if stratified:
        skf = StratifiedKFold(n_splits=fold, random_state=seed, shuffle=True)
        for idx, (train_index, test_index) in enumerate(skf.split(df.image, df.label)):
            df.loc[test_index, 'fold'] = idx
    else:
        kf = KFold(n_splits=fold, random_state=seed, shuffle=True)
        for idx, (train_index, test_index) in enumerate(kf.split(df.image)):
            df.loc[test_index, 'fold'] = idx

    if valid_fold is not None and test_fold is not None:
        df_train = df[df['fold']!=valid_fold]
        df_train = df_train[df_train['fold']!=test_fold]

        df_valid = df[df['fold']==valid_fold]
        df_test = df[df['fold']==test_fold]

        X_train = df_train['image'].values
        X_valid = df_valid['image'].values
        X_test = df_test['image'].values

        Y_train = df_train['label'].values
        Y_valid = df_valid['label'].values
        Y_test = df_test['label'].values

        return X_train, Y_train, all_class, X_valid, Y_valid, X_test, Y_test

    elif valid_fold is not None:
        df_train = df[df['fold']!=valid_fold]

        df_valid = df[df['fold']==valid_fold]

        X_train = df_train['image'].values
        X_valid = df_valid['image'].values

        Y_train = df_train['label'].values
        Y_valid = df_valid['label'].values

        return X_train, Y_train, all_class, X_valid, Y_valid

    else:
        df_train = df

        X_train = df_train['image'].values

        Y_train = df_train['label'].values

        return X_train, Y_train, all_class

def build_decoder(with_labels=True, label_mode='int', all_class=None, target_size=(256, 256), ext='png'):
    def decode_img_preprocess(img):
        img = tf.image.resize(img, target_size)
        img = tf.image.random_flip_left_right(img)
        img = tf.cast(img, tf.float32) / 255.0
        return img

    def decode_img(path):
        """
        path to image
        """
        file_bytes = tf.io.read_file(path)
        img = tf.io.decode_image(file_bytes, channels=3, expand_animations=False)

        # if ext == 'png':
        #     img = tf.image.decode_png(file_bytes, channels=3)
        # elif ext in ['jpg', 'jpeg']:
        #     img = tf.image.decode_jpeg(file_bytes, channels=3)
        # else:
        #     raise ValueError("Image extension not supported")

        img = decode_img_preprocess(img)
        return img
    
    def decode_label(label):
        """
        label: int label
        """
        if label_mode == 'int':
            return label
        if label_mode == 'cate': # categorical
            return tf.one_hot(label, len(all_class))
        if label_mode == 'cate_int':
            return tf.one_hot(label, len(all_class)), label
        raise ValueError("Label mode not supported")
        
    def decode_with_labels(path, label):
        return decode_img(path), decode_label(label)
    
    return decode_with_labels if with_labels else decode_img

def build_dataset(paths, labels=None, labelsID=None, bsize=32, cache=True,
                  decode_fn=None, augment=False, 
                  repeat=True, shuffle=1024,
                  cache_dir=""):
    """
    paths: paths to images
    labels: int label
    """              
    if cache_dir != "" and cache is True:
        os.makedirs(cache_dir, exist_ok=True)
    if decode_fn is None:
        decode_fn = build_decoder(labels is not None)
    
    AUTO = tf.data.experimental.AUTOTUNE
    dataset_input = tf.data.Dataset.from_tensor_slices((paths))
    dataset_label = tf.data.Dataset.from_tensor_slices((labels))

    dset = tf.data.Dataset.zip((dataset_input, dataset_label))
    dset = dset.cache(cache_dir) if cache else dset
    dset = dset.repeat() if repeat else dset
    dset = dset.shuffle(shuffle) if shuffle else dset
    dset = dset.map(decode_fn, num_parallel_calls=AUTO)
    dset = dset.batch(bsize)
    dset = dset.prefetch(AUTO)
    
    return dset

def build_dataset_from_X_Y(X_path, Y_int, all_class, with_labels, label_mode, img_size, ext,
                           batch_size, repeat, shuffle, augment):
    decoder = build_decoder(with_labels=with_labels, label_mode=label_mode, all_class=all_class, 
                            target_size=img_size, ext=ext)

    dataset = build_dataset(X_path, Y_int, bsize=batch_size, decode_fn=decoder,
                            repeat=repeat, shuffle=shuffle, augment=augment)

    return dataset

def get_tf_dataset_from_phrase(route, phrase, with_labels, label_mode, img_size, ext,
                               batch_size, repeat, shuffle, augment, return_classes):
    X_path, Y_int, all_class = get_data_from_phrase(route, phrase)
    
    n_images = len(Y_int)

    dataset = build_dataset_from_X_Y(X_path, Y_int, all_class, with_labels, label_mode, img_size, ext,
                                     batch_size, repeat, shuffle, augment)

    if return_classes:
        return all_class, n_images, dataset

    return n_images, dataset

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="2, 3"

    route = '/data/hieunmt/dataset/motorreid/download/dataset'

    X_path, Y_int, all_class = get_data_from_phrase(route, 'train')

    print(X_path[:3])
    print(Y_int[:3])
    print(len(all_class))

    im_size = 256
    img_size = (im_size, im_size)
    ext = 'png'
    label_mode = 'cate_int'
    BATCH_SIZE = 128

    train_decoder = build_decoder(with_labels=True, label_mode=label_mode, all_class=all_class, 
                                  target_size=img_size, ext=ext)
    valid_decoder = build_decoder(with_labels=True, label_mode=label_mode, all_class=all_class, 
                                  target_size=img_size, ext=ext)
    steps_per_epoch = len(X_path) // BATCH_SIZE

    train_repeat = True
    train_shuffle = 4096
    train_augment = False
    train_dataset = build_dataset(X_path, Y_int, bsize=BATCH_SIZE, decode_fn=train_decoder,
                                  repeat=True, shuffle=train_shuffle, augment=train_augment)

    for x, y in train_dataset:
        break
    print(x)
    print(y)

    import cv2
    import numpy as np

    cv2.imwrite("sample.png", np.array(x[0])*255)

