import math
import tensorflow as tf
from tensorflow.keras import backend as K 
import tensorflow_addons as tfa

def get_mat(shear, height_zoom, width_zoom, height_shift, width_shift):
    # returns 3x3 transformmatrix which transforms indicies
    # CONVERT DEGREES TO RADIANS

    shear = math.pi * shear / 180.

    def get_3x3_mat(lst):
        return tf.reshape(tf.concat([lst],axis=0), [3,3])
    
    # ROTATION MATRIX
    one  = tf.constant([1],dtype='float32')
    zero = tf.constant([0],dtype='float32')
      
    # SHEAR MATRIX
    c2 = tf.math.cos(shear)
    s2 = tf.math.sin(shear)    
    
    shear_matrix = get_3x3_mat([one,  s2,   zero, 
                               zero, c2,   zero, 
                                zero, zero, one])        
    # ZOOM MATRIX
    zoom_matrix = get_3x3_mat([one/height_zoom, zero,           zero, 
                               zero,            one/width_zoom, zero, 
                               zero,            zero,           one])    
    # SHIFT MATRIX
    shift_matrix = get_3x3_mat([one,  zero, height_shift, 
                                zero, one,  width_shift, 
                                zero, zero, one])
    
    return K.dot(shear_matrix,K.dot(zoom_matrix, shift_matrix))

def transformMAT(image, Crot, Cshr, Chzoom, Cwzoom, Chshift, Cwshift, Cfill_mode, DIM):#[rot,shr,h_zoom,w_zoom,h_shift,w_shift]):
    NEW_DIM = DIM[0]
    
    rot = Crot * tf.random.normal([1], dtype='float32')
    shr = Cshr * tf.random.normal([1], dtype='float32') 
    h_zoom = 1.0 + tf.random.normal([1], dtype='float32') / Chzoom
    w_zoom = 1.0 + tf.random.normal([1], dtype='float32') / Cwzoom
    h_shift = Chshift * tf.random.normal([1], dtype='float32') 
    w_shift = Cwshift * tf.random.normal([1], dtype='float32') 
    
    transformation_matrix=tf.linalg.inv(get_mat(shr,h_zoom,w_zoom,h_shift,w_shift))
    
    flat_tensor=tfa.image.transform_ops.matrices_to_flat_transforms(transformation_matrix)
    
    image=tfa.image.transform(image,flat_tensor, fill_mode=Cfill_mode)
    
    rotation = math.pi * rot / 180.
    
    image=tfa.image.rotate(image,-rotation, fill_mode=Cfill_mode)
    
#     image = tf.reshape(image, [*DIM, 3])    
    
    return image

def dropout(image,DIM=CFG.img_size, PROBABILITY = 0.6, CT = 5, SZ = 0.1):
    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]
    # output - image with CT squares of side size SZ*DIM removed
    
    # DO DROPOUT WITH PROBABILITY DEFINED ABOVE
    P = tf.cast( tf.random.uniform([],0,1)<PROBABILITY, tf.int32)
    if (P==0)|(CT==0)|(SZ==0): 
        return image
    
    for k in range(CT):
        # CHOOSE RANDOM LOCATION
        x = tf.cast( tf.random.uniform([],0,DIM[1]),tf.int32)
        y = tf.cast( tf.random.uniform([],0,DIM[0]),tf.int32)
        # COMPUTE SQUARE 
        WIDTH = tf.cast( SZ*min(DIM),tf.int32) * P
        ya = tf.math.maximum(0,y-WIDTH//2)
        yb = tf.math.minimum(DIM[0],y+WIDTH//2)
        xa = tf.math.maximum(0,x-WIDTH//2)
        xb = tf.math.minimum(DIM[1],x+WIDTH//2)
        # DROPOUT IMAGE
        one = image[ya:yb,0:xa,:]
        two = tf.zeros([yb-ya,xb-xa,3], dtype = image.dtype) 
        three = image[ya:yb,xb:DIM[1],:]
        middle = tf.concat([one,two,three],axis=1)
        image = tf.concat([image[0:ya,:,:],middle,image[yb:DIM[0],:,:]],axis=0)
        image = tf.reshape(image,[*DIM,3])

#     image = tf.reshape(image,[*DIM,3])
    
    return image

def grid_mask(image, d1=35, ratio=0.25, max_angle=90, batch_size=1, grid_prob=0.2):
    def get_batch_rotation_matrix(angles, batch_size=0):
        """Returns a tf.Tensor of shape (batch_size, 3, 3) with each element along the 1st axis being
           an image rotation matrix (which transforms indicies).

        Args:
            angles: 1-D Tensor with shape [batch_size].

        Returns:
            A 3-D Tensor with shape [batch_size, 3, 3].
        """    

        if batch_size == 0:
            batch_size = BATCH_SIZE

        # CONVERT DEGREES TO RADIANS
        angles = tf.constant(math.pi) * angles / 180.0

        # shape = (batch_size,)
        one = tf.ones_like(angles, dtype=tf.float32)
        zero = tf.zeros_like(angles, dtype=tf.float32)

        # ROTATION MATRIX
        c1 = tf.math.cos(angles) # shape = (batch_size,)
        s1 = tf.math.sin(angles) # shape = (batch_size,)

        # Intermediate matrix for rotation, shape = (9, batch_size) 
        rotation_matrix_temp = tf.stack([c1, s1, zero, -s1, c1, zero, zero, zero, one], axis=0)
        # shape = (batch_size, 9)
        rotation_matrix_temp = tf.transpose(rotation_matrix_temp)
        # Fianl rotation matrix, shape = (batch_size, 3, 3)
        rotation_matrix = tf.reshape(rotation_matrix_temp, shape=(batch_size, 3, 3))

        return rotation_matrix

    def batch_random_rotate(images, max_angles, batch_size=0):
        """Returns a tf.Tensor of the same shape as `images`, represented a batch of randomly transformed images.

        Args:
            images: 4-D Tensor with shape (batch_size, width, hight, depth).
                Currently, `depth` can only be 3.

        Returns:
            A 4-D Tensor with the same shape as `images`.
        """ 

        # input `images`: a batch of images [batch_size, dim, dim, 3]
        # output: images randomly rotated, sheared, zoomed, and shifted
        DIM = images.shape[1]
        XDIM = DIM % 2  # fix for size 331

        if batch_size == 0:
            batch_size = BATCH_SIZE

        angles = max_angles * tf.random.normal([batch_size], dtype='float32')


        # GET TRANSFORMATION MATRIX
        # shape = (batch_size, 3, 3)
        m = get_batch_rotation_matrix(angles, batch_size) 

        # LIST DESTINATION PIXEL INDICES
        x = tf.repeat(tf.range(DIM // 2, -DIM // 2, -1), DIM)  # shape = (DIM * DIM,)
        y = tf.tile(tf.range(-DIM // 2, DIM // 2), [DIM])  # shape = (DIM * DIM,)
        z = tf.ones([DIM * DIM], dtype='int32')  # shape = (DIM * DIM,)
        idx = tf.stack([x, y, z])  # shape = (3, DIM * DIM)

        # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS
        idx2 = tf.linalg.matmul(m, tf.cast(idx, dtype='float32'))  # shape = (batch_size, 3, DIM ** 2)
        idx2 = K.cast(idx2, dtype='int32')  # shape = (batch_size, 3, DIM ** 2)
        idx2 = K.clip(idx2, -DIM // 2 + XDIM + 1, DIM // 2)  # shape = (batch_size, 3, DIM ** 2)

        # FIND ORIGIN PIXEL VALUES
        # shape = (batch_size, 2, DIM ** 2)
        idx3 = tf.stack([DIM // 2 - idx2[:, 0, ], DIM // 2 - 1 + idx2[:, 1, ]], axis=1)  

        # shape = (batch_size, DIM ** 2, 3)
        d = tf.gather_nd(images, tf.transpose(idx3, perm=[0, 2, 1]), batch_dims=1)

        # shape = (batch_size, DIM, DIM, 3)
        new_images = tf.reshape(d, (batch_size, DIM, DIM, 3))

        return new_images

    def batch_get_grid_mask(d1, d2, ratio=0.25, max_angle=90, batch_size=0):

        # ratio: the ratio of black region
        DIM_H, DIM_W = CFG.img_size

        if batch_size == 0:
            batch_size = BATCH_SIZE

        # Length of diagonal
        hh = tf.cast((tf.math.ceil(tf.math.sqrt(2.0) * DIM_H)), tf.int64)
        hh = hh + tf.math.floormod(hh, 2)

        # We look squares of size dxd inside each image
        d = tf.random.uniform(shape=[], minval=d1, maxval=d2, dtype=tf.int64)

        # Inside each square of size dxd, we mask a square of size LxL (L <= d)
        l = tf.cast(tf.cast(d, tf.float32) * ratio + 0.5, tf.int64)

        lower_limit = -1
        upper_limit = tf.math.floordiv(hh, d) + 1
        indices = tf.range(lower_limit, upper_limit)  # shape = [upper_limit + 1]

        # The 1st component has shape [upper_limit + 1, 1]
        # The 2nd component has shae [1: L]
        # The addition has shape [upper_limit + 1: L]
        # The final output has sahpe [upper_limit + 1 * L]
        ranges = tf.reshape((d * indices)[:, tf.newaxis] + tf.range(l, dtype=tf.int64)[tf.newaxis, :], shape=[-1])
        shift = tf.random.uniform(shape=[], minval=0, maxval=d, dtype=tf.int64)

        ranges = shift + ranges

        clip_mask = tf.logical_or(ranges < 0 , ranges > hh - 1)
        ranges = tf.boolean_mask(ranges, tf.logical_not(clip_mask))

        hh_ranges = tf.tile(tf.range(0,hh), [tf.cast(tf.reduce_sum(tf.ones_like(ranges)), tf.int64)])

        ranges = tf.repeat(ranges, hh)

        y_hh_indices = tf.transpose(tf.stack([ranges, hh_ranges]))
        x_hh_indices = tf.transpose(tf.stack([hh_ranges, ranges]))

        y_mask_sparse = tf.SparseTensor(tf.cast(y_hh_indices, tf.int64),  tf.zeros_like(ranges), [hh, hh])

        y_mask = tf.sparse.to_dense(y_mask_sparse, 1, False)

        x_mask_sparse = tf.SparseTensor(tf.cast(x_hh_indices, tf.int64), tf.zeros_like(ranges), [hh, hh])
        x_mask = tf.sparse.to_dense(x_mask_sparse, 1, False)

        mask = tf.expand_dims( tf.clip_by_value(x_mask + y_mask, 0, 1), axis=-1)

        mask = batch_random_rotate(tf.broadcast_to(mask[tf.newaxis, :, :, :], shape=[batch_size, mask.shape[0], mask.shape[1], 3]), max_angle, batch_size)

        mask = tf.image.crop_to_bounding_box(mask, int((hh - DIM_H) // 2), int((hh - DIM_W) // 2), int(tf.cast(DIM_H, dtype=tf.int64)), int(tf.cast(DIM_W, dtype=tf.int64)))

        return mask

    P = tf.cast( tf.random.uniform([],0,1)<grid_prob, tf.int32)
    if (P==0): 
        return image
    
    # d1, d2 determined the width of the grid
    # d1
    d2 = d1 + 1 + tf.cast(d1 * tf.random.uniform(shape=[]), dtype=tf.int64)
    ratio = ratio + 0.25 * tf.random.uniform(shape=[])

    mask = batch_get_grid_mask(d1, d2, ratio, max_angle, batch_size)

    output = image * tf.cast(mask, tf.float32)
    
    return tf.squeeze(output)

def jitter(x, pad_divide=8, replace_value=0):
    """Flip left/right and jitter."""
    DIM = CFG.img_size
    
    image_size = min([x.shape[0], x.shape[1]])
    
    pad_size = tf.cast(image_size // pad_divide, tf.int32)
    
    x = tf.pad(
        x,
        paddings=[[pad_size, pad_size], [pad_size, pad_size], [0, 0]],
        constant_values=replace_value,
    )
    
    image = tf.image.random_crop(x, [*DIM, 3])

#     image = tf.reshape(image,[*DIM,3])

    return image

def color(img, AUGMENT_CFG):
    img = tf.image.random_hue(img, AUGMENT_CFG.hue)
    img = tf.image.random_saturation(img, AUGMENT_CFG.sat[0], AUGMENT_CFG.sat[1])
    img = tf.image.random_contrast(img, AUGMENT_CFG.cont[0], AUGMENT_CFG.cont[1])
    img = tf.image.random_brightness(img, AUGMENT_CFG.bri)

    return img

def augment_one_image(img, AUGMENT_CFG, dim=CFG.img_size):
    img = transformMAT(img, AUGMENT_CFG.rot, AUGMENT_CFG.shr, AUGMENT_CFG.hzoom, AUGMENT_CFG.wzoom, AUGMENT_CFG.hshift, AUGMENT_CFG.wshift, AUGMENT_CFG.fill_mode, dim) if AUGMENT_CFG.transform else img
    img = tf.image.random_flip_left_right(img) if AUGMENT_CFG.hflip else img
    img = tf.image.random_flip_up_down(img) if AUGMENT_CFG.vflip else img
    img = color(img, AUGMENT_CFG) if AUGMENT_CFG.color else img

    choose_cutout = tf.random.uniform([],0,2,tf.int32)
    if choose_cutout == 0:
        img = dropout(img, DIM=dim, PROBABILITY = AUGMENT_CFG.drop_prob, CT = AUGMENT_CFG.drop_cnt, SZ = AUGMENT_CFG.drop_size)
    elif choose_cutout == 1:
        img = grid_mask(img, d1=AUGMENT_CFG.d1, ratio=AUGMENT_CFG.ratio, max_angle=AUGMENT_CFG.max_angle, grid_prob=AUGMENT_CFG.grid_prob)

    if AUGMENT_CFG.jitter:
        jitter_color = tf.random.uniform([],0,256,tf.float32)
        img = jitter(img, AUGMENT_CFG.pad_divide, jitter_color)
        
    img = tf.clip_by_value(img, clip[0], clip[1]) if AUGMENT_CFG.do_clip else img         
    img = tf.reshape(img, [*dim, 3])

    return img

def augment_image(image, label, augment_cfg):
    # data augmentation. Thanks to the dataset.prefetch(AUTO) statement in the next function (below),
    # this happens essentially for free on TPU. Data pipeline code is executed on the "CPU" part
    # of the TPU while the TPU itself is computing gradients.
    # image is only 1 image
    image = augment_one_image(image, augment_cfg)
    return image, label  
def batch_cutmix(images, labels, PROBABILITY=1.0, batch_size=0):
    DIM_H, DIM_W = CFG.img_size
    CLASSES = n_labels
    
    if batch_size == 0:
        batch_size = CFG.batch_size
    
    # DO CUTMIX WITH PROBABILITY DEFINED ABOVE
    # This is a tensor containing 0 or 1 -- 0: no cutmix.
    # shape = [batch_size]
    do_cutmix = tf.cast(tf.random.uniform([batch_size], 0, 1) <= PROBABILITY, tf.int32)
    
    # Choose random images in the batch for cutmix
    # shape = [batch_size]
    new_image_indices = tf.cast(tf.random.uniform([batch_size], 0, batch_size), tf.int32)
    
    # Choose random location in the original image to put the new images
    # shape = [batch_size]
    new_x = tf.cast(tf.random.uniform([batch_size], 0, DIM_W), tf.int32)
    new_y = tf.cast(tf.random.uniform([batch_size], 0, DIM_H), tf.int32)
    
    # Random width for new images, shape = [batch_size]
    b = tf.random.uniform([batch_size], 0, 1) # this is beta dist with alpha=1.0
    new_width = tf.cast(DIM_W * tf.math.sqrt(1-b), tf.int32) * do_cutmix
    
    # shape = [batch_size]
    new_y0 = tf.math.maximum(0, new_y - new_width // 2)
    new_y1 = tf.math.minimum(DIM_H, new_y + new_width // 2)
    new_x0 = tf.math.maximum(0, new_x - new_width // 2)
    new_x1 = tf.math.minimum(DIM_W, new_x + new_width // 2)
    
    # shape = [batch_size, DIM]
    target_y = tf.broadcast_to(tf.range(DIM_H), shape=(batch_size, DIM_H))
    
    # shape = [batch_size, DIM]
    mask_y = tf.math.logical_and(new_y0[:, tf.newaxis] <= target_y, target_y <= new_y1[:, tf.newaxis])
    
    # shape = [batch_size, DIM]
    target_x = tf.broadcast_to(tf.range(DIM_W), shape=(batch_size, DIM_W))
    
    # shape = [batch_size, DIM]
    mask_x = tf.math.logical_and(new_x0[:, tf.newaxis] <= target_x, target_x <= new_x1[:, tf.newaxis])    
    
    # shape = [batch_size, DIM, DIM]
    mask = tf.cast(tf.math.logical_and(mask_y[:, :, tf.newaxis], mask_x[:, tf.newaxis, :]), tf.float32)
    
    # All components are of shape [batch_size, DIM, DIM, 3]
    new_images =  images * tf.broadcast_to(1 - mask[:, :, :, tf.newaxis], [batch_size, DIM_H, DIM_W, 3]) + \
                    tf.gather(images, new_image_indices) * tf.broadcast_to(mask[:, :, :, tf.newaxis], [batch_size, DIM_H, DIM_W, 3])
    
    a = tf.cast(new_width ** 2 / DIM_W ** 2, tf.float32)    
        
    # Make labels
    if len(labels.shape) == 1:
        labels = tf.one_hot(labels, CLASSES)
        
    new_labels =  (1-a)[:, tf.newaxis] * labels + a[:, tf.newaxis] * tf.gather(labels, new_image_indices)        
        
    return new_images, new_labels

def batch_mixup(images, labels, PROBABILITY=1.0, batch_size=0):
    CLASSES = n_labels
    
    if batch_size == 0:
        batch_size = CFG.batch_size
    
    # Do `batch_mixup` with a probability = `PROBABILITY`
    # This is a tensor containing 0 or 1 -- 0: no mixup.
    # shape = [batch_size]
    do_mixup = tf.cast(tf.random.uniform([batch_size], 0, 1) <= PROBABILITY, tf.int32)

    # Choose random images in the batch for cutmix
    # shape = [batch_size]
    new_image_indices = tf.cast(tf.random.uniform([batch_size], 0, batch_size), tf.int32)
    
    # ratio of importance of the 2 images to be mixed up
    # shape = [batch_size]
    a = tf.random.uniform([batch_size], 0, 1) * tf.cast(do_mixup, tf.float32)  # this is beta dist with alpha=1.0
                
    # The second part corresponds to the images to be added to the original images `images`.
    new_images =  (1-a)[:, tf.newaxis, tf.newaxis, tf.newaxis] * images + a[:, tf.newaxis, tf.newaxis, tf.newaxis] * tf.gather(images, new_image_indices)

    # Make labels
    if len(labels.shape) == 1:
        labels = tf.one_hot(labels, CLASSES)
    new_labels =  (1-a)[:, tf.newaxis] * labels + a[:, tf.newaxis] * tf.gather(labels, new_image_indices)

    return new_images, new_labels

def augment_batch(images, labels, augment_cfg): 
    P = tf.random.uniform([],0,3,tf.int32) # return [0, 1]
    
    if augment_cfg.mixup and P == 0:
        return batch_mixup(images, labels, PROBABILITY=augment_cfg.mixup_prob)
    
    if augment_cfg.cutmix and P == 1:
        return batch_cutmix(images, labels, PROBABILITY=augment_cfg.cutmix_prob)
    
    return images, labels