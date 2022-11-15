class AUGMENT_IMAGE_CFG:
    # augmentation
    im_size = 256
    img_size = (im_size, im_size)
    batch_size = 16

    # AUGMENTATION
    augment   = True
    transform = True
    color     = True
    
    # TRANSFORMATION
    fill_mode = 'reflect'
    rot    = 30.0
    shr    = 0.0
    hzoom  = 16.0 # the smaller the more zoom, > 0
    wzoom  = 16.0 # the smaller the more zoom, > 0
    hshift = 2.0 # the bigger the more shift
    wshift = 2.0 # the bigger the more shift

    # FLIP
    hflip = True
    vflip = True

    # JITTER
    jitter = True
    pad_divide = 16 # the bigger the less jitter, > 0
    
    # COLOR: bri, contrast, ...
    sat  = [0.9, 1.1]
    cont = [0.9, 1.1]
    bri  = 0.1
    hue  = 0.0
    
    do_clip = True
    clip = [0, 255]
    
    # DDROPOUT
    drop_prob   = 0.4
    drop_cnt    = 1
    drop_size   = 0.2
    
    # GRIDMASK
    grid_prob = 0.1
    d1        = 4 # the bigger the bigger the 'hole'
    ratio     = 0.25 # ratio of length ~ width of hole
    max_angle = 90