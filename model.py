from layers import *

# !pip install -U git+https://github.com/leondgarse/keras_cv_attention_models -q
from keras_cv_attention_models import efficientnet, convnext

def get_base_model(name, input_shape):
    if name == 'EfficientNetV2S':
        return efficientnet.EfficientNetV2S(num_classes=0, input_shape=input_shape, pretrained="imagenet21k")

    if name == 'EfficientNetV1B1':
        return efficientnet.EfficientNetV1B1(num_classes=0, input_shape=input_shape, pretrained="imagenet")

    if name == 'EfficientNetV1B2':
        return efficientnet.EfficientNetV1B2(num_classes=0, input_shape=input_shape, pretrained="imagenet")

    if name == 'EfficientNetV1B3':
        return efficientnet.EfficientNetV1B3(num_classes=0, input_shape=input_shape, pretrained="imagenet")

    if name == 'EfficientNetV1B4':
        return efficientnet.EfficientNetV1B4(num_classes=0, input_shape=input_shape, pretrained="imagenet")

    if name == 'EfficientNetV1B5':
        return efficientnet.EfficientNetV1B5(num_classes=0, input_shape=input_shape, pretrained="imagenet")

    if name == 'EfficientNetV1B6':
        return efficientnet.EfficientNetV1B6(num_classes=0, input_shape=input_shape, pretrained="imagenet")

    if name == 'EfficientNetV1B7':
        return efficientnet.EfficientNetV1B7(num_classes=0, input_shape=input_shape, pretrained="imagenet")

    if name == 'ConvNeXtTiny':
        return convnext.ConvNeXtTiny(num_classes=0, input_shape=input_shape, pretrained="imagenet21k-ft1k")

    if name == 'ResNet50':
        return tf.keras.applications.resnet50.ResNet50(include_top=False, input_shape=input_shape, weights='imagenet')

    raise Exception("Cannot find this base model:", name)

def create_simple_emb_model(base, final_dropout=0.1, have_emb_layer=True, emb_dim=128, name="embedding"):
    feature = base.output

    x = GlobalAveragePooling2D()(feature)
    x = Dropout(final_dropout)(x)

    if have_emb_layer:
        x = Dense(emb_dim, use_bias=False, name='bottleneck')(x)
        x = BatchNormalization(name='bottleneck_bn')(x)
    
    model = Model(base.input, x, name=name)

    return model

def create_emb_model(base, final_dropout=0.1, have_emb_layer=True, name="embedding",
                     emb_dim=512, extract_dim=128, dense_dim=96, trans_layers=1,
                     kernel_sizes=[1, 3, 7], dilation_rates=[4, 8, 12]):
    backbone_layer_names = [
                'stack_0_block1_output',
                'stack_1_block3_output',
                'stack_2_block3_output',
                'stack_4_block8_output',
                'post_swish'
            ]

    backbone_layers = [base.get_layer(layer_name).output for layer_name in backbone_layer_names]

    global_feature = GlobalAveragePooling2D()(backbone_layers[-1])

    backbone_layers = backbone_layers[:-1]

    list_features = []
    for backbone_layer in backbone_layers:
        f_mkn = mkn_conv(backbone_layer, extract_dim, kernel_sizes)
        f_atrous = atrous_conv(backbone_layer, extract_dim, dilation_rates)
        f = f_mkn + f_atrous
        f = wBiFPNAdd()(f)
        f = GlobalAveragePooling2D()(f)
        list_features.append(f)

    x = tf.stack(list_features, 1)

    num_heads = x.shape[1]

    pe = PositionEmbedding(input_shape=(num_heads, extract_dim),
                           input_dim=num_heads,
                           output_dim=extract_dim,
                           mode=PositionEmbedding.MODE_ADD,
    )

    x = pe(x)

    for i in range(trans_layers):
        x = TransformerEncoder(extract_dim, dense_dim, num_heads)(x)

    x = layers.GlobalAveragePooling1D()(x)

    x = Concatenate()([global_feature, x])
    x = Dropout(final_dropout)(x)

    if have_emb_layer:
        x = Dense(emb_dim, use_bias=False, name='bottleneck')(x)
        x = BatchNormalization(name='bottleneck_bn')(x)
    
    model = Model(base.input, x, name=name)

    return model

def create_model(input_shape, emb_model, n_labels, use_normdense=True, use_cate_int=False):
    inp = Input(shape=input_shape, name="input_1")
    
    x = emb_model(inp)
    
    if use_normdense:
        cate_output = NormDense(n_labels, name='cate_output')(x)
    else:
        cate_output = Dense(n_labels, name='cate_output')(x)

    if not use_cate_int:
        model = Model([inp], [cate_output])
    else:
        model = Model([inp], [cate_output, x])
    
    return model

if __name__ == "__main__":
    import os
    from utils import *
    from multiprocess_dataset import *

    os.environ["CUDA_VISIBLE_DEVICES"]=""

    settings = get_settings()
    globals().update(settings)

    route_dataset = path_join(route, 'dataset')

    img_size = (im_size, im_size)
    input_shape = (im_size, im_size, 3)

    use_cate_int = False
    if label_mode == 'cate_int':
        use_cate_int = True

    n_labels = 100

    base = get_base_model(base_name, input_shape)
    if use_simple_emb:
        emb_model = create_emb_model(base, final_dropout, have_emb_layer, emb_dim)
    else:
        emb_model = create_emb_model(base, final_dropout, have_emb_layer, "embedding",
                                     emb_dim, extract_dim, dense_dim, trans_layers,
                                     kernel_sizes, dilation_rates)
    model = create_model(input_shape, emb_model, n_labels, use_normdense, use_cate_int)
    model.summary()