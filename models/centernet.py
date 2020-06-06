from keras.layers import *
from keras.models import *
from keras.regularizers import *
import numpy as np
import keras
from losses import loss
import tensorflow as tf
from dla import DLA

def nms(heat, kernel=3):
    hmax = MaxPool2D(kernel, strides=1,padding='same')(heat)
    heat = tf.where(tf.equal(hmax, heat), heat, tf.zeros_like(heat))
    return heat


def topk(hm, max_objects=100):
    hm = nms(hm)
    # (b, h * w * c)
    b, h, w, c = hm.shape.as_list()
    # hm2 = tf.transpose(hm, (0, 3, 1, 2))
    # hm2 = tf.reshape(hm2, (b, c, -1))
    hm = Reshape((h*w*c,))(hm)
    # (b, k), (b, k)
    scores, indices = tf.nn.top_k(hm, k=max_objects)
    # scores2, indices2 = tf.nn.top_k(hm2, k=max_objects)
    # scores2 = tf.reshape(scores2, (b, -1))
    # topk = tf.nn.top_k(scores2, k=max_objects)
    class_ids = indices % c
    xs = indices // c % w
    ys = indices // c // w
    indices = ys * w + xs
    return scores, indices, class_ids, xs, ys


def evaluate_batch_item(batch_item_detections, num_classes, max_objects_per_class=20, max_objects=100,
                        iou_threshold=0.5, score_threshold=0.1):
    batch_item_detections = tf.boolean_mask(batch_item_detections,
                                            tf.greater(batch_item_detections[:, 4], score_threshold))
    detections_per_class = []
    for cls_id in range(num_classes):
        class_detections = tf.boolean_mask(batch_item_detections, tf.equal(batch_item_detections[:, 5], cls_id))
        nms_keep_indices = tf.image.non_max_suppression(class_detections[:, :4],
                                                        class_detections[:, 4],
                                                        max_objects_per_class,
                                                        iou_threshold=iou_threshold)
        class_detections = K.gather(class_detections, nms_keep_indices)
        detections_per_class.append(class_detections)

    batch_item_detections = K.concatenate(detections_per_class, axis=0)

    def filter():
        # nonlocal batch_item_detections
        _, indices = tf.nn.top_k(batch_item_detections[:, 4], k=max_objects)
        batch_item_detections_ = tf.gather(batch_item_detections, indices)
        return batch_item_detections_

    def pad():
        # nonlocal batch_item_detections
        batch_item_num_detections = tf.shape(batch_item_detections)[0]
        batch_item_num_pad = tf.maximum(max_objects - batch_item_num_detections, 0)
        batch_item_detections_ = tf.pad(tensor=batch_item_detections,
                                        paddings=[
                                            [0, batch_item_num_pad],
                                            [0, 0]],
                                        mode='CONSTANT',
                                        constant_values=0.0)
        return batch_item_detections_

    batch_item_detections = tf.cond(tf.shape(batch_item_detections)[0] >= 100,
                                    filter,
                                    pad)
    return batch_item_detections


def decode(hm, wh, reg, max_objects=100, nms=True, flip_test=False, num_classes=20, score_threshold=0.1):
    if flip_test:
        hm = (hm[0:1] + hm[1:2, :, ::-1]) / 2
        wh = (wh[0:1] + wh[1:2, :, ::-1]) / 2
        reg = reg[0:1]
    scores, indices, class_ids, xs, ys = topk(hm, max_objects=max_objects)
    b, h, w, c = hm._keras_shape
    # (b, h * w, 2)
    reg = Reshape((h*w, reg._keras_shape[-1]))(reg)
    # (b, h * w, 2)
    wh = Reshape((h*w, wh._keras_shape[-1]))(wh)
    # (b, k, 2)
    ii = tf.tile(tf.range(tf.shape(reg)[0])[:, tf.newaxis], (1, max_objects))
    idx = tf.stack([ii, indices], axis=-1)

    topk_reg = tf.gather_nd(reg, idx)
    # (b, k, 2)
    topk_wh = tf.cast(tf.gather_nd(wh, idx), tf.float32)
    topk_cx = tf.cast(tf.expand_dims(xs, axis=-1), tf.float32) + topk_reg[..., 0:1]
    topk_cy = tf.cast(tf.expand_dims(ys, axis=-1), tf.float32) + topk_reg[..., 1:2]
    scores = tf.expand_dims(scores, axis=-1)
    class_ids = tf.cast(tf.expand_dims(class_ids, axis=-1), tf.float32)
    topk_x1 = topk_cx - topk_wh[..., 0:1] / 2
    topk_x2 = topk_cx + topk_wh[..., 0:1] / 2
    topk_y1 = topk_cy - topk_wh[..., 1:2] / 2
    topk_y2 = topk_cy + topk_wh[..., 1:2] / 2
    # (b, k, 6)
    detections = tf.concat([topk_x1, topk_y1, topk_x2, topk_y2, scores, class_ids], axis=-1)
    if nms:
        detections = tf.map_fn(lambda x: evaluate_batch_item(x[0],
                                                             num_classes=num_classes,
                                                             score_threshold=score_threshold),
                               elems=[detections],
                               dtype=tf.float32)
    return detections


bn_mom = 0.9
act_type = 'relu'
wd = 0.0001
epsilon = 2e-5

conv_option = {'kernel_regularizer': l2(wd),
               'use_bias': False}
conv_dw_option = {'depthwise_regularizer': l2(wd),
               'use_bias': False}
bn_option = {'epsilon': epsilon,
             'momentum': bn_mom}


def Act(act_type):
    if act_type == 'prelu':
        body = PReLU(shared_axes=[1, 2], alpha_regularizer=l2(0.00004))
    elif act_type == 'leaky':
        body = LeakyReLU(0.1171875)
    else:
        body = Activation(act_type)
    return body


def IDAUp(layers, node_kernel, num_filter, up_factors):
    """
    aggregate all layers and make nodes
    first layer should have smaller stride
    :param layers:
    :param node_kernel:
    :param num_filter:
    :param up_factors:
    :return:
    """
    # every layer do a upsample and projection except first layer

    for i, l in enumerate(layers):
        in_channel = l._keras_shape[-1]
        x = l

        if in_channel != num_filter:
            x = Conv2D(num_filter, kernel_size=1, **conv_option)(x)
            x = BatchNormalization(**bn_option)(x)
            x = Act(act_type)(x)

        if up_factors[i] != 1:
            x = UpSampling2D()(x)
            x = ZeroPadding2D((up_factors[i]*2+1)//2)(x)
            x = DepthwiseConv2D(kernel_size=up_factors[i]*2+1)(x)

        # replace old layer with new one
        layers[i] = x
    # all layer should have same size(h,w,c)

    x = layers[0]
    y = []
    # then aggregate all layer and make nodes
    for i in range(1,len(layers)):
        # every step aggregate two layer and make a node x
        x = Concatenate(axis=-1)([x, layers[i]])
        x = ZeroPadding2D(node_kernel//2)(x)
        x = Conv2D(num_filter, kernel_size=node_kernel,**conv_option)(x)
        x = BatchNormalization(**bn_option)(x)
        x = Act(act_type)(x)
        y.append(x)
    return x, y


def DLA_up(layers, channels, scales=(1,2,4,8,16)):
    """
    deep aggregation
    scales update
    step1:
        8, 16 -> 8, 8
    step2:
        4, 8, 8 -> 4, 4, 4
    step3:
        2, 4, 4, 4 -> 2 2 2 2
    step4:
        1 2 2 2 2 -> 1 1 1 1 1
    :param layers: output from backbone model
    :param channels: output channel for each step
    :param scales: stride of each layers
    :return:
    """
    layers = list(layers)
    assert (len(layers)==len(scales) and len(layers)==len(channels))
    scales = np.asarray(scales, 'int')
    assert (len(layers) > 1)
    x = layers[-1]
    for i in range(len(layers) - 1):
        # get last several layer to aggregation from 2 to all
        j = -i-2
        x,y = IDAUp(layers[j:], node_kernel=3, num_filter=channels[j], up_factors=scales[j:]//scales[j])
        # update layers with new node update scale with new scale
        layers[j+1:] = y
        scales[j+1:] = scales[j]
    return x


def DLA_seg(shape, backbone, down_ratio=4):

    assert (down_ratio in [2,4,8,16])
    first_level = int(np.log2(down_ratio))

    mode = 0
    if isinstance(shape, tuple) or isinstance(shape, list):
        inp = Input(shape)
    else:
        inp = shape
        mode = 1

    layers = backbone(inp)
    print layers
    channels = [item._keras_shape[-1] for item in layers]
    scales = [2**i for i in range(len(channels[first_level:]))]

    x = DLA_up(layers[first_level:], channels[first_level:], scales)

    if mode==1:
        return x
    else:
        return Model(inp, x)


def centernet(shape, backbone, num_classes, bottleneck=256):

    max_objects = 100
    flip_test = False

    inp = Input(shape)
    hm_input = Input(shape=(128, 128, num_classes))
    wh_input = Input(shape=(max_objects, 2))
    reg_input = Input(shape=(max_objects, 2))
    reg_mask_input = Input(shape=(max_objects,))
    index_input = Input(shape=(max_objects,))
    x = DLA_seg(inp, backbone)

    if bottleneck:
        x = Conv2D(bottleneck, 3, padding='same')(x)
        x = Act(act_type)(x)

    # hm header
    y1 = Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(x)
    y1 = BatchNormalization()(y1)
    y1 = ReLU()(y1)
    y1 = Conv2D(num_classes, 1, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4), activation='sigmoid')(y1)

    # wh header
    y2 = Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(x)
    y2 = BatchNormalization()(y2)
    y2 = ReLU()(y2)
    y2 = Conv2D(2, 1, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(y2)

    # reg offset header
    y3 = Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(x)
    y3 = BatchNormalization()(y3)
    y3 = ReLU()(y3)
    y3 = Conv2D(2, 1, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(y3)

    loss_ = Lambda(loss, name='centernet_loss')(
        [y1, y2, y3, hm_input, wh_input, reg_input, reg_mask_input, index_input])
    model = Model(inputs=[inp, hm_input, wh_input, reg_input, reg_mask_input, index_input], outputs=[loss_])

    # detections = decode(y1, y2, y3)
    detections = Lambda(lambda x: decode(*x,
                                         max_objects=max_objects,
                                         score_threshold=0.5,
                                         nms=nms,
                                         flip_test=flip_test,
                                         num_classes=num_classes))([y1, y2, y3])
    prediction_model = Model(inputs=inp, outputs=detections)
    debug_model = Model(inputs=inp, outputs=[y1, y2, y3])
    return model, prediction_model, debug_model


if __name__ == '__main__':
    shape = (512,512,3)
    dla_model = DLA(shape,return_levels=True)
    _, pred_model,_ = centernet(shape, dla_model, 20)
    pred_model.predict(np.random.random((1,512,512,3)))
    pred_model.summary()
