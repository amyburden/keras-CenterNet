from keras_resnet import models as resnet_models
from keras.applications.resnet50 import ResNet50
from keras.layers import *
from keras.models import Model
from keras.initializers import normal, constant, zeros
from keras.regularizers import l2
import keras.backend as K
import tensorflow as tf

from losses import loss, CTLoss


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
    hm = Reshape((-1,))(hm)
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


def decode(hm, wh, reg, max_objects=100, nms=True, flip_test=False, num_classes=20, score_threshold=0.1, cat_spec_wh=True):
    if flip_test:
        hm = (hm[0:1] + hm[1:2, :, ::-1]) / 2
        wh = (wh[0:1] + wh[1:2, :, ::-1]) / 2
        reg = reg[0:1]
    scores, indices, class_ids, xs, ys = topk(hm, max_objects=max_objects)
    b, h, w, c = hm._keras_shape
    # (b, h * w, 2)
    reg = Reshape((h*w, reg._keras_shape[-1]))(reg)
    # (b, h * w, 2*c) or (b, h * w, 2)
    wh = Reshape((h*w, wh._keras_shape[-1]))(wh)
    # (b, k, 2)
    ii = tf.tile(tf.range(tf.shape(reg)[0])[:, tf.newaxis], (1, max_objects))
    idx = tf.stack([ii, indices], axis=-1)
    
    # (b, k, 2)
    topk_reg = tf.gather_nd(reg, idx)
    topk_wh = tf.cast(tf.gather_nd(wh, idx), tf.float32)
    if not cat_spec_wh:
        topk_wh = Reshape((max_objects, 2))(topk_wh)
    else:
        # (b, k, c, 2)
        ii = tf.tile(tf.range(tf.shape(reg)[0])[:, tf.newaxis], (1, max_objects))
        jj = tf.tile(tf.range(max_objects)[tf.newaxis, :], (tf.shape(reg)[0], 1))
        idx = tf.stack([ii, jj, class_ids], axis=-1)
        topk_wh = Reshape((max_objects, c, 2))(topk_wh)
        topk_wh = tf.cast(tf.gather_nd(topk_wh, idx), tf.float32)
        topk_wh = Reshape((max_objects, 2))(topk_wh)
        
    
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


def centernet(num_classes, backbone='resnet18', input_size=512, max_objects=100, score_threshold=0.1,cat_spec_wh=False, 
              return_model_only=False,
              nms=True,
              flip_test=False,
              freeze_bn=False):
    assert backbone in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
    output_size = input_size // 4
    image_input = Input(shape=(input_size, input_size, 3))
    hm_input = Input(shape=(output_size, output_size, num_classes))
    if not cat_spec_wh:
        wh_input = Input(shape=(max_objects, 2))
        wh_mask_input = Input(shape=(max_objects,))
    else:
        wh_input = Input(shape=(max_objects, num_classes*2))
        wh_mask_input = Input(shape=(max_objects,num_classes))
        
    reg_input = Input(shape=(max_objects, 2))
    reg_mask_input = Input(shape=(max_objects,))
    index_input = Input(shape=(max_objects,))

    if backbone == 'resnet18':
        resnet = resnet_models.ResNet18(image_input, include_top=False, freeze_bn=freeze_bn)
    elif backbone == 'resnet34':
        resnet = resnet_models.ResNet34(image_input, include_top=False, freeze_bn=freeze_bn)
    elif backbone == 'resnet50':
        resnet = resnet_models.ResNet50(image_input, include_top=False, freeze_bn=freeze_bn)
#         resnet = ResNet50(input_tensor=image_input, include_top=False)
    elif backbone == 'resnet101':
        resnet = resnet_models.ResNet101(image_input, include_top=False, freeze_bn=freeze_bn)
    else:
        resnet = resnet_models.ResNet152(image_input, include_top=False, freeze_bn=freeze_bn)

    # (b, 16, 16, 2048)
    C5 = resnet.outputs[-1]
    # C5 = resnet.get_layer('activation_49').output
    x = Dropout(rate=0.5)(C5)
    # decoder
    num_filters = 256
    for i in range(3):
        # x = Conv2D(num_filters, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(
        #     x)
        # x = BatchNormalization()(x)
        # x = ReLU()(x)
        x = UpSampling2D()(x)
        x = Conv2D(num_filters // pow(2, i), (4,4), use_bias=False, padding='same',
                   kernel_initializer='he_normal',kernel_regularizer=l2(5e-4))(x)  
        x = BatchNormalization()(x)
        x = ReLU()(x)

    # hm header
    y1 = Conv2D(64, 3, padding='same', use_bias=True, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(x)
#     y1 = BatchNormalization()(y1)
#     y1 = ReLU()(y1)
    y1 = Conv2D(num_classes, 1, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4), activation='sigmoid')(y1)

    # wh header
    y2 = Conv2D(64, 3, padding='same', use_bias=True, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(x)
#     y2 = BatchNormalization()(y2)
#     y2 = ReLU()(y2)
    if cat_spec_wh:
        y2 = Conv2D(2*num_classes, 1, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(y2)
    else:
        y2 = Conv2D(2, 1, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(y2)

    # reg header
    y3 = Conv2D(64, 3, padding='same', use_bias=True, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(x)
#     y3 = BatchNormalization()(y3)
#     y3 = ReLU()(y3)
    y3 = Conv2D(2, 1, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(y3)
#     loss_ = Lambda(loss, name='centernet_loss')(
#         [y1, y2, y3, hm_input, wh_input, wh_mask_input, reg_input, reg_mask_input, index_input])
    loss_=CTLoss(name='centernet_loss')([y1, y2, y3, hm_input, wh_input, wh_mask_input, reg_input, reg_mask_input, index_input])
    model = Model(inputs=[image_input, hm_input, wh_input, wh_mask_input, reg_input, reg_mask_input, index_input], outputs=[loss_])
    
    if return_model_only:
        return model
    # detections = decode(y1, y2, y3)
    detections = Lambda(lambda x: decode(*x,
                                         max_objects=max_objects,
                                         score_threshold=score_threshold,
                                         nms=nms,
                                         flip_test=flip_test,
                                         num_classes=num_classes,
                                         cat_spec_wh=cat_spec_wh))([y1, y2, y3])
    prediction_model = Model(inputs=image_input, outputs=detections)
    debug_model = Model(inputs=image_input, outputs=[y1, y2, y3])
    return model, prediction_model, debug_model
