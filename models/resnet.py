from keras_resnet import models as resnet_models
from keras.applications.resnet50 import ResNet50
from keras.layers import *
from keras.models import Model
from keras.regularizers import l2
import keras.backend as K
import tensorflow as tf

from losses import loss, CTLoss
from models.utils import nms, topk, evaluate_batch_item, decode




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
