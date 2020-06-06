from keras.layers import *
from keras.models import *
from keras.regularizers import *
import numpy as np
import keras

bn_mom =  0.9
act_type = 'relu'
wd =  0.0001
epsilon = 2e-5

conv_option = {'kernel_regularizer': l2(wd),
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

def residual_block(data, num_filter,stride=1, dialation=1, residual=None):

    # conv-bn-relu-conv-bnjavascript:void(0)
    conv1 = data
    conv1 = ZeroPadding2D(1)(conv1)
    conv1 = Conv2D(filters=num_filter, kernel_size=3, strides=stride, **conv_option)(conv1)
    bn2 = BatchNormalization(**bn_option)(conv1)
    act1 = Act(act_type=act_type)(bn2)

    act1 = ZeroPadding2D(1)(act1)
    conv2 = Conv2D(filters=num_filter, kernel_size=3, strides=1, dilation_rate=dialation, **conv_option)(act1)
    bn3 = BatchNormalization(**bn_option)(conv2)
    if residual is not None:
        out = Add()([bn3, residual])
    else:
        assert stride==1
        out = Add()([bn3, data])
    out = Act(act_type)(out)
    return out


def root(children, filters, kernel, residual):
    x = children
    x = Concatenate(axis=-1)(x)
    x = ZeroPadding2D((kernel-1)//2)(x)
    x = Conv2D(filters, kernel, **conv_option)(x)
    x = BatchNormalization(**bn_option)(x)
    if residual:
        x = Add()([x, children[0]])
    x = Act(act_type)(x)
    return x


def Tree(data, levels, block, num_filter, stride=1,  root_kernel_size=1, dilation=1,
         level_root=False, root_residual=False, children=None):

    children = [] if children is None else children
    in_channel = data._keras_shape[-1]
    if stride > 1:
        bottom = MaxPool2D(stride)(data)
    else:
        bottom = data

    if in_channel != num_filter:
        residual = Conv2D(num_filter, kernel_size=1)(bottom)
        residual = BatchNormalization(**bn_option)(residual)
    else:
        residual = bottom

    if level_root:
        children.append(bottom)


    if levels == 1:
        x1 = block(data, num_filter, stride, dilation, residual)
        x2 = block(x1, num_filter,   1,      dilation)
        x = root([x2, x1]+children, num_filter, root_kernel_size, root_residual)
    else:
        x1 = Tree(data, levels-1, block, num_filter, stride=stride, root_kernel_size=root_kernel_size, dilation=dilation,
                  level_root=False, root_residual=root_residual, children=None)
        children.append(x1)
        x = Tree(x1, levels-1, block, num_filter, stride=1, root_kernel_size=root_kernel_size, dilation=dilation,
                 level_root=False, root_residual=root_residual, children=children)

    return x

def DLA(shape, n_layer=34, block=residual_block, root_residual=False, return_levels=False, n_class=1000, **kwargs):

    if n_layer == 34:
        levels = [1, 1, 1, 2, 2, 1]
        channels =  [16, 32, 64, 128, 256, 512]
    mode = 0
    if isinstance(shape,tuple) or isinstance(shape,list):
        inp = Input(shape)
    else:
        inp = shape
        mode = 1
    conv1 = Conv2D(channels[0], kernel_size=7, strides=1, padding="same", **conv_option)(inp)
    bn1 = BatchNormalization(**bn_option)(conv1)
    act1 = Act(act_type)(bn1)

    conv2 = Conv2D(channels[0], kernel_size=3, strides=1, padding="same", **conv_option)(act1)
    bn2 = BatchNormalization(**bn_option)(conv2)
    act2 = Act(act_type)(bn2)
    level0 = act2

    conv3 = Conv2D(channels[1], kernel_size=3, strides=2, padding="same", **conv_option)(act2)
    bn3 = BatchNormalization(**bn_option)(conv3)
    act3 = Act(act_type)(bn3)
    level1 = act3

    level2 = Tree(level1, levels[2], block, channels[2], stride=2,  root_kernel_size=1, dilation=1,
                  level_root=False, root_residual=root_residual, children=None)
    level3 = Tree(level2, levels[3], block, channels[3], stride=2, root_kernel_size=1, dilation=1,
                  level_root=True, root_residual=root_residual, children=None)
    level4 = Tree(level3, levels[4], block, channels[4], stride=2, root_kernel_size=1, dilation=1,
                  level_root=True, root_residual=root_residual, children=None)
    level5 = Tree(level4, levels[5], block, channels[5], stride=2, root_kernel_size=1, dilation=1,
                  level_root=True, root_residual=root_residual, children=None)
    if return_levels:
        x = [level0, level1, level2, level3, level4, level5]
    else:
        x = GlobalAveragePooling2D()(level5)
        x = Dense(n_class)(x)

    if mode == 0:

        m = Model(inp, x)
        return m
    else:
        return x

if __name__ == '__main__':
    model = DLA((512,512,3))
    model.summary()
    model.save('dla34.hdf5')