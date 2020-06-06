import tensorflow as tf
import keras.backend as K
from keras.layers import *
from keras.losses import mean_absolute_error


def focal_loss(hm_pred, hm_true, alpha=2, beta=4):
    pos_mask = tf.cast(tf.equal(hm_true, 1), tf.float32)
    neg_mask = tf.cast(tf.less(hm_true, 1), tf.float32)
    neg_weights = tf.pow(1 - hm_true, beta)

    pos_loss = -tf.log(tf.clip_by_value(hm_pred, 1e-4, 1. - 1e-4)) * tf.pow(1 - hm_pred, alpha) * pos_mask
    neg_loss = -tf.log(tf.clip_by_value(1 - hm_pred, 1e-4, 1. - 1e-4)) * tf.pow(hm_pred, alpha) * neg_weights * neg_mask

    num_pos = tf.reduce_sum(pos_mask)
    pos_loss = tf.reduce_sum(pos_loss)
    neg_loss = tf.reduce_sum(neg_loss)

    cls_loss = tf.cond(tf.greater(num_pos, 0), lambda: (pos_loss + neg_loss) / num_pos, lambda: neg_loss)
    return cls_loss


def reg_l1_loss(y_pred, y_true, indices, mask):
    b, h, w, c = y_pred._keras_shape
    k = indices._keras_shape[1]
    y_pred = Reshape((-1, c))(y_pred)
    indices = tf.cast(indices, tf.int32)

    ii = tf.tile(tf.range(tf.shape(y_pred)[0])[:, tf.newaxis], (1, k))
    idx = tf.stack([ii, indices], axis=-1)
    y_pred = tf.gather_nd(y_pred, idx)
    mask = Reshape((k, -1, 1))(mask)
    mask = tf.tile(mask, (1, 1, 1, 2))
    mask = Reshape((k, -1))(mask)
    total_loss = tf.reduce_sum(tf.abs(y_true * mask - y_pred * mask))
    reg_loss = total_loss / (tf.reduce_sum(mask) + 1e-4)
    return reg_loss


def loss(args):
    hm_pred, wh_pred, reg_pred, hm_true, wh_true, wh_mask, reg_true, reg_mask, indices = args
    hm_loss = focal_loss(hm_pred, hm_true)
    wh_loss = 0.1 * reg_l1_loss(wh_pred, wh_true, indices, wh_mask)
    reg_loss = reg_l1_loss(reg_pred, reg_true, indices, reg_mask)
    total_loss = hm_loss + wh_loss + reg_loss
    return total_loss


class CTLoss(Layer):
    # initialize the layer, and set an extra parameter axis. No need to include inputs parameter!
    def __init__(self, alpha=2, beta=4, wh_weight=0.1, **kwargs):
        self.alpha = alpha
        self.beta = beta
        self.wh_weight = wh_weight
        self.result = None
        super(CTLoss, self).__init__(**kwargs)

    # first use build function to define parameters, Creates the layer weights.
    # input_shape will automatic collect input shapes to build layer
    def build(self, input_shape):
        print(input_shape)
        super(CTLoss, self).build(input_shape)

    # This is where the layer's logic lives. In this example, I just concat two tensors.
    def call(self, inputs, **kwargs):
        hm_pred, wh_pred, reg_pred, hm_true, wh_true, wh_mask, reg_true, reg_mask, indices = inputs
        hm_loss = focal_loss(hm_pred, hm_true,self.alpha, self.beta)
        wh_loss = self.wh_weight * reg_l1_loss(wh_pred, wh_true, indices, wh_mask)
        reg_loss = reg_l1_loss(reg_pred, reg_true, indices, reg_mask)
        total_loss = hm_loss + wh_loss + reg_loss
        self.result = tf.reshape(total_loss,(1,))
        return self.result

    # return output shape
    def compute_output_shape(self, input_shape):
        # return K.int_shape(self.result)
        return (1,)