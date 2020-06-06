import cv2
import keras
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import warnings
from copy import deepcopy

from generators.utils import get_affine_transform, affine_transform
from generators.utils import gaussian_radius, draw_gaussian, gaussian_radius_2, draw_gaussian_2


class Generator(keras.utils.Sequence):
    """
    Abstract generator class.
    """

    def __init__(
            self,
            multi_scale=False,
            multi_image_sizes=(320, 352, 384, 416, 448, 480, 512, 544, 576, 608),
            batch_size=1,
            group_method='ratio',  # one of 'none', 'random', 'ratio'
            shuffle_groups=True,
            input_size=512,
            max_objects=100,
            cat_spec_wh= False,
            transformations=None
    ):
        """
        Initialize Generator object.

        Args:
            batch_size: The size of the batches to generate.
            group_method: Determines how images are grouped together (defaults to 'ratio', one of ('none', 'random', 'ratio')).
            shuffle_groups: If True, shuffles the groups each epoch.
            input_size:
            max_objects:
        """
        self.batch_size = int(batch_size)
        self.group_method = group_method
        self.shuffle_groups = shuffle_groups
        self.input_size = input_size
        self.output_size = self.input_size // 4
        self.max_objects = max_objects
        self.cat_spec_wh = cat_spec_wh
        self.groups = None
        self.multi_scale = multi_scale
        self.multi_image_sizes = multi_image_sizes
        self.current_index = 0
        self.transformations = transformations

        # Define groups
        self.group_images()

        # Shuffle when initializing
        if self.shuffle_groups:
            random.shuffle(self.groups)

    def on_epoch_end(self):
        if self.shuffle_groups:
            random.shuffle(self.groups)
        self.current_index = 0

    def size(self):
        """
        Size of the dataset.
        """
        raise NotImplementedError('size method not implemented')

    def num_classes(self):
        """
        Number of classes in the dataset.
        """
        raise NotImplementedError('num_classes method not implemented')

    def has_label(self, label):
        """
        Returns True if label is a known label.
        """
        raise NotImplementedError('has_label method not implemented')

    def has_name(self, name):
        """
        Returns True if name is a known class.
        """
        raise NotImplementedError('has_name method not implemented')

    def name_to_label(self, name):
        """
        Map name to label.
        """
        raise NotImplementedError('name_to_label method not implemented')

    def label_to_name(self, label):
        """
        Map label to name.
        """
        raise NotImplementedError('label_to_name method not implemented')

    def image_aspect_ratio(self, image_index):
        """
        Compute the aspect ratio for an image with image_index.
        """
        raise NotImplementedError('image_aspect_ratio method not implemented')

    def load_image(self, image_index):
        """
        Load an image at the image_index.
        """
        raise NotImplementedError('load_image method not implemented')

    def load_annotations(self, image_index):
        """
        Load annotations for an image_index.
        """
        raise NotImplementedError('load_annotations method not implemented')

    def load_annotations_group(self, group):
        """
        Load annotations for all images in group.
        """
        # load_annotations {'labels': np.array, 'annotations': np.array}
        annotations_group = [self.load_annotations(image_index) for image_index in group]
        for annotations in annotations_group:
            assert (isinstance(annotations,
                               dict)), '\'load_annotations\' should return a list of dictionaries, received: {}'.format(
                type(annotations))
            assert (
                    'labels' in annotations), '\'load_annotations\' should return a list of dictionaries that contain \'labels\' and \'bboxes\'.'
            assert (
                    'bboxes' in annotations), '\'load_annotations\' should return a list of dictionaries that contain \'labels\' and \'bboxes\'.'

        return annotations_group

    def filter_annotations(self, image_group, annotations_group, group):
        """
        Filter annotations by removing those that are outside of the image bounds or whose width/height < 0.
        """
        # test all annotations
        for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            # test x2 < x1 | y2 < y1 | x1 < 0 | y1 < 0 | x2 <= 0 | y2 <= 0 | x2 >= image.shape[1] | y2 >= image.shape[0]
            invalid_indices = np.where(
                (annotations['bboxes'][:, 2] <= annotations['bboxes'][:, 0]) |
                (annotations['bboxes'][:, 3] <= annotations['bboxes'][:, 1]) |
                (annotations['bboxes'][:, 0] < 0) |
                (annotations['bboxes'][:, 1] < 0) |
                (annotations['bboxes'][:, 2] <= 0) |
                (annotations['bboxes'][:, 3] <= 0) |
                (annotations['bboxes'][:, 2] > image.shape[1]) |
                (annotations['bboxes'][:, 3] > image.shape[0])
            )[0]

            # delete invalid indices
            if len(invalid_indices):
                warnings.warn('Image with id {} (shape {}) contains the following invalid boxes: {}.'.format(
                    group[index],
                    image.shape,
                    annotations['bboxes'][invalid_indices, :]
                ))
                for k in annotations_group[index].keys():
                    annotations_group[index][k] = np.delete(annotations[k], invalid_indices, axis=0)
            if annotations['bboxes'].shape[0] == 0:
                continue
                warnings.warn('Image with id {} (shape {}) contains no valid boxes before transform'.format(
                    group[index],
                    image.shape,
                ))
        return image_group, annotations_group

    def load_image_group(self, group):
        """
        Load images for all images in a group.
        """
        return [self.load_image(image_index) for image_index in group]

    def random_transform_group_entry(self, image, annotations):
        """
        Randomly transforms image and annotation.
        """
        # randomly transform both image and annotations
        if self.transformations is not None:
            for transform in self.transformations:
                labels = np.copy(annotations['labels'])
                labels = np.expand_dims(labels, axis=-1)
                bboxes = np.copy(annotations['bboxes'])
                anno = np.concatenate([labels, bboxes],axis=-1)
                image, anno_n = transform(image, anno)
                annotations['labels'] = np.asarray([item[0] for item in anno_n])
                annotations['bboxes'] = np.reshape(np.asarray([item[1:] for item in anno_n]),(-1,4))
        return image, annotations

    def random_transform_group(self, image_group, annotations_group):
        """
        Randomly transforms each image and its annotations.
        """

        assert (len(image_group) == len(annotations_group))

        for index in range(len(image_group)):
            # transform a single group entry
            image_group[index], annotations_group[index] = self.random_transform_group_entry(image_group[index],
                                                                                             annotations_group[index])

        return image_group, annotations_group

    def group_images(self):
        """
        Order the images according to self.order and makes groups of self.batch_size.
        """
        # determine the order of the images

        order = list(range(self.size()))
        if self.group_method == 'random':
            random.shuffle(order)
        elif self.group_method == 'ratio':
            order.sort(key=lambda x: self.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        self.groups = [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in
                       range(0, len(order), self.batch_size)]

    def compute_inputs(self, image_group, annotations_group):
        """
        Compute inputs for the network using an image_group.
        """
        # construct an image batch object
        batch_images = np.zeros((len(image_group), self.input_size, self.input_size, 3), dtype=np.float32)

        batch_hms = np.zeros((len(image_group), self.output_size, self.output_size, self.num_classes()),
                             dtype=np.float32)
        batch_hms_2 = np.zeros((len(image_group), self.output_size, self.output_size, self.num_classes()),
                               dtype=np.float32)
        if not self.cat_spec_wh:
            batch_whs = np.zeros((len(image_group), self.max_objects, 2), dtype=np.float32)
            batch_wh_masks = np.zeros((len(image_group), self.max_objects), dtype=np.float32)
        else:
            batch_whs = np.zeros((len(image_group), self.max_objects, self.num_classes()*2), dtype=np.float32)
            batch_wh_masks = np.zeros((len(image_group), self.max_objects, self.num_classes()), dtype=np.float32)
        batch_regs = np.zeros((len(image_group), self.max_objects, 2), dtype=np.float32)
        batch_reg_masks = np.zeros((len(image_group), self.max_objects), dtype=np.float32)
        batch_indices = np.zeros((len(image_group), self.max_objects), dtype=np.float32)

        # copy all images to the upper left part of the image batch object
        for b, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            c = np.array([image.shape[1] / 2., image.shape[0] / 2.], dtype=np.float32)
            s = max(image.shape[0], image.shape[1]) * 1.0
            # trans_input = get_affine_transform(c, s, self.input_size)

            # inputs
            image = self.preprocess_image(image)
            batch_images[b] = image

            # outputs
            bboxes = annotations['bboxes']
#             assert bboxes.shape[0] != 0
            class_ids = annotations['labels']
#             assert class_ids.shape[0] != 0

            trans_output = get_affine_transform(c, s, self.output_size)
            for i in range(bboxes.shape[0]):
                bbox = bboxes[i].copy()*1.0
                cls_id = int(class_ids[i])
                # (x1, y1)
                bbox[:2] = affine_transform(bbox[:2], trans_output)
                # (x2, y2)
                bbox[2:] = affine_transform(bbox[2:], trans_output)
                bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, self.output_size - 1)
                bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, self.output_size - 1)
                h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
                if h > 0 and w > 0:
                    radius_h, radius_w = gaussian_radius((math.ceil(h), math.ceil(w)))
                    radius_h = max(0, int(radius_h))
                    radius_w = max(0, int(radius_w))

                    radius = gaussian_radius_2((math.ceil(h), math.ceil(w)))
                    radius = max(0, int(radius))
                    ct = np.array([(bbox[0] + bbox[2]) / 2., (bbox[1] + bbox[3]) / 2.], dtype=np.float32)
                    ct_int = ct.astype(np.int32)
                    draw_gaussian(batch_hms[b, :, :, cls_id], ct_int, radius_h, radius_w)
                    draw_gaussian_2(batch_hms_2[b, :, :, cls_id], ct_int, radius)
                    if not self.cat_spec_wh:
                        batch_whs[b, i] = 1. * w, 1. * h
                        batch_wh_masks[b, i] = 1
                    else:
                        batch_whs[b, i, cls_id*2:cls_id*2+2] = 1. * w, 1. * h
                        batch_wh_masks[b, i, cls_id] = 1
                        
                    batch_indices[b, i] = ct_int[1] * self.output_size + ct_int[0]
                    batch_regs[b, i] = ct - ct_int
                    batch_reg_masks[b, i] = 1

                    # hm = batch_hms[b, :, :, cls_id]
                    # hm = np.round(hm * 255).astype(np.uint8)
                    # hm = cv2.cvtColor(hm, cv2.COLOR_GRAY2BGR)
                    # hm_2 = batch_hms_2[b, :, :, cls_id]
                    # hm_2 = np.round(hm_2 * 255).astype(np.uint8)
                    # hm_2 = cv2.cvtColor(hm_2, cv2.COLOR_GRAY2BGR)
                    # cv2.rectangle(hm, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 1)
                    # cv2.rectangle(hm_2, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 1)
                    # cv2.namedWindow('hm', cv2.WINDOW_NORMAL)
                    # cv2.imshow('hm', np.hstack([hm, hm_2]))
                    # cv2.waitKey()
            # print(np.sum(batch_reg_masks[b]))
            # for i in range(self.num_classes()):
            #     plt.subplot(4, 5, i + 1)
            #     hm = batch_hms[b, :, :, i]
            #     plt.imshow(hm, cmap='gray')
            #     plt.axis('off')
            # plt.show()
            # hm = np.sum(batch_hms[0], axis=-1)
            # hm = np.round(hm * 255).astype(np.uint8)
            # hm = cv2.cvtColor(hm, cv2.COLOR_GRAY2BGR)
            # hm_2 = np.sum(batch_hms_2[0], axis=-1)
            # hm_2 = np.round(hm_2 * 255).astype(np.uint8)
            # hm_2 = cv2.cvtColor(hm_2, cv2.COLOR_GRAY2BGR)
            # for i in range(bboxes.shape[0]):
            #     x1, y1 = np.round(affine_transform(bboxes[i, :2], trans_input)).astype(np.int32)
            #     x2, y2 = np.round(affine_transform(bboxes[i, 2:], trans_input)).astype(np.int32)
            #     x1_, y1_ = np.round(affine_transform(bboxes[i, :2], trans_output)).astype(np.int32)
            #     x2_, y2_ = np.round(affine_transform(bboxes[i, 2:], trans_output)).astype(np.int32)
            #     class_id = class_ids[i]
            #     cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
            #     cv2.putText(image, str(class_id), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 3)
            #     cv2.rectangle(hm, (x1_, y1_), (x2_, y2_), (0, 255, 0), 1)
            #     cv2.rectangle(hm_2, (x1_, y1_), (x2_, y2_), (0, 255, 0), 1)
            # cv2.namedWindow('hm', cv2.WINDOW_NORMAL)
            # cv2.imshow('hm', np.hstack([hm, hm_2]))
            # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            # cv2.imshow('image', image)
            # cv2.waitKey()
        return [batch_images, batch_hms_2, batch_whs, batch_wh_masks, batch_regs, batch_reg_masks, batch_indices]

    def compute_targets(self, image_group, annotations_group):
        """
        Compute target outputs for the network using images and their annotations.
        """
        return np.zeros((len(image_group),))

    def compute_inputs_targets(self, group):
        """
        Compute inputs and target outputs for the network.
        """

        # load images and annotations
        # list
        image_group = self.load_image_group(group)
        annotations_group = self.load_annotations_group(group)

        # check validity of annotations
        image_group, annotations_group = self.filter_annotations(image_group, annotations_group, group)

        # randomly transform data
        image_group, annotations_group = self.random_transform_group(image_group, annotations_group)

        # check validity of annotations
        image_group, annotations_group = self.filter_annotations(image_group, annotations_group, group)

        if len(image_group) == 0:
            return None, None

        # compute network inputs
        inputs = self.compute_inputs(image_group, annotations_group)

        # compute network targets
        targets = self.compute_targets(image_group, annotations_group)

        return inputs, targets

    def __len__(self):
        """
        Number of batches for generator.
        """

        return len(self.groups)

    def __getitem__(self, index):
        """
        Keras sequence method for generating batches.
        """
        group = self.groups[self.current_index]
        if self.multi_scale:
            if self.current_index % 10 == 0:
                random_size_index = np.random.randint(0, len(self.multi_image_sizes))
                self.image_size = self.multi_image_sizes[random_size_index]
        inputs, targets = self.compute_inputs_targets(group)
        while inputs is None:
            current_index = self.current_index + 1
            if current_index >= len(self.groups):
                current_index = current_index % (len(self.groups))
            self.current_index = current_index
            group = self.groups[self.current_index]
            inputs, targets = self.compute_inputs_targets(group)
        current_index = self.current_index + 1
        if current_index >= len(self.groups):
            current_index = current_index % (len(self.groups))
        self.current_index = current_index
        return inputs, targets

    def preprocess_image(self, image, c=None, s=None, tgt_w=None, tgt_h=None):
        if c is not None and s is not None and tgt_w is not None and tgt_h is not None:
            trans_input = get_affine_transform(c, s, (tgt_w, tgt_h))
            image = cv2.warpAffine(image, trans_input, (tgt_w, tgt_h), flags=cv2.INTER_LINEAR)
        image = image.astype(np.float32)

#         image[..., 0] -= 103.939
#         image[..., 1] -= 116.779
#         image[..., 2] -= 123.68
        image = image/256.-0.5
        return image
