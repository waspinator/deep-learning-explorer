#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt


def display_images(images, titles=None, cols=4, cmap=None, norm=None,
                   interpolation=None):
    """Display the given set of images, optionally with titles.
    images: list or array of image tensors in HWC format.
    titles: optional. A list of titles to display with each image.
    cols: number of images per row
    cmap: Optional. Color map to use. For example, "Blues".
    norm: Optional. A Normalize instance to map values to colors.
    interpolation: Optional. Image interporlation to use for display.
    """
    titles = titles if titles is not None else [""] * len(images)
    rows = len(images) // cols + 1
    plt.figure(figsize=(14, 14 * rows // cols))
    i = 1
    for image, title in zip(images, titles):
        plt.subplot(rows, cols, i)
        plt.title(title, fontsize=9)
        plt.axis('off')
        plt.imshow(image.astype(np.float32), cmap=cmap,
                   norm=norm, interpolation=interpolation)
        i += 1
    plt.show()


def display_top_masks(image, mask, class_ids, class_names, limit=4):
    """Display the given image and the top few class masks."""
    to_display = []
    titles = []
    to_display.append(image)
    titles.append("H x W={}x{}".format(image.shape[0], image.shape[1]))
    # Pick top prominent classes in this image
    unique_class_ids = np.unique(class_ids)
    mask_area = [np.sum(mask[:, :, np.where(class_ids == i)[0]])
                 for i in unique_class_ids]
    top_ids = [v[0] for v in sorted(zip(unique_class_ids, mask_area),
                                    key=lambda r: r[1], reverse=True) if v[1] > 0]
    # Generate images and titles
    for i in range(limit):
        class_id = top_ids[i] if i < len(top_ids) else -1
        # Pull masks of instances belonging to the same class.
        m = mask[:, :, np.where(class_ids == class_id)[0]]
        m = np.sum(m * np.arange(1, m.shape[-1] + 1), -1)
        to_display.append(m)
        titles.append(class_names[class_id] if class_id != -1 else "-")
    display_images(to_display, titles=titles, cols=limit + 1, cmap="Blues_r")


def display_classes(instance_masks, class_ids, class_names, image=None,
                    overlay=False, outline=False, legend=True, figsize=(16, 16)):
    # TODO: visualize overlapping classes (sum of colors, stripped, hatched, ...)
    """Display classes for an image

    overlay: if true display classes on top of image, otherwise just show the classes
    outline: if true only draw an outline around the classes, otherwise fill in 
    """
    number_of_classes = len(np.unique(class_ids))
    mask_size = instance_masks.shape[0:2]
    classes_array = np.zeros(mask_size + (number_of_classes,))
    combined_class_array = np.zeros(mask_size)

    # group instances together by class
    for class_id, class_name in enumerate(class_names):
        m = instance_masks[:, :, np.where(class_ids == class_id)[0]]
        print(m.shape)

    for i in range(instance_masks.shape[-1]):
        instance_mask = instance_masks[..., i]
        # np.logical_or(instance_mask)
        # print(instance_mask.shape)

    # for class_id in class_ids:
    #    print(class_names[class_id])
    #


def display_generator_output(generator, image_indexes, class_names=None):
    # TODO: make this work with general data

    data, labels = generator._get_batches_of_transformed_samples(image_indexes)

    image = data[0, :].astype(np.float32)
    masks = labels[0, :]
    background_mask = masks[:, :, 0]
    square_mask = masks[:, :, 1]
    circle_mask = masks[:, :, 2]
    triangle_mask = masks[:, :, 3]

    display_images([image, square_mask, circle_mask, triangle_mask])
