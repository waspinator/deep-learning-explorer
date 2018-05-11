#!/usr/bin/env python3

import numpy as np
import mrcnn.utils
import mrcnn.model as modellib
import pycocotools
import pycococreatortools.pycococreatortools as pycococreatortools
import datetime

def compute_per_class_precision(gt_boxes, gt_class_ids, gt_masks,
              pred_boxes, pred_class_ids, pred_scores, pred_masks,
              class_infos, iou_threshold=0.5):
    """
        Compute per class precision
    """
    
    class_precisions = {}
    
    for class_info in class_infos:
        if class_info["name"] == "BG":
            continue
        
        class_gt_indexes = np.where(gt_class_ids == class_info["id"])
        class_gt_boxes = gt_boxes[class_gt_indexes]
        class_gt_masks = gt_masks[:, :, class_gt_indexes[0]]        
        class_gt_ids = np.full(np.size(class_gt_indexes), class_info["id"])
        
        class_pred_indexes = np.where(pred_class_ids == class_info["id"])
        class_pred_boxes = pred_boxes[class_pred_indexes]
        class_pred_masks = pred_masks[:, :, class_pred_indexes[0]]
        class_pred_scores = pred_scores[class_pred_indexes]
        class_pred_ids = np.full(np.size(class_pred_indexes), class_info["id"])
        
        if np.shape(class_gt_masks)[2] == 0 and np.shape(class_pred_masks)[2] == 0:
            continue   

        if np.shape(class_gt_masks)[2] == 0:
            class_gt_indexes = (np.array([0]),)
            class_gt_boxes = np.array([[1, 1, 1, 1]])
            class_gt_masks =  np.zeros([np.shape(class_gt_masks)[0], np.shape(class_gt_masks)[1], 1])
            class_gt_ids = np.full(np.size(class_gt_indexes), class_info["id"])

        if np.shape(class_pred_masks)[2] == 0:
            class_pred_indexes = (np.array([0]),)
            class_pred_masks =  np.zeros([np.shape(class_gt_masks)[0], np.shape(class_gt_masks)[1], 1])
            class_pred_boxes = np.array([[1, 1, 1, 1]])
            class_pred_scores = np.array([0])
            class_pred_ids = np.full(np.size(class_pred_indexes), class_info["id"])
 
        AP, precisions, recalls, overlaps =\
            mrcnn.utils.compute_ap(class_gt_boxes, class_gt_ids, class_gt_masks,
                            class_pred_boxes, class_pred_ids, class_pred_scores, class_pred_masks,
                            iou_threshold)
        
        class_precisions[class_info["name"]] = {
            "average_precision": AP,
            "precisions": precisions,
            "recalls": recalls,
            "overlaps": overlaps
        }
    
    return class_precisions

def compute_multiple_per_class_precision(model, inference_config, dataset, 
                                        number_of_images=10, iou_threshold=0.5):
    """
        Compute per class precision on multiple images
    """

    image_ids = np.random.choice(dataset.image_ids, number_of_images, replace=False)

    class_precisions = {}

    for image_id in image_ids:
        image, _, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset, inference_config,
                                image_id, use_mini_mask=False)

        results = model.detect([image], verbose=0)
        r = results[0]

        class_precision_info =\
        compute_per_class_precision(gt_bbox, gt_class_id, gt_mask,
                r["rois"], r["class_ids"], r["scores"], r["masks"],
                dataset.class_info, iou_threshold)
        
        for class_name in class_precision_info:
            if class_precisions.get(class_name):
                class_precisions[class_name].append(class_precision_info[class_name]['average_precision'])
            else:
                class_precisions[class_name] = [class_precision_info[class_name]['average_precision']]
                
    return class_precisions

def result_to_coco(result, class_names, image_size, tolerance=2, INFO=None, LICENSES=None):
    """Encodes Mask R-CNN detection result into COCO format
    """

    if INFO is None:
        INFO = {
            "description": "Mask R-CNN Result",
            "url": "https://github.com/waspinator/deep-learning-explorer",
            "version": "0.1.0",
            "year": 2018,
            "contributor": "waspinator",
            "date_created": datetime.datetime.utcnow().isoformat(' ')
        }

    if LICENSES is None:
        LICENSES = [
            {
                "id": 1,
                "name": "Attribution-NonCommercial-ShareAlike License",
                "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
            }
        ]

    IMAGES = [
        {
            "id": 1,
            "width": image_size[1],
            "height": image_size[0],
            "license": 1
        }
    ]

    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": [],
        "images": IMAGES,
        "annotations": []
    }

    for index, class_name in enumerate(class_names):
        if class_name == 'BG':
            continue
        
        category = {
            "id": index,
            "name": class_name,
            "supercategory": ""
        }

        coco_output["categories"].append(category)

    for index in range(result['masks'].shape[-1]):
        mask = result['masks'][...,index]

        bounding_box = np.array([
            result['rois'][index][1],
            result['rois'][index][0],
            result['rois'][index][3] - result['rois'][index][1],
            result['rois'][index][2] - result['rois'][index][0]
        ])

        annotation = pycococreatortools.create_annotation_info(
            annotation_id=index,
            image_id=1,
            category_info={"id": result['class_ids'][index].item(), "is_crowd": False},
            binary_mask=mask,
            image_size=image_size,
            tolerance=tolerance,
            bounding_box=bounding_box
            )
        
        if annotation is not None:
            annotation['confidence'] = "{:.4f}".format(result['scores'][index].item())
            coco_output['annotations'].append(annotation)

    return coco_output
