#!/usr/bin/env python
# coding=utf8

import numpy as np

def non_maximum_suppress(box_and_scores, thresh):
    lefts = box_and_scores[:, 0]
    bottoms = box_and_scores[:, 1]
    rights = box_and_scores[:, 2]
    tops = box_and_scores[:, 3]
    box_scores = box_and_scores[:, 4]
    box_areas = (rights - lefts + 1) * (tops - bottoms + 1)
    score_order = box_scores.argsort()[::-1]

    keep_indexes = []
    while score_order.size > 0:
        current_max_score_index = score_order[0]
        keep_indexes.append(current_max_score_index)
        intersections_lefts = np.maximum(lefts[current_max_score_index], lefts[score_order[1:]])
        intersections_bottoms = np.maximum(bottoms[current_max_score_index], bottoms[score_order[1:]])
        intersections_rights = np.maximum(rights[current_max_score_index], rights[score_order[1:]])
        intersections_tops = np.maximum(tops[current_max_score_index], tops[score_order[1:]])

        intersections_widths = np.maximum(0.0, intersections_rights - intersections_lefts + 1)
        intersections_heights = np.maximum(0.0, intersections_tops - intersections_bottoms + 1)

        intersections_areas = intersections_widths * intersections_heights
        iou = intersections_areas / (box_areas[current_max_score_index]
                                     + box_areas[score_order[1:]] - intersections_areas)
        suppressed_boxes_index = np.where(iou <= thresh)[0]
        score_order = score_order[suppressed_boxes_index + 1]
    return keep_indexes

def test():
    boxes_and_scores = np.array([[1.0, 2.0, 4.0, 7.0, 0.88],
                                 [0.8, 1.2, 3.3, 6.9, 0.86],
                                 [1.2, 2.2, 4.5, 6.5, 0.93],
                                 [1.5, 1.7, 4.3, 7.2, 0.95],
                                 [1.33, 1.85, 4.2, 7.15, 0.92]])
    kept_indexes = non_maximum_suppress(boxes_and_scores, 0.96)
    print(boxes_and_scores[kept_indexes])


if __name__ == '__main__':
    test()