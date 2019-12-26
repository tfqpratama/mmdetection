from argparse import ArgumentParser

import mmcv
import numpy as np

from mmdet import datasets
from mmdet.core import eval_map


def clean_result(det_result, score_thr=0.3):
    # det_result = det_result[:class_count]
    for i in range(len(det_result)):
        if det_result[i].size != 0:
            det_result[i] = det_result[i][det_result[i][:,-1] > score_thr]
    return det_result

def voc_eval(result_file, dataset_list, score_thr, iou_thr=0.5):
    det_results = mmcv.load(result_file)
    for i in range(len(det_results)):
        det_results[i] = clean_result(det_results[i], score_thr=score_thr)

    gt_bboxes = []
    gt_labels = []
    gt_ignore = []

    for dataset in dataset_list:
        for i in range(len(dataset)):
            ann = dataset.get_ann_info(i)
            bboxes = ann['bboxes']
            labels = ann['labels']
            if 'bboxes_ignore' in ann:
                ignore = np.concatenate([
                    np.zeros(bboxes.shape[0], dtype=np.bool),
                    np.ones(ann['bboxes_ignore'].shape[0], dtype=np.bool)
                ])
                gt_ignore.append(ignore)
                bboxes = np.vstack([bboxes, ann['bboxes_ignore']])
                labels = np.concatenate([labels, ann['labels_ignore']])
            gt_bboxes.append(bboxes)
            gt_labels.append(labels)

    if not gt_ignore:
        gt_ignore = None

    dataset_name = dataset.CLASSES
    eval_map(
        det_results,
        gt_bboxes,
        gt_labels,
        gt_ignore=gt_ignore,
        scale_ranges=None,
        iou_thr=iou_thr,
        dataset=dataset_name,
        print_summary=True)


def main():
    parser = ArgumentParser(description='VOC Evaluation')
    parser.add_argument('result', help='result file path')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--score_thr', type=float, default=0.3, help='confidence threshold')
    parser.add_argument(
        '--iou-thr',
        type=float,
        default=0.5,
        help='IoU threshold for evaluation')
    args = parser.parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    test_dataset = []
    for ann_file, img_prefix in zip(cfg.data.test.ann_file.copy(), cfg.data.test.img_prefix.copy()):
        cfg.data.test.ann_file = ann_file
        cfg.data.test.img_prefix = img_prefix
        test_dataset.append(mmcv.runner.obj_from_dict(cfg.data.test, datasets))

    voc_eval(args.result, test_dataset, args.score_thr, args.iou_thr)


if __name__ == '__main__':
    main()
