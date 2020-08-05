import numpy as np


def cal_accuracy_iou(preds, labels, seg_classes, pt=True):
    '''

    :param pred: shape=(B, N)
    :param labels: shape=(B, N)
    :param seg_classes: dict: cat->labels
    :return:
    '''
    nclasses, n = len(seg_classes), len(preds)
    shape_ious = {cat: [] for cat in seg_classes}
    shape_count = {cat: 0.0 for cat in seg_classes}
    shape_points_seen = {cat: 0.0 for cat in seg_classes}
    shape_points_correct = {cat: 0.0 for cat in seg_classes}
    seg2cat = {}
    for k, vs in seg_classes.items():
        for v in vs:
            seg2cat[v] = k
    for i in range(n):
        pred, label = preds[i], labels[i]
        npoints = len(pred)
        cat = seg2cat[label[0]]
        shape_count[cat] += 1
        shape_points_seen[cat] += npoints
        shape_points_correct[cat] += np.sum(pred == label)
        part_ious = []
        for l in seg_classes[cat]:
            intersection = np.sum(np.all([pred == l, label == l], axis=0))
            union = np.sum(np.any([pred == l, label == l], axis=0))
            if union < 1:
                part_ious.append(1.0)
                continue
            part_ious.append(intersection / union)
        shape_ious[cat].append(np.mean(part_ious))

    if pt:
        print('='*40)
    weighted_acc = 0.0
    weighted_average_iou = 0.0
    accs, ious = [], []
    for cat in sorted(seg_classes.keys()):
        acc = shape_points_correct[cat] / float(shape_points_seen[cat])
        iou = np.mean(shape_ious[cat])
        if pt:
            print('{} | acc: {:.4f}, iou: {:.4f}'.format(cat, acc, iou))
        accs.append(round(acc * 100, 1))
        ious.append(round(iou * 100, 1))
        weighted_acc += shape_count[cat] * acc
        weighted_average_iou += shape_count[cat] * iou
    #print('accs: ', accs)
    #print('ious: ', ious)
    weighted_acc = weighted_acc / np.sum(list(shape_count.values())).astype(np.float32)
    weighted_average_iou = weighted_average_iou / np.sum(list(shape_count.values())).astype(np.float32)
    return weighted_average_iou, weighted_acc