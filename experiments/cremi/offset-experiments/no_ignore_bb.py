import os
import h5py
import numpy as np
import vigra
import json
from math import sqrt

from cremi.evaluation import NeuronIds
from cremi import Volume
from skunkworks.postprocessing.watershed.dam_ws import DamWatershed


def cremi_scores(seg, gt):
    if 0 in gt:
        gt[gt == 0] = -1
    seg = Volume(seg)
    metric = NeuronIds(Volume(gt))
    vis, vim = metric.voi(seg)
    are = metric.adapted_rand(seg)
    # cremi uses the geometric mean of rand and vi !
    cs = sqrt(are * (vis + vim))
    return {'cremi-score': cs, 'vi-merge': vim, 'vi-split': vis, 'adapted-rand': are}


def check_prediction(sample):
    bb = np.s_[95:]
    gt_path = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cremi/sample%s/gt/sample%s_neurongt_automatically_realignedV2.h5' % (sample, sample)
    with h5py.File(gt_path, 'r') as f:
        gt = f['data'][bb].astype('int64')

    ignore_mask = gt == 0
    project_directory = '/groups/saalfeld/home/papec/Work/neurodata_hdd/networks/neurofire/mws'
    pred_path = os.path.join(project_directory,
                             'Predictions',
                             'prediction_sample%s.h5' % sample)
    prediction = vigra.readHDF5(pred_path, 'data')

    stride = [2, 10, 10]
    with open('./mws_offsets.json', 'r') as f:
        affinity_offsets = json.load(f)
    mws = DamWatershed(affinity_offsets, stride, randomize_bounds=False)
    print('Predicting mws.. %s' % sample)
    mws_seg = mws(prediction).astype('int64')
    print('.. done, sample %s' % sample)

    gt = gt[ignore_mask]
    mws_seg = mws_seg[ignore_mask]
    assert gt.shape == mws_seg.shape
    print(cremi_scores(mws_seg, gt))


if __name__ == '__main__':
    check_prediction("A")
