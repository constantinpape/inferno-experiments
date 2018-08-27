#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/inferno/bin/python

import os
import json
import argparse
import h5py
from concurrent import futures

import numpy as np

from inferno.trainers.basic import Trainer
from inferno.utils.io_utils import yaml2dict
from skunkworks.inference import SimpleInferenceEngine
from neurofire.datasets.isbi2012.loaders.raw import RawVolumeHDF5

from inferno.extensions.metrics.arand import adapted_rand
from inferno.extensions.metrics.voi import voi


def load_volume(inference_config):
    config = yaml2dict(inference_config)
    vol_config = config['volume_config']['raw']
    slicing_config = config['slicing_config']
    return RawVolumeHDF5(**vol_config, **slicing_config)


def run_inference(project_dir, out_file, inference_config):

    print("Loading model...")
    model = Trainer().load(from_directory=os.path.join(project_dir, "Weights"), best=True).model
    print("Loading dataset...")
    dataset = load_volume(inference_config)

    engine = SimpleInferenceEngine.from_config(inference_config, model)
    print("Run prediction...")
    out = engine.infer(dataset)
    if out_file != '':
        print("Save prediction to %s ..." % out_file)
        with h5py.File(out_file, 'w') as f:
            f.create_dataset('data', data=out, compression='gzip')
    return out


def cc_segmenter(prediction, thresholds=[.9, .925, .95, .975, .99], invert=True):
    from affogato.segmentation import connected_components
    if invert:
        prediction = 1. - prediction
    return [connected_components(prediction, thresh)[0]
            for thresh in thresholds]


def zws_segmenter(prediction, thresholds=[0.5], invert=True):
    from affogato.segmentation import compute_zws_segmentation
    if invert:
        prediction = 1. - prediction
    # parameters that are not exposed
    lower_thresh = 0.2
    higher_thresh = 0.98
    size_thresh = 25
    return [compute_zws_segmentation(prediction, lower_thresh, higher_thresh, thresh, size_thresh)
            for thresh in thresholds]


def mws_segmenter(prediction, offset_version='v2'):
    from affogato.segmentation import compute_mws_segmentation
    from train_affs import get_default_offsets, get_mws_offsets
    assert offset_version in ('v1', 'v2')
    offsets = get_default_offsets() if offset_version == 'v1' else get_mws_offsets()
    # invert the lr channels
    prediction[:2] *= -1
    prediction[:2] += 1
    # TODO change this api
    return compute_mws_segmentation(prediction, offsets, 2, strides=[4, 4])


def cremi_score(seg, gt):
    assert seg.shape == gt.shape
    rand = 1. - adapted_rand(seg, gt)[0]
    vis, vim = voi(seg, gt)
    cs = np.sqrt((vis + vim) * rand)
    return cs, vis, vim, rand


def evaluate(prediction, algo='cc'):
    # get the segmentation algorithm
    if algo == 'cc':
        segmenter = cc_segmenter
    elif algo == 'mws':
        # TODO expose offset version somehow
        segmenter = mws_segmenter
    elif algo == 'zws':
        segmenter = zws_segemnter
    else:
        raise NotImplementedError('Algorithm %s not implemented' % algo)

    gt_path = '/g/kreshuk/data/isbi2012_challenge/vnc_train_volume.h5'
    with h5py.File(gt_path) as f:
        gt = f['volumes/labels/neuron_ids_3d'][:]
    assert gt.shape == prediction.shape[1:]

    def eval_z(z):
        gtz = gt[z]
        seg = segmenter(prediction[:, z])
        if isinstance(seg, list):
            score = [cremi_score(sg, gtz) for sg in seg]
            # print(score)
            max_index = np.argmax([sc[0] for sc in score])
            score = score[max_index]
        else:
            score = cremi_score(seg, gtz)
        return score

    with futures.ThreadPoolExecutor(5) as tp:
        tasks = [tp.submit(eval_z, z) for z in range(prediction.shape[1])]
        scores = np.mean([t.result() for t in tasks], axis=0)
    # print(scores[0], scores[1], scores[2], scores[3])
    return scores


def view_res(prediction):
    from cremi_tools.viewer.volumina import view
    raw_path = '/home/constantin/Work/neurodata_hdd/isbi12_data/isbi2012_test_volume.h5'
    with h5py.File(raw_path, 'r') as f:
        raw = f['volumes/raw'][:]
    view([raw, prediction.transpose((1, 2, 3, 0))])


def main(project_dir, out_file, inference_config, key,
         algorithm='cc', view_result=False):
    out = run_inference(project_dir, out_file, inference_config)

    if view_result:
        view_res(out)

    if algorithm in ('no', ''):
        return

    score = evaluate(out, algorithm)
    if algorithm != 'cc':
        key += '_' + algorithm
    if os.path.exists('results.json'):
        with open('results.json') as f:
            results = json.load(f)
        if key in results:
            raise RuntimeError("Key %s is already in results, will not override !" % key)
    else:
        results = {}
    results[key] = {'cremi-score': score[0],
                    'vi-split': score[1],
                    'vi-merge': score[2],
                    'rand': score[3]}
    with open('results.json', 'w') as f:
        json.dump(results, f, sort_keys=True, indent=4)


def set_device(device):
    print("Setting cuda devices to", device)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('project_directory', type=str)
    parser.add_argument('result_key', type=str)
    parser.add_argument('--out_file', type=str, default='')
    parser.add_argument('--inference_config', type=str, default='template_config/inf_config.yml')
    parser.add_argument('--algorithm', type=str, default='cc')
    parser.add_argument('--view_result', type=int, default=0)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    if args.device != 0:
        set_device(args.device)
    main(args.project_directory, args.out_file, args.inference_config, args.result_key, args.algorithm,
         bool(args.view_result))
