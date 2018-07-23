import os
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
    else:
        raise NotImplementedError()

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('project_directory', type=str)
    parser.add_argument('out_file', type=str)
    parser.add_argument('--inference_config', type=str, default='template_config/inf_config.yml')
    args = parser.parse_args()

    out = run_inference(args.project_directory, args.out_file, args.inference_config)
    # with h5py.File('pred.h5') as f:
    #    out = f['data'][:]
    score = evaluate(out)
    # TODO serialize the score properly
    print(scores[0], scores[1], scores[2], scores[3])
