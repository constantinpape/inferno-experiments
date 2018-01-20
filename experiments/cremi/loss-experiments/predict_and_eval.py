import os
import numpy as np
import argparse
import vigra
import h5py
import json

from skunkworks.datasets.cremi.loaders import RawVolumeWithDefectAugmentation
from skunkworks.inference import SimpleInferenceEngine
from skunkworks.postprocessing import local_affinity_multicut_from_wsdt2d
from inferno.trainers.basic import Trainer
from inferno.utils.io_utils import toh5

from cremi.evaluation import NeuronIds
from cremi import Volume


def cremi_scores(seg, gt):
    gt[gt == 0] = -1
    seg = Volume(seg)
    metric = NeuronIds(Volume(gt))
    vis, vim = metric.voi(seg)
    are = metric.adapted_rand(seg)
    cs = (are + vis + vim) / 3
    return {'cremi-score': cs, 'vi-merge': vim, 'vi-split': vis, 'adapted-rand': are}


def predict(project_folder,
            sample,
            only_nn_channels=True):

    gpu = 0
    checkpoint = os.path.join(project_folder, 'Weights')
    data_config_file = './template_config/prediction_configs/sample%s.yml' % sample
    print("[*] Loading CREMI with Configuration at: {}".format(data_config_file))
    # Load CREMI sample
    cremi = RawVolumeWithDefectAugmentation.from_config(data_config_file)

    # Load model
    trainer = Trainer().load(from_directory=checkpoint, best=True).cuda(gpu)
    model = trainer.model

    inference_config_file = './template_config/prediction_configs/inference_config.yml'
    inference_engine = SimpleInferenceEngine.from_config(inference_config_file, model)
    output = inference_engine.infer(cremi)

    print("[*] Output has shape {}".format(str(output.shape)))
    save_folder = os.path.join(project_folder, 'Predictions')
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    save_path = os.path.join(save_folder, 'prediction_sample%s.h5' % sample)

    if only_nn_channels:
        output = output[:3]
        save_path = save_path[:-3] + '_nnaffinities.h5'

    toh5(output.astype('float32'), save_path, compression='lzf')


def evaluate(project_folder, sample):
    prediction = vigra.readHDF5(os.path.join(project_directory,
                                             'Predictions',
                                             'prediction_sample%s_nnaffinities.h5' % sample), 'data')
    multicutter = local_affinity_multicut_from_wsdt2d(n_threads=12)
    mc_seg = multicutter(prediction)

    gt_path = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cremi/sample%s/gt/sample%s_neurongt_automatically_realignedV2.h5' % (sample, sample)
    bb = np.s_[85:]
    with h5py.File(gt_path, 'r') as f:
        gt = f['data'][bb].astype('int64')
    assert gt.shape == mc_seg.shape
    evals = cremi_scores(mc_seg, gt)

    eval_file = os.path.join(project_directory, 'evaluation.json')
    if os.path.exists(eval_file):
        with open(eval_file, 'r') as f:
            res = json.load(f)
    else:
        res = {}

    res[sample] = evals
    with open(eval_file, 'w') as f:
        json.dump(res, f, indent=4, sort_keys=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('project_directory', type=str)
    parser.add_argument('gpu', type=int)
    args = parser.parse_args()

    project_directory = args.project_directory
    gpu = args.gpu

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    # samples = ('A', 'B', 'C')
    samples = ('A',)
    # for sample in samples:
    #     predict(project_directory, sample)

    for sample in samples:
        evaluate(project_directory, sample)
