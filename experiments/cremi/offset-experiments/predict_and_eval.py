import os
import numpy as np
import argparse
import vigra
import h5py
import json
from math import sqrt
from concurrent import futures
from threading import Lock

from skunkworks.datasets.cremi.loaders import RawVolumeWithDefectAugmentation
from skunkworks.inference import SimpleInferenceEngine
from skunkworks.postprocessing import local_affinity_multicut_from_wsdt2d
from skunkworks.postprocessing.watershed.dam_ws import DamWatershed
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
    # cremi uses the geometric mean of rand and vi !
    cs = sqrt(are * (vis + vim))
    return {'cremi-score': cs, 'vi-merge': vim, 'vi-split': vis, 'adapted-rand': are}


def predict(project_folder,
            sample,
            gpu,
            only_nn_channels=False):

    checkpoint = os.path.join(project_folder, 'Weights')
    # TODO we need different configs for u-net and HED
    data_config_file = './template_config/prediction_configs/sample%s.yml' % sample
    print("[*] Loading CREMI with Configuration at: {}".format(data_config_file))
    # Load CREMI sample
    cremi = RawVolumeWithDefectAugmentation.from_config(data_config_file)

    # Load model
    trainer = Trainer().load(from_directory=checkpoint, best=True)
    model = trainer.model.cuda(gpu)

    print("[*] Start inference on gpu:", gpu)
    inference_config_file = './template_config/prediction_configs/inference_config.yml'
    inference_engine = SimpleInferenceEngine.from_config(inference_config_file, model, gpu)
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
    return output.astype('float32')


def evaluate_mc(project_folder, sample, prediction, bb):
    pred_path = os.path.join(project_directory,
                             'Predictions',
                             'prediction_sample%s_nnaffinities.h5' % sample)
    # prediction = vigra.readHDF5(pred_path, 'data')
    multicutter = local_affinity_multicut_from_wsdt2d(n_threads=12)
    mc_seg = multicutter(prediction).astype('int64')

    gt_path = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cremi/sample%s/gt/sample%s_neurongt_automatically_realignedV2.h5' % (sample, sample)
    with h5py.File(gt_path, 'r') as f:
        gt = f['data'][bb].astype('int64')
    assert gt.shape == mc_seg.shape
    vigra.writeHDF5(mc_seg, pred_path, 'multicut', compression='gzip')

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


def evaluate_mws(project_folder, sample, prediction, bb):
    stride = [2, 10, 10]
    with open('./mws_offsets.json', 'r') as f:
        affinity_offsets = json.load(f)
    mws = DamWatershed(affinity_offsets, stride, randomize_bounds=False)
    print('Predicting mws.. %s' % sample)
    mws_seg = mws(prediction).astype('int64')
    print('.. done, sample %s' % sample)

    gt_path = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cremi/sample%s/gt/sample%s_neurongt_automatically_realignedV2.h5' % (sample, sample)
    with h5py.File(gt_path, 'r') as f:
        gt = f['data'][bb].astype('int64')
    assert gt.shape == mws_seg.shape
    pred_path = os.path.join(project_directory,
                             'Predictions',
                             'prediction_sample%s.h5' % sample)
    vigra.writeHDF5(mws_seg, pred_path, 'mws', compression='gzip')
    quit()

    evals = cremi_scores(mws_seg, gt)
    with Lock():
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
    args = parser.parse_args()
    project_directory = args.project_directory

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(8)))
    bb = np.s_[95:]

    def eval_sample(sample, gpu):
        # prediction = predict(project_directory, sample, gpu)
        print("Loading data for sample", sample, "...")
        pred_path = os.path.join(project_directory,
                                 'Predictions',
                                 'prediction_sample%s.h5' % sample)
        prediction = vigra.readHDF5(pred_path, 'data')
        print("... done")
        evaluate_mws(project_directory, sample, prediction, bb)

    samples = ('A', 'B', 'C')
    gpus = (5, 6, 7)
    with futures.ThreadPoolExecutor(3) as tp:
        tasks = [tp.submit(eval_sample, sample, gpu) for sample, gpu in zip(samples, gpus)]
        [t.result() for t in tasks]
    # for sample, gpu in zip(samples, gpus):
    #     eval_sample(sample, gpu)
