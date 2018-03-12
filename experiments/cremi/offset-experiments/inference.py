import os
import argparse
import vigra
import json
import h5py
import z5py
from concurrent import futures
import numpy as np

from dam_ws import DamWatershed
# from skunkworks.datasets.cremi.loaders import RawVolumeWithDefectAugmentation
# from skunkworks.inference import SimpleInferenceEngine
# from inferno.trainers.basic import Trainer


def inference(project_folder, sample, gpu,
              prefix=None, only_nn_channels=False):

    data_config_file = './template_config/prediction_configs/sample%s.yml' % sample
    cremi = RawVolumeWithDefectAugmentation.from_config(data_config_file)

    # Load model
    checkpoint = os.path.join(project_folder, 'Weights')
    trainer = Trainer().load(from_directory=checkpoint, best=True)
    model = trainer.model.cuda(gpu)

    print("[*] Start inference on gpu:", gpu)
    inference_config_file = './template_config/prediction_configs/inference_config.yml'
    inference_engine = SimpleInferenceEngine.from_config(inference_config_file, model, gpu)
    output = inference_engine.infer(cremi).astype('float32')

    print("[*] Output has shape {}".format(str(output.shape)))
    save_folder = os.path.join(project_folder, 'Predictions')
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    if prefix is None:
        save_path = os.path.join(save_folder, 'prediction_sample%s.h5' % sample)
    else:
        save_path = os.path.join(save_folder, 'prediction_%s_sample%s.h5' % (prefix, sample))

    if only_nn_channels:
        output = output[:3]
        save_path = save_path[:-3] + '_nnaffinities.h5'

    vigra.writeHDF5(output, save_path, 'data', compression='gzip')
    return output


def run_mws(project_directory, sample):
    print("Loading predictions...")
    ds = z5py.File(os.path.join(project_directory, 'Predictions',
                                'prediction_sample%s.n5' % sample))['full_affs']
    shape = ds.shape[1:]
    shape_diff = (shape[0] - 125) // 2
    bb = np.s_[shape_diff-3:shape[0]-(shape_diff-3),:]
    prediction = ds[(slice(None),) + bb]
    print("... done")
    print(prediction.shape)
    stride = [2, 10, 10]
    with open('./mws_offsets.json', 'r') as f:
        affinity_offsets = json.load(f)
    mws = DamWatershed(affinity_offsets, stride, randomize_bounds=False)
    print('Predicting mws.. %s' % sample)
    mws_seg = mws(prediction).astype('int64')
    print('.. done, sample %s' % sample)
    seg_path = os.path.join(project_directory, 'Predictions', 'sample%s_mws_seg_original.h5' % sample)
    with h5py.File(seg_path) as f:
        out = f.create_dataset('volumes/labels/neuron_ids', shape=shape, dtype='uint64', compression='gzip')
        out[shape_diff-3:shape[0]-(shape_diff-3)] = mws_seg


def warp_to_cremi(project_directory, sample):
    from cremi_tools.alignment import backalign_segmentation
    path = os.path.join(project_directory, 'Predictions', 'sample%s_mws_seg_original.h5' % sample)
    out_path = os.path.join(project_directory, 'Predictions', 'sample%s_mws_seg_cremi.h5' % sample)
    backalign_segmentation(sample, path, out_path, postprocess=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('project_directory', type=str)
    args = parser.parse_args()
    project_directory = args.project_directory

    samples = ('A+',)

    # gpus = (5, 6, 7)
    # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(8)))
    # with futures.ThreadPoolExecutor(3) as tp:
    #     tasks = [tp.submit(inference, project_directory, sample, gpu)
    #              for sample, gpu in zip(samples, gpus)]
    #     predictions = [t.result() for t in tasks]

    # with futures.ThreadPoolExecutor(1) as tp:
    #     tasks = [tp.submit(run_mws, project_directory, sample)
    #              for i, sample in enumerate(samples)]
    #     [t.result() for t in tasks]

    with futures.ThreadPoolExecutor(3) as tp:
        tasks = [tp.submit(warp_to_cremi, project_directory, sample) for sample in samples]
        [t.result() for t in tasks]
