import os
import sys
import logging
import argparse
import yaml
import json

# FIXME needed to prevent segfault at import ?!
import vigra

from inferno.trainers.basic import Trainer
from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger
from inferno.trainers.callbacks.scheduling import AutoLR
from inferno.utils.io_utils import yaml2dict
from inferno.trainers.callbacks.essentials import SaveAtBestValidationScore
from inferno.io.transform.base import Compose

import neurofire.models as models
from neurofire.criteria.loss_wrapper import LossWrapper
from neurofire.criteria.loss_transforms import MaskTransitionToIgnoreLabel, RemoveSegmentationFromTarget, InvertTarget

# Do we implement this in neurofire again ???
# from skunkworks.datasets.cremi.criteria import Euclidean, AsSegmentationCriterion

from skunkworks.datasets.cremi.loaders import get_cremi_loaders_realigned

# Import the different creiterions, we support.
# TODO generalized sorensen dice and tversky loss
from inferno.extensions.criteria import SorensenDiceLoss

# validation
from skunkworks.metrics import ArandErrorFromSegmentationPipeline

# multicut pipeline
# from skunkworks.postprocessing.pipelines import local_affinity_multicut_from_wsdt2d
from skunkworks.postprocessing.watershed.dam_ws import DamWatershed

logging.basicConfig(format='[+][%(asctime)-15s][%(name)s %(levelname)s]'
                           ' %(message)s',
                    stream=sys.stdout,
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def set_up_training(project_directory,
                    config,
                    data_config,
                    load_pretrained_model):
    # Get model
    if load_pretrained_model:
        model = Trainer().load(from_directory=project_directory,
                               filename='Weights/checkpoint.pytorch').model
    else:
        model_name = config.get('model_name')
        model = getattr(models, model_name)(**config.get('model_kwargs'))

    affinity_offsets = data_config['volume_config']['segmentation']['affinity_offsets']
    loss = LossWrapper(criterion=SorensenDiceLoss(),
                       transforms=Compose(MaskTransitionToIgnoreLabel(affinity_offsets),
                                          RemoveSegmentationFromTarget(),
                                          InvertTarget()))

    # Build trainer and validation metric
    logger.info("Building trainer.")
    smoothness = 0.95

    # use multicut pipeline for validation
    # metric = ArandErrorFromSegmentationPipeline(local_affinity_multicut_from_wsdt2d(n_threads=10,
    #                                                                                 time_limit=120))

    # use damws for validation
    stride = [2, 10, 10]
    metric = ArandErrorFromSegmentationPipeline(DamWatershed(affinity_offsets, stride, randomize_bounds=False))
    trainer = Trainer(model)\
        .save_every((1000, 'iterations'), to_directory=os.path.join(project_directory, 'Weights'))\
        .build_criterion(loss)\
        .build_optimizer(**config.get('training_optimizer_kwargs'))\
        .evaluate_metric_every('never')\
        .validate_every((100, 'iterations'), for_num_iterations=1)\
        .register_callback(SaveAtBestValidationScore(smoothness=smoothness, verbose=True))\
        .build_metric(metric)\
        .register_callback(AutoLR(factor=0.98,
                                  patience='100 iterations',
                                  monitor_while='validating',
                                  monitor_momentum=smoothness,
                                  consider_improvement_with_respect_to='previous'))

    logger.info("Building logger.")
    # Build logger
    tensorboard = TensorboardLogger(log_scalars_every=(1, 'iteration'),
                                    log_images_every=(100, 'iterations')).observe_states(
        ['validation_input', 'validation_prediction, validation_target'],
        observe_while='validating'
    )

    trainer.build_logger(tensorboard, log_directory=os.path.join(project_directory, 'Logs'))
    return trainer


def load_checkpoint(project_directory):
    logger.info("Trainer from checkpoint")
    trainer = Trainer().load(from_directory=os.path.join(project_directory, "Weights"))
    return trainer


def training(project_directory,
             train_configuration_file,
             data_configuration_file,
             validation_configuration_file,
             max_training_iters=int(1e5),
             from_checkpoint=True,
             load_pretrained_model=False):

    assert not (from_checkpoint and load_pretrained_model)

    logger.info("Loading config from {}.".format(train_configuration_file))
    config = yaml2dict(train_configuration_file)

    logger.info("Loading training data loader from %s." % data_configuration_file)
    train_loader = get_cremi_loaders_realigned(data_configuration_file)
    data_config = yaml2dict(data_configuration_file)

    logger.info("Loading validation data loader from %s." % validation_configuration_file)
    validation_loader = get_cremi_loaders_realigned(validation_configuration_file)

    # load network and training progress from checkpoint
    if from_checkpoint:
        trainer = load_checkpoint(project_directory)
    else:
        trainer = set_up_training(project_directory,
                                  config,
                                  data_config,
                                  load_pretrained_model)

    trainer.set_max_num_iterations(max_training_iters)

    # Bind loader
    logger.info("Binding loaders to trainer.")
    trainer.bind_loader('train', train_loader).bind_loader('validate', validation_loader)

    # Set devices
    if config.get('devices'):
        logger.info("Using devices {}".format(config.get('devices')))
        trainer.cuda(config.get('devices'))

    # Go!
    logger.info("Lift off!")
    trainer.fit()


def make_train_config(train_config_file, offsets, gpus):
    template = yaml2dict('./template_config/train_config.yml')
    template['model_kwargs']['out_channels'] = len(offsets)
    template['devices'] = gpus
    with open(train_config_file, 'w') as f:
        yaml.dump(template, f)


def make_data_config(data_config_file, offsets, n_batches):
    template = yaml2dict('./template_config/data_config.yml')
    template['volume_config']['segmentation']['affinity_offsets'] = offsets
    template['loader_config']['batch_size'] = n_batches
    template['loader_config']['num_workers'] = 12 * n_batches
    with open(data_config_file, 'w') as f:
        yaml.dump(template, f)


def make_validation_config(validation_config_file, offsets):
    template = yaml2dict('./template_config/validation_config.yml')
    template['volume_config']['segmentation']['affinity_offsets'] = offsets
    with open(validation_config_file, 'w') as f:
        yaml.dump(template, f)


def parse_offsets(offset_file):
    assert os.path.exists(offset_file)
    with open(offset_file, 'r') as f:
        offsets = json.load(f)
    return offsets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('project_directory', type=str)
    parser.add_argument('offset_file', type=str)
    parser.add_argument('--gpus', nargs='+', default=[0, 1], type=int)
    parser.add_argument('--max_train_iters', type=int, default=int(1e5))

    args = parser.parse_args()

    project_directory = args.project_directory
    if not os.path.exists(project_directory):
        os.mkdir(project_directory)

    # We still leave options for varying the offsets
    # to be more flexible later variable
    offset_file = args.offset_file
    offsets = parse_offsets(offset_file)

    gpus = list(args.gpus)
    # set the proper CUDA_VISIBLE_DEVICES env variables
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
    gpus = list(range(len(gpus)))

    train_config = os.path.join(project_directory, 'train_config.yml')
    make_train_config(train_config, offsets, gpus)

    data_config = os.path.join(project_directory, 'data_config.yml')
    make_data_config(data_config, offsets, len(gpus))

    validation_config = os.path.join(project_directory, 'validation_config.yml')
    make_validation_config(validation_config, offsets)

    # TODO make accessible:
    # - starting training from checkpoint
    # - loading pretrained model
    training(project_directory,
             train_config,
             data_config,
             validation_config,
             max_training_iters=args.max_train_iters)


if __name__ == '__main__':
    main()
