import os
import sys
import logging
import argparse
import yaml

# FIXME needed to prevent segfault at import ?!
import vigra

from inferno.trainers.basic import Trainer
from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger
# from inferno.trainers.callbacks.scheduling import AutoLR
from inferno.trainers.callbacks.scheduling import ManualLR
from inferno.utils.io_utils import yaml2dict
# from inferno.trainers.callbacks.essentials import SaveAtBestValidationScore
from inferno.io.transform.base import Compose

import neurofire.models as models
from neurofire.criteria.loss_wrapper import LossWrapper
from neurofire.criteria.loss_transforms import MaskIgnoreLabel, RemoveSegmentationFromTarget  # , InvertTarget
from neurofire.datasets.merge_net import get_cremi_merge_loaders

# Do we implement this in neurofire again ???
# from skunkworks.datasets.cremi.criteria import Euclidean, AsSegmentationCriterion

# Import the different creiterions, we support.
from inferno.extensions.criteria import SorensenDiceLoss

logging.basicConfig(format='[+][%(asctime)-15s][%(name)s %(levelname)s]'
                           ' %(message)s',
                    stream=sys.stdout,
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def set_up_training(project_directory,
                    config,
                    data_config,
                    load_pretrained_model,
                    max_iters):
    # Get model
    if load_pretrained_model:
        model = Trainer().load(from_directory=project_directory,
                               filename='Weights/checkpoint.pytorch').model
    else:
        model_name = config.get('model_name')
        model = getattr(models, model_name)(**config.get('model_kwargs'))

    loss = LossWrapper(criterion=SorensenDiceLoss(),
                       transforms=Compose(MaskIgnoreLabel(),
                                          RemoveSegmentationFromTarget()))
    # TODO loss transforms:
    # - Invert Target ???

    # Build trainer and validation metric
    logger.info("Building trainer.")
    # smoothness = 0.95

    # TODO set up validation ?!
    trainer = Trainer(model)\
        .save_every((1000, 'iterations'), to_directory=os.path.join(project_directory, 'Weights'))\
        .build_criterion(loss)\
        .build_optimizer(**config.get('training_optimizer_kwargs'))\
        .evaluate_metric_every('never')\
        .register_callback(ManualLR(decay_specs=[((k * 100, 'iterations'), 0.99)
                                                 for k in range(1, max_iters // 100)]))
    # .validate_every((100, 'iterations'), for_num_iterations=1)\
    # .register_callback(SaveAtBestValidationScore(smoothness=smoothness, verbose=True))\
    # .build_metric(metric)\
    # .register_callback(AutoLR(factor=0.98,
    #                           patience='100 iterations',
    #                           monitor_while='validating',
    #                           monitor_momentum=smoothness,
    #                           consider_improvement_with_respect_to='previous'))

    logger.info("Building logger.")
    # Build logger
    tensorboard = TensorboardLogger(log_scalars_every=(1, 'iteration'),
                                    log_images_every=(100, 'iterations'))  # .observe_states(
    #     ['validation_input', 'validation_prediction, validation_target'],
    #     observe_while='validating'
    # )

    trainer.build_logger(tensorboard, log_directory=os.path.join(project_directory, 'Logs'))
    return trainer


def load_checkpoint(project_directory):
    logger.info("Trainer from checkpoint")
    trainer = Trainer().load(from_directory=os.path.join(project_directory, "Weights"))
    return trainer


def training(project_directory,
             train_configuration_file,
             data_configuration_file,
             validation_configuration_file=None,
             max_training_iters=int(1e5),
             from_checkpoint=False,
             load_pretrained_model=False):

    assert not (from_checkpoint and load_pretrained_model)

    logger.info("Loading config from {}.".format(train_configuration_file))
    config = yaml2dict(train_configuration_file)
    data_config = yaml2dict(data_configuration_file)

    # load network and training progress from checkpoint
    if from_checkpoint:
        trainer = load_checkpoint()
    else:
        trainer = set_up_training(project_directory,
                                  config,
                                  data_config,
                                  load_pretrained_model,
                                  max_training_iters)

    logger.info("Loading training data loader from %s." % data_configuration_file)
    train_loader = get_cremi_merge_loaders(data_configuration_file)
    if validation_configuration_file is not None:
        logger.info("Loading validation data loader from %s." % validation_configuration_file)
        validation_loader = get_cremi_merge_loaders(validation_configuration_file)

    trainer.set_max_num_iterations(max_training_iters)

    # Bind loader
    logger.info("Binding loaders to trainer.")
    trainer.bind_loader('train', train_loader)
    if validation_configuration_file is not None:
        trainer.bind_loader('validate', validation_loader)

    # Set devices
    if config.get('devices'):
        logger.info("Using devices {}".format(config.get('devices')))
        trainer.cuda(config.get('devices'))

    # Go!
    logger.info("Lift off!")
    trainer.fit()


def make_train_config(train_config_file, distances, gpus):
    template = yaml2dict('./template_config/train_config.yml')
    template['devices'] = gpus
    template['model_kwargs']['out_channels'] = len(distances)
    with open(train_config_file, 'w') as f:
        yaml.dump(template, f)


def make_data_config(data_config_file, distances, n_batches):
    template = yaml2dict('./template_config/data_config.yml')
    template['master_config']['false_merge_config']['target_distances'] = distances
    template['loader_config']['batch_size'] = n_batches
    template['loader_config']['num_workers'] = 10 * n_batches
    with open(data_config_file, 'w') as f:
        yaml.dump(template, f)


def make_validation_config(validation_config_file, offsets):
    template = yaml2dict('./template_config/validation_config.yml')
    template['volume_config']['segmentation']['affinity_offsets'] = offsets
    with open(validation_config_file, 'w') as f:
        yaml.dump(template, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('project_directory', type=str)
    parser.add_argument('--distances', nargs='+', default=[5, 10, 15, 25, 50], type=int)
    parser.add_argument('--gpus', nargs='+', default=[1, 2], type=int)
    parser.add_argument('--max_train_iters', type=int, default=int(1e5))

    args = parser.parse_args()

    project_directory = args.project_directory
    if not os.path.exists(project_directory):
        os.mkdir(project_directory)

    gpus = list(args.gpus)
    # set the proper CUDA_VISIBLE_DEVICES env variables
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
    gpus = list(range(len(gpus)))

    distances = list(args.distances)

    train_config = os.path.join(project_directory, 'train_config.yml')
    make_train_config(train_config, distances, gpus)

    data_config = os.path.join(project_directory, 'data_config.yml')
    make_data_config(data_config, distances, len(gpus))

    # TODO reactivate validation
    # validation_config = os.path.join(project_directory, 'validation_config.yml')
    # make_validation_config(validation_config)

    # TODO make accessible:
    # - starting training from checkpoint
    # - loading pretrained model
    training(project_directory,
             train_config,
             data_config,
             validation_configuration_file=None,
             max_training_iters=args.max_train_iters)


if __name__ == '__main__':
    main()
