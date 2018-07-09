import os
import sys
import logging
import argparse
import yaml

# FIXME needed to prevent segfault at import ?!
# import vigra

from inferno.trainers.basic import Trainer
from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger
from inferno.trainers.callbacks.scheduling import AutoLR
from inferno.utils.io_utils import yaml2dict
from inferno.trainers.callbacks.essentials import SaveAtBestValidationScore
from inferno.io.transform.base import Compose

import neurofire.models as models
from neurofire.criteria.loss_wrapper import LossWrapper
from neurofire.criteria.loss_transforms import ApplyAndRemoveMask, RemoveSegmentationFromTarget
from neurofire.criteria.multi_scale_loss import MultiScaleLoss
from neurofire.metrics.arand import ArandErrorFromConnectedComponentsOnAffinities

from skunkworks.datasets.isbi2012.loaders import get_isbi_loader

# TODO try different loss functions ?!
from inferno.extensions.criteria import SorensenDiceLoss

# validate by connected components on affinities


logging.basicConfig(format='[+][%(asctime)-15s][%(name)s %(levelname)s]'
                           ' %(message)s',
                    stream=sys.stdout,
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def set_up_training(project_directory,
                    config,
                    data_config,
                    load_pretrained_model,
                    n_scales=4):
    # Get model
    if load_pretrained_model:
        model = Trainer().load(from_directory=project_directory,
                               filename='Weights/checkpoint.pytorch').model
    else:
        model_name = config.get('model_name')
        model = getattr(models, model_name)(**config.get('model_kwargs'))

    loss_train_ = LossWrapper(criterion=SorensenDiceLoss(),
                              transforms=ApplyAndRemoveMask())
    loss_val_ = LossWrapper(criterion=SorensenDiceLoss(),
                            transforms=Compose(RemoveSegmentationFromTarget(), ApplyAndRemoveMask()))

    if n_scales > 1:
        # TODO we don't need the multi-scale loss if we are training lr affinities
        # TODO how to weight scale levels ?
        # if n_scales == 6:
        #     #
        #     scale_weights = [1. / 2 ** scale for scale in range(n_scales - 1)]
        # else:
        #     scale_weights = [1. / 2 ** scale for scale in range(n_scales)]
        scale_weights = [1.] * n_scales
        loss_train = MultiScaleLoss(loss_train_, n_scales=n_scales,
                                    scale_weights=scale_weights,
                                    fill_missing_targets=True)
        loss_val = MultiScaleLoss(loss_val_, n_scales=n_scales,
                                  scale_weights=scale_weights,
                                  fill_missing_targets=True)
    else:
        loss_train = loss_train_
        loss_val = loss_val_

    # Build trainer and validation metric
    logger.info("Building trainer.")
    smoothness = 0.95

    # validate by connected components on affinities
    metric = ArandErrorFromConnectedComponentsOnAffinities()

    # TODO set validation loss
    trainer = Trainer(model)\
        .save_every((1000, 'iterations'), to_directory=os.path.join(project_directory, 'Weights'))\
        .build_criterion(loss_train)\
        .build_validation_criterion(loss_val)\
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
             n_scales,
             max_training_iters=int(1e5),
             from_checkpoint=False,
             load_pretrained_model=False):

    assert not (from_checkpoint and load_pretrained_model)

    logger.info("Loading config from {}.".format(train_configuration_file))
    config = yaml2dict(train_configuration_file)

    logger.info("Loading training data loader from %s." % data_configuration_file)
    train_loader = get_isbi_loader(data_configuration_file)
    data_config = yaml2dict(data_configuration_file)

    logger.info("Loading validation data loader from %s." % validation_configuration_file)
    validation_loader = get_isbi_loader(validation_configuration_file)

    # load network and training progress from checkpoint
    if from_checkpoint:
        trainer = load_checkpoint()
    else:
        trainer = set_up_training(project_directory,
                                  config,
                                  data_config,
                                  load_pretrained_model,
                                   n_scales=n_scales)

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


def make_train_config(train_config_file, affinity_config, gpus, architecture):
    if architecture == 'mad':
        template = './template_config/train_config_mad.yml'
        n_out = 2
    else:
        if 'offsets' in affinity_config:
            template = './template_config/train_config_unet_lr.yml'
            n_out = len(affinity_config['offsets'])
        else:
            template = './template_config/train_config_unet_ms.yml'
            n_out = 2
    template = yaml2dict(template)
    template['model_kwargs']['out_channels'] = n_out
    template['devices'] = gpus
    with open(train_config_file, 'w') as f:
        yaml.dump(template, f)


def make_data_config(data_config_file, affinity_config, n_batches):
    template = yaml2dict('./template_config/data_config.yml')
    template['volume_config']['segmentation']['affinity_config'] = affinity_config
    template['loader_config']['batch_size'] = n_batches
    template['loader_config']['num_workers'] = 12 * n_batches
    with open(data_config_file, 'w') as f:
        yaml.dump(template, f)


def make_validation_config(validation_config_file, affinity_config):
    template = yaml2dict('./template_config/validation_config.yml')
    affinity_config.update({'retain_segmentation': True})
    template['volume_config']['segmentation']['affinity_config'] = affinity_config
    with open(validation_config_file, 'w') as f:
        yaml.dump(template, f)


def get_default_offsets():
    return [[-1, 0], [0, -1],
            [-3, 0], [0, -3],
            [-9, 0], [0, -9],
            [-27, 0], [0, -27]]


def get_default_block_shapes():
    return [[1, 1], [2, 2], [4, 4], [8, 8], [16, 16]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('project_directory', type=str)
    parser.add_argument('--train_multiscale', type=int, default=1)
    parser.add_argument('--architecture', type=str, default='mad')
    parser.add_argument('--gpus', nargs='+', default=[0], type=int)
    parser.add_argument('--max_train_iters', type=int, default=int(1e5))

    args = parser.parse_args()

    project_directory = args.project_directory
    if not os.path.exists(project_directory):
        os.mkdir(project_directory)

    train_multiscale = bool(args.train_multiscale)
    architecture = args.architecture
    assert architecture in ('mad', 'unet')
    if architecture == 'mad':
        assert train_multiscale
        n_scales = 6
    else:
        n_scales = 4

    # check if we train multiscale or long-range affinities
    affinity_config = {'retain_mask': True, 'ignore_label': None}
    if train_multiscale:
        block_shapes = get_default_block_shapes()[:n_scales]
        affinity_config['block_shapes'] = block_shapes
    else:
        offsets = get_default_offsets()
        affinity_config['offsets'] = offsets
        n_scales = 1

    gpus = list(args.gpus)
    # set the proper CUDA_VISIBLE_DEVICES env variables
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
    gpus = list(range(len(gpus)))

    train_config = os.path.join(project_directory, 'train_config.yml')
    make_train_config(train_config, affinity_config, gpus, architecture)

    data_config = os.path.join(project_directory, 'data_config.yml')
    make_data_config(data_config, affinity_config, len(gpus))

    validation_config = os.path.join(project_directory, 'validation_config.yml')
    make_validation_config(validation_config, affinity_config)

    # TODO make accessible:
    # - starting training from checkpoint
    # - loading pretrained model
    training(project_directory,
             train_config,
             data_config,
             validation_config,
             max_training_iters=args.max_train_iters,
             n_scales=n_scales)


if __name__ == '__main__':
    main()
