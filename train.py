from pytorch_lightning.callbacks import ModelCheckpoint
import os
import argparse 
import gc
import datetime
import numpy as np
import pandas as pd

import numpy as np
import torch

import pytorch_lightning as pl
from lightning_module import LitModel
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
SEED = 1
torch.manual_seed(SEED)
np.random.seed(SEED)


def main(hparams):
    """
    Main training routine specific for this project
    :param hparams:
    """
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
#    print(hparams)
    model = LitModel(vars(hparams))
    # model.__build_model__()

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    name = "runs/f_t-"+"{date:%d-%m-%Y_%H:%M:%S}".format(
                                                date=datetime.datetime.now())
                                                
    logger = TensorBoardLogger("tb_logs", name=name)
    # logger.log_graph(model)
    # logger.log_hyperparams(hparams)
    checkpoint_callback = ModelCheckpoint(
        # filename='{epoch}-{val_mse:.2f}.ckpt',
        filename='model',
        # dirpath=os.path.join(os.getcwd(), 'tb_logs/', name),  # "/"
        # filepath=os.path.join(os.getcwd(), 'tb_logs/', name + '_checkpoints'),
        # dirpath=os.path.join(os.getcwd(), 'tb_logs/', name + '_checkpoints'),
        save_last=True,
        save_top_k=1,
        verbose=True,
        monitor='loss',
        mode='min',
        )
    print([hparams.first_gpu+el for el in range(hparams.gpus)])

    trainer = pl.Trainer(
        max_epochs=1000000,
        callbacks=[checkpoint_callback],
        precision=64,
        logger=logger,
        # distributed_backend=hparams.distributed_backend,
        # gpus=1,  # [hparams.first_gpu+el for el in range(hparams.gpus)]
        # num_nodes=4
        # use_amp=hparams.use_16bit
        # check_val_every_n_epoch=2,
        # auto_scale_batch_size='binsearch',
        # accumulate_grad_batches=2,
        # fast_dev_run=True,
        # accumulate_grad_batches=hparams.acc_batches,
        # auto_lr_find=hparams.lr,
        # weights_summary="full"  ??????
        )

    lr_finder = trainer.tuner.lr_find(model)
    lr_finder.results
    print(lr_finder.results)
    # fig = lr_finder.plot(suggest=True)  # Plot
    # fig.show()
    trainer.tune(model)



    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model)



if __name__ == '__main__':
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments

    root_dir = os.path.dirname(os.path.realpath(__file__))
    # each LightningModule defines arguments relevant to it
    parser = LitModel.add_model_specific_args()
    # parser.get_default()
    parser.print_help()
    parser.print_usage()

    hyperparams = parser.parse_args()

    hyperparams.cutoff_inp = 11  # 20 * 0.529772  # conversion angstrom to a0 atomic unit   10.59544 # 5
    hyperparams.band_reducer = 16  # number_of_electrons = highest_occupied * 2  # 8
    hyperparams.file = "MoS2_modPBE0_vasprun.xml"
    # hyperparams.file = "graphite.xml.bz2"
    # hyperparams.file = "diamond.xml.bz2"
    # hyperparams.file = "Zn.xml.bz2"
    # hyperparams.file = "Al.xml.bz2"

    hyperparams.gauss = True
    hyperparams.spd = False
    hyperparams.lr = 1e-10
    hyperparams.model_poly = 22
    hyperparams.model_gauss = 3
    # hyperparams.num_nodes = 1


    # ---------------------
    # RUN TRAINING
    # ---------------------
    print(str(os.path.join(os.getcwd(), 'tb_logs/',
                           "runs/f_t-"+"{date:%d-%m-%Y_%H:%M:%S}".format(date=datetime.datetime.now()))))
    print(type(hyperparams), hyperparams)

    main(hyperparams)
