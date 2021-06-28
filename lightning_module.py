import pytorch_lightning as pl
# from main_fixed import DFTB, eval_vasp_xml, eval_model
from pytorch_relevant import DFTB, eval_vasp_xml, eval_model

from argparse import ArgumentParser
from torch import nn
import torch
from torch.utils.data import Dataset, DataLoader


class DFTBDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, kpts, true):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.true = torch.tensor(true)
        self.kpts = torch.tensor(kpts)

    def __len__(self):
        return len(self.kpts)

    def __getitem__(self, idx):
        return self.kpts[idx], self.true[idx]


class LitModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams.update(hparams)
        self.save_hyperparameters()
        # self.hparams=hparams
        self.cutoff_inp = self.hparams.cutoff_inp
        self.highest_occupied = self.hparams.highest_occupied
        self.spd = self.hparams.spd
        self.spin_up = self.hparams.spin_up
        self.slice_size = self.hparams.slice_size
        self.gauss = self.hparams.gauss
        self.model_poly = self.hparams.model_poly
        self.model_gauss = self.hparams.model_gauss
        self.criterion = nn.MSELoss()
        self.band_reducer = self.hparams.band_reducer
        self.learning_rate = self.hparams.lr
        self.__build_model__()

    def __build_model__(self):
        kpts_inp, weights_inp, lattice_inp, positions_inp, species_inp, species_dict_inp, true_inp, highest_occupied = \
            eval_vasp_xml(file=self.hparams.file,
                          recip=False,
                          print_out=False,
                          spin_up=self.spin_up,
                          slice_size=self.slice_size)
        self.positions_inp = nn.Parameter(positions_inp, requires_grad=False)
        self.species_inp = nn.Parameter(species_inp, requires_grad=False)
        self.species_dict_inp = species_dict_inp
        self.lattice_inp = nn.Parameter(lattice_inp, requires_grad=False)
        self.highest_occupied = nn.Parameter(highest_occupied, requires_grad=False)

        number_of_electrons = self.highest_occupied * 2  # 8
        params_inp, params_diag_inp = eval_model(species_dict=self.species_dict_inp,
                                                 model_poly=self.model_poly,
                                                 model_gauss=self.model_gauss,
                                                 print_out=False,
                                                 gauss=self.gauss,
                                                 spd=self.spd
                                                 )

        self.dataset = DFTBDataset(kpts_inp, true_inp)

        self.model = DFTB(n_species=len(species_dict_inp)-1,
                          cutoff=self.cutoff_inp,
                          lattice=self.lattice_inp,
                          diag_params=params_diag_inp,
                          highest_occupied=self.highest_occupied,
                          init_param_values=params_inp,
                          gauss=self.gauss,
                          spd=self.spd)

    def forward(self, k_pts):
        return self.model(self.positions_inp, self.species_inp, k_pts)

    def training_step(self, batch, batch_idx):
        k_pts, true = batch
        results = self(k_pts)
        # loss = self.criterion(true_up, results[:, :])
        # loss = self.criterion(true[:, :results.shape[1]], results[:, :])
        band_count = (min(results.shape[1], true.shape[1]) - self.band_reducer)
        loss = self.criterion(true[:, :band_count], results[:, :band_count])  # dftb_eig   true_up_inp
        self.log("loss", loss)  # for saving ??????????
        # self.save_hyperparameters()
        # self.log("hparams", self.hparams)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.dftb_params['V'].parameters(), lr=self.learning_rate)

    def train_dataloader(self):
        params = {"batch_size": len(self.dataset),
                  "pin_memory": False,
                  "shuffle": True,
                  "drop_last": True
                  }
        print('length of train_subset', len(self.dataset))
        train_generator = DataLoader(self.dataset, **params)
        return train_generator

    @staticmethod
    def add_model_specific_args():  # pragma: no-cover
        """
        Parameters you define here will be available to your model through self.hparams
        :param parser:
        :param root_dir:
        :return:
        """
        parser = ArgumentParser(fromfile_prefix_chars='@', add_help=True)
        parser.add_argument("--file",
                            type=str,
                            default="graphite.xml.bz2",
                            metavar="PATH",
                            help="dataset path")

        parser.add_argument("--gauss",
                            action="store_false",
                            help="Disable gauss")

        parser.add_argument("--spd",
                            action="store_false",  # "store_false"
                            help="enable spd orbitals")

        parser.add_argument("--highest_occupied",
                            default=4,
                            type=int,
                            help="highest_occupied orbital")

        parser.add_argument("--model_poly",
                            default=12,
                            type=int,
                            help="number of polynoms used in fit")

        parser.add_argument("--model_gauss",
                            default=3,
                            type=int,
                            help="number of gaussians used times 3")

        parser.add_argument("--cutoff_inp",
                            default=5.0,
                            type=float,
                            help="cutoff radius")

        parser.add_argument("--lr",
                            default=10e-7,
                            type=float,
                            help="learningrate")

        parser.add_argument('--gpus',
                            type=int,
                            default=1,
                            help='how many gpus')

        parser.add_argument('--acc_batches',
                            type=int,
                            default=1,
                            help='gpu number to use [first_gpu-first_gpu+gpus]')

        parser.add_argument('--distributed_backend',
                            type=str,
                            default='ddp',
                            help='supports three options dp, ddp, ddp2')

        parser.add_argument('--use_16bit',
                            dest='use_16bit',
                            action='store_true',
                            help='if true uses 16 bit precision')

        parser.add_argument('--first_gpu',
                            type=int,
                            default=0,
                            help='gpu number to use [first_gpu-first_gpu+gpus]')

        parser.add_argument("--slice_size",
                            default=1,
                            type=int,
                            help="parameter for slicing kpts")

        parser.add_argument("--spin_up",
                            action="store_false",
                            help="Choose if spin should be distinguished")

        parser.add_argument("--band_reducer",
                            default=2,
                            type=int,
                            help="choose amount of bands not fitted")

        parser.add_argument("--num_nodes",
                            default=1,
                            type=int,
                            help="choose amount of notes to compute on")

        return parser
