# -*- coding: utf-8 -*-

# @Time    : 2021/1/3 20:56
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause

import os
import schnetpack as spk

import logging
import os
import re
import shutil
import tarfile
import tempfile
from urllib import request as request

import numpy as np
from ase.io.extxyz import read_xyz
from ase.units import Debye, Bohr, Hartree, eV

import schnetpack as spk
from schnetpack.datasets import DownloadableAtomsData, QM9

import schnetpack as spk

# from schnetpack import AtomsData, generate_db
# def main(db_path):
#     os.makedirs(os.path.dirname(db_path), exist_ok=True)
#     generate_db(
#         r'C:\Users\Administrator\Desktop\ethanol.xyz', r'C:\Users\Administrator\Desktop\ethanol.db', "Properties=species", ['energy']
#     )
# if __name__ == "__main__":
#     main(r'C:\Users\Administrator\Desktop\ethanol.xyz')

import schnetpack as spk
import os

forcetut = './forcetut'
if not os.path.exists(forcetut):
    os.makedirs(forcetut)

from schnetpack.datasets import MD17

ethanol_data = MD17(os.path.join(forcetut, 'ethanol.db'), molecule='ethanol')
atoms, properties = ethanol_data.get_properties(0)
print('Loaded properties:\n', *['{:s}\n'.format(i) for i in properties.keys()])

from ase.visualize import view
view(atoms, viewer='x3d')
train, val, test = spk.train_test_split(
        data=ethanol_data,
        num_train=1000,
        num_val=500,
        split_file=os.path.join(forcetut, "split.npz"),
    )

train_loader = spk.AtomsLoader(train, batch_size=100, shuffle=True)
val_loader = spk.AtomsLoader(val, batch_size=100)

means, stddevs = train_loader.get_statistics(
    spk.datasets.MD17.energy,
)
n_features = 128

schnet = spk.representation.SchNet(
    n_atom_basis=n_features,
    n_filters=n_features,
    n_gaussians=25,
    n_interactions=3,
    cutoff=5.,
    cutoff_network=spk.nn.cutoff.CosineCutoff
)
energy_model = spk.atomistic.Atomwise(
    n_in=n_features,
    property=MD17.energy,
    mean=means[MD17.energy],
    stddev=stddevs[MD17.energy],
    derivative=MD17.forces,
    negative_dr=True
)
model = spk.AtomisticModel(representation=schnet, output_modules=energy_model)

import torch

# tradeoff
rho_tradeoff = 0.1

# loss function
def loss(batch, result):
    # compute the mean squared error on the energies
    diff_energy = batch[MD17.energy]-result[MD17.energy]
    err_sq_energy = torch.mean(diff_energy ** 2)

    # compute the mean squared error on the forces
    diff_forces = batch[MD17.forces]-result[MD17.forces]
    err_sq_forces = torch.mean(diff_forces ** 2)

    # build the combined loss function
    err_sq = rho_tradeoff*err_sq_energy + (1-rho_tradeoff)*err_sq_forces

    return err_sq

from torch.optim import Adam

# build optimizer
optimizer = Adam(model.parameters(), lr=5e-4)

import schnetpack.train as trn

# set up metrics
metrics = [
    spk.metrics.MeanAbsoluteError(MD17.energy),
    spk.metrics.MeanAbsoluteError(MD17.forces)
]

# construct hooks
hooks = [
    trn.CSVHook(log_path=forcetut, metrics=metrics),
    trn.ReduceLROnPlateauHook(
        optimizer,
        patience=5, factor=0.8, min_lr=1e-6,
        stop_after_min=True
    )
]

trainer = trn.Trainer(
    model_path=forcetut,
    model=model,
    hooks=hooks,
    loss_fn=loss,
    optimizer=optimizer,
    train_loader=train_loader,
    validation_loader=val_loader,
)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# determine number of epochs and train
n_epochs = 5
trainer.train(device=device, n_epochs=n_epochs)