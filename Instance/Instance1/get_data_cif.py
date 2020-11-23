# -*- coding: utf-8 -*-

# @Time    : 2020/11/9 20:48
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause
from ruamel_yaml import safe_load

functional_dir = {"PBE": "POT_GGA_PAW_PBE",
                  "PBE_52": "POT_GGA_PAW_PBE_52",
                  "PBE_54": "POT_GGA_PAW_PBE_54",
                  "LDA": "POT_LDA_PAW",
                  "LDA_52": "POT_LDA_PAW_52",
                  "LDA_54": "POT_LDA_PAW_54",
                  "PW91": "POT_GGA_PAW_PW91",
                  "LDA_US": "POT_LDA_US",
                  "PW91_US": "POT_GGA_US_PW91"}


from pymatgen.io.vasp import Incar, Kpoints, Potcar, VaspInput, Poscar,Potcar
POTCAR = Potcar(symbols=["Si", "C"],functional="PBE")
POTCAR.write_file("/public/home/MachineLearning/wcx/sic/")
# POTCAR = Potcar(symbols=["Ba_sv","Hf_pv","O"],functional="PW91")



ALGO="Normal"
ADDGRID=True
ISPIN=2
ICHARG=0
NELM=100
IBRION=-1
EDIFF=0.0001
NSW=0
ISIF=3
ENCUT=520

LREAL="Auto"
ISMEAR=-5
SIGMA=0.05
LWAVE=False
LCHARG=True
LVHAR=True
LORBIT=11
LELF=True
LASPH=True
LAECHG=True