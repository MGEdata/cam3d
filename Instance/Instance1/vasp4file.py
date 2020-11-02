from pymatgen import MPRester
from pymatgen.io.cif import CifParser
from pymatgen.io.vasp.inputs import Incar, Poscar, Kpoints, VaspInput, Potcar
#
# API_KEY = 'Di2IZMunaeR8vr9w'  ##Materialsproject自己号带的
# name = "mp-3626"
#
#
# def que_p(target):  # 分子式，比如Fe2O3
#     a = MPRester(API_KEY)
#     data = a.get_data(target, data_type='vasp', prop="cif")
#     return data
#
#
# data = que_p("mp-3626")
# data = data[0]["cif"]
#
# cif = CifParser.from_string(data)
# structure = cif.get_structures(data)[0]
# ############
# # cifwriter = CifWriter(structure)
# # cifwriter.write_file("cif")
# ############
#
# pos = Poscar(structure)
#
# kpo = Kpoints.automatic_density_by_vol(structure, 300, force_gamma=False)
#
# incat_str = """
# ALGO = Fast
# EDIFF = 0.002
# ENCUT = 520
# IBRION = 2
# ISIF = 3
# ISMEAR = -5
# ISPIN = 2
# LASPH = True
# LORBIT = 11
# LREAL = Auto
# LWAVE = False
# MAGMOM = 40*0.6
# NELM = 100
# NSW = 99
# PREC = Accurate
# SIGMA = 0.05
# """
#
# incar = Incar.from_string(incat_str)
# symbols = pos.site_symbols
# pot = Potcar(symbols=["Sr_sv", "Zr_sv", "O"])
#
# file4 = VaspInput(incar, kpo, pos, pot, optional_files={"cif": cif})
# file4.write_input(r"%s" % name)
#

import os

from pandas.io import json
from pymatgen.io.vasp import Incar, Kpoints, Potcar, VaspInput, Poscar, Vasprun
from pymatgen.io.vasp.sets import MPRelaxSet
from tqdm import tqdm

# file=VaspInput.from_directory('/home/iap13/wcx/cam3d/Instance/Instance1/cx_files/')
#
# vasp_cmd = ["mpirun", "-np", "16", "vasp_std"]
# file.run_vasp(run_dir="cx_files", vasp_cmd=vasp_cmd)


from pymatgen import MPRester
from pymatgen.io.vasp.sets import MPRelaxSet
#
#
VASP_CMD = ["mpirun", "-np", "16", "vasp_std"]
#
#
def main():
    # mpr = MPRester("Di2IZMunaeR8vr9w")
    mpr = MPRester()
    structure = mpr.get_structures("Li2O")[0]
    for k_dens in [100, 200, 400, 800]:
        vis = MPRelaxSet(structure,
            user_kpoints_settings={"reciprocal_density": k_dens})
        vi = vis.get_vasp_input()
        kpoints = vi["KPOINTS"].kpts[0][0]
        d = "Li2O_kpoints_%d" % kpoints

        # Directly run vasp.
        vi.run_vasp(d, vasp_cmd=VASP_CMD)
        # Use the final structure as the new initial structure to speed up calculations.
        structure = Vasprun("%s/vasprun.xml" % d).final_structure


if __name__ == "__main__":
    main()
