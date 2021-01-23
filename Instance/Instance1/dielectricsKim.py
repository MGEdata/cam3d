import os
import warnings

from pymatgen.io.cif import CifParser, CifFile
from pymatgen.io.vasp import Incar, Poscar, Potcar, Kpoints, VaspInput
from tqdm import tqdm

warnings.filterwarnings("ignore")

jiaoben = r"run.lsf"
f = open(jiaoben)
li = f.readlines()
li = "".join(li)
f.close()

name_map = {'C': "C", 'F': "F", 'Ge': "Ge_d", 'H': "H",
            'N': "N", "Br": "Br", "I": "I", "Pb": "Pb_d",
            "Sn": "Sn_d", "Cl": "Cl", "O": "O"}

li2 = """
ALGO = Fast
EDIFF = 1e-06
ENCUT = 600
IBRION = -1
ICHARG = 1
ISIF = 2
ISMEAR = -5
ISPIN = 2
ISTART = 1
KSPACING = 0.2
LAECHG = True
LCHARG = True
LELF = True
LORBIT = 11
LREAL = Auto
LWAVE = False
NELM = 200
PREC = Accurate
SYMPREC = 1e-08"""

INCAR = Incar.from_string(li2)
INCAR.update({"KSPACING": 0.2, "IBRION": -1, "ISIF": 2, "NELM": 200, "LELF": True, "LAECHG": True, 'LREAL': "Auto"})

path = r"/home/iap13/wcx/cam3d/Instance/Instance1/dielectricsKim"
path = os.path.join(path, "cif_merge")
files = os.listdir(path)
lists = []
for k, i in tqdm(enumerate(files)):
    f = CifFile.from_file(os.path.join(path, i))
    cif = CifParser.from_string(f.orig_string)
    structure = cif.get_structures()[0]
    name = structure.symbol_set

    try:
        name = [name_map[i] for i in name]
        POTCAR = Potcar(symbols=name)
        POSCAR = Poscar(structure)
    except(OSError):
        print(name)

    else:
        KPOINT = Kpoints.automatic_density(POSCAR.structure, 300)

        file = VaspInput(INCAR, KPOINT, POSCAR, POTCAR, optional_files={"run.lsf": li})
        file.write_input(r"/share/home/skk/wcx/dielectricsKim.files/{}".format(k))
        os.remove(r"/share/home/skk/wcx/dielectricsKim.files/{}/KPOINTS".format(k))
        lists.append(r"{}".format(k))

# ############
# cifwriter = CifWriter(structure)
# cifwriter.write_file("cif")

lists = ", ".join(lists)
f = open("name.txt", mode="w")
li = f.write(lists)
f.close()
