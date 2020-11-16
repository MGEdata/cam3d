import os

from pandas.io import json
from pymatgen.io.vasp import Incar, Kpoints, Potcar, VaspInput, Poscar
from tqdm import tqdm


f = open("../../tools/run.lsf")
li = f.readlines()
li = "".join(li)
f.close()

a = json.read_json("dielectricsPetousis.json", orient='columns')
data = a["meta"]


a00 = data[0]

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

a00_list = data
lists = []

for k, a00 in tqdm(enumerate(a00_list)):
    formula = a00["formula"]
    # group = a00["space_group"]
    POSCAR = a00["poscar"]
    name = a00['potcar']
    name = name.split(",")

    try:
        POTCAR = Potcar(symbols=name)

        POSCAR = Poscar.from_string(POSCAR)
    except(OSError):
        pass

    else:
        KPOINT = Kpoints.automatic_density(POSCAR.structure, 300)

        file = VaspInput(INCAR, KPOINT, POSCAR, POTCAR, optional_files={"run.lsf": li})
        file.write_input(r"/share/home/skk/wcx/dielectricsPe.files/{}".format(k))
        os.remove(r"/share/home/skk/wcx/dielectricsPe.files/{}/KPOINTS".format(k))
        lists.append(r"{}".format(k))

    # break

lists = ", ".join(lists)
f = open("name.txt", mode="w")
f.write(lists)
f.close()
