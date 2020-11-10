import os

from pandas.io import json
from pymatgen.io.vasp import Incar, Kpoints, Potcar, VaspInput, Poscar
from tqdm import tqdm

f = open("vasp.run")
li = f.readlines()
li = "".join(li)
f.close()

a = json.read_json("dielectricsPetousis.json", orient='columns')
data = a["meta"]


a00 = data[0]

INCAR = a00["incar"]
INCAR = Incar.from_string(INCAR)
INCAR.update({"KSPACING": 0.1, "IBRION": -1, "ISIF": 2, "NELM": 200, "LELF": True, "LAECHG": True, 'LREAL': "Auto"})
[INCAR.pop(i) for i in ["NWRITE"]]

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
        KPOINT = Kpoints.automatic_density(POSCAR.structure, 3000)

        file = VaspInput(INCAR, KPOINT, POSCAR, POTCAR, optional_files={"vasp.run": li})
        file.write_input(r"/home/iap13/wcx/dielectricsPe.files/{}".format(k))
        os.remove(r"/home/iap13/wcx/dielectricsPe.files/{}/KPOINTS".format(k))
        lists.append(r"{}".format(k))

    # break

lists = ", ".join(lists)
f = open("name.txt", mode="w")
li = f.write(lists)
f.close()
