import os

from pandas.io import json
from pymatgen.io.vasp import Incar, Kpoints, Potcar, VaspInput, Poscar
from tqdm import tqdm

f = open("vasp.run")
li = f.readlines()
li = "".join(li)
f.close()


a = json.read_json("dielectricsQu.json", orient='columns')

a00 = a.iloc[0][3]
INCAR = a00["INCAR"][0]
INCAR = Incar.from_string(INCAR)
INCAR.update({"KSPACING": 0.1, "IBRION": -1, "ISIF": 2, "NELM": 200, "LELF": True, "LAECHG": True, 'LREAL': "Auto"})
[INCAR.pop(i) for i in ["KPOINT_BSE", "LEPSILON", "SIGMA", "MAGMOM", 'LASPH', 'LPEAD', 'LVHAR']]

a00_list = a.iloc[:, 0]
lists = []
for k, a00 in tqdm(enumerate(a00_list)):
    formula = a00["formula"][0][0]
    group = a00["space group number"][0]
    POSCAR = a00["POSCAR"][0]
    KPOINTs = a00["kpoints"]
    name = eval(a00['POTCAR'][0])
    name = [i.split(" ")[1] for i in name]

    KPOINTs.insert(2, "Line-Mode")
    KPOINT = Kpoints.from_string("".join(KPOINTs))

    POTCAR = Potcar(symbols=name)
    POSCAR = Poscar.from_string(POSCAR)

    file = VaspInput(INCAR, KPOINT, POSCAR, POTCAR, optional_files={"vasp.run": li})
    file.write_input(r"/home/iap13/wcx/dielectrics.Qu/{}".format(k))
    os.remove(r"/home/iap13/wcx/dielectrics.Qu/{}/KPOINTS".format(k))
    lists.append(r"{}".format(k))

lists = ", ".join(lists)
f = open("name.txt", mode="w")
li = f.write(lists)
f.close()
