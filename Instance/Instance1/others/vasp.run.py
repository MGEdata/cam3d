import subprocess

from monty.io import zopen
from monty.json import MSONable
from monty.json import MontyDecoder
from monty.os import cd
from monty.os.path import zpath
from pymatgen import SETTINGS
from pymatgen.util.typing import PathLike


class VaspInput(dict, MSONable):
    """
    Class to contain a set of vasp input objects corresponding to a run.
    """

    def __init__(self, incar, kpoints, poscar, potcar, optional_files=None, **kwargs):
        """
        Args:
            incar: Incar object.
            kpoints: Kpoints object.
            poscar: Poscar object.
            potcar: Potcar object.
            optional_files: Other input files supplied as a dict of {
                filename: object}. The object should follow standard pymatgen
                conventions in implementing a as_dict() and from_dict method.
        """
        super().__init__(**kwargs)
        self.update(
            {"INCAR": incar, "KPOINTS": kpoints, "POSCAR": poscar, "POTCAR": potcar}
        )
        if optional_files is not None:
            self.update(optional_files)

    def __str__(self):
        output = []
        for k, v in self.items():
            output.append(k)
            output.append(str(v))
            output.append("")
        return "\n".join(output)

    def as_dict(self):
        """
        :return: MSONable dict.
        """
        d = {k: v.as_dict() for k, v in self.items()}
        d["@module"] = self.__class__.__module__
        d["@class"] = self.__class__.__name__
        return d

    @classmethod
    def from_dict(cls, d):
        """
        :param d: Dict representation.
        :return: VaspInput
        """
        dec = MontyDecoder()
        sub_d = {"optional_files": {}}
        for k, v in d.items():
            if k in ["INCAR", "POSCAR", "POTCAR", "KPOINTS"]:
                sub_d[k.lower()] = dec.process_decoded(v)
            elif k not in ["@module", "@class"]:
                sub_d["optional_files"][k] = dec.process_decoded(v)
        return cls(**sub_d)

    def write_input(self, output_dir=".", make_dir_if_not_present=True):
        """
        Write VASP input to a directory.
        Args:
            output_dir (str): Directory to write to. Defaults to current
                directory (".").
            make_dir_if_not_present (bool): Create the directory if not
                present. Defaults to True.
        """
        if make_dir_if_not_present and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for k, v in self.items():
            if v is not None:
                with zopen(os.path.join(output_dir, k), "wt") as f:
                    f.write(v.__str__())

    @staticmethod
    def from_directory(input_dir, optional_files=None):
        """
        Read in a set of VASP input from a directory. Note that only the
        standard INCAR, POSCAR, POTCAR and KPOINTS files are read unless
        optional_filenames is specified.
        Args:
            input_dir (str): Directory to read VASP input from.
            optional_files (dict): Optional files to read in as well as a
                dict of {filename: Object type}. Object type must have a
                static method from_file.
        """
        sub_d = {}
        for fname, ftype in [
            ("INCAR", Incar),
            ("KPOINTS", Kpoints),
            ("POSCAR", Poscar),
            ("POTCAR", Potcar),
        ]:
            try:
                fullzpath = zpath(os.path.join(input_dir, fname))
                sub_d[fname.lower()] = ftype.from_file(fullzpath)
            except FileNotFoundError:  # handle the case where there is no KPOINTS file
                sub_d[fname.lower()] = None
                pass

        sub_d["optional_files"] = {}
        if optional_files is not None:
            for fname, ftype in optional_files.items():
                sub_d["optional_files"][fname] = ftype.from_file(
                    os.path.join(input_dir, fname)
                )
        return VaspInput(**sub_d)

    def run_vasp(
        self,
        run_dir: PathLike = ".",
        vasp_cmd: list = None,
        output_file: PathLike = "vasp.out",
        err_file: PathLike = "vasp.err",
    ):
        """
        Write input files and run VASP.
        :param run_dir: Where to write input files and do the run.
        :param vasp_cmd: Args to be supplied to run VASP. Otherwise, the
            PMG_VASP_EXE in .pmgrc.yaml is used.
        :param output_file: File to write output.
        :param err_file: File to write err.
        """
        self.write_input(output_dir=run_dir)
        vasp_cmd = vasp_cmd or SETTINGS.get("PMG_VASP_EXE")
        vasp_cmd = [os.path.expanduser(os.path.expandvars(t)) for t in vasp_cmd]
        if not vasp_cmd:
            raise RuntimeError(
                "You need to supply vasp_cmd or set the PMG_VASP_EXE in .pmgrc.yaml to run VASP."
            )
        with cd(run_dir):
            with open(output_file, "w") as f_std, open(
                err_file, "w", buffering=1
            ) as f_err:
                subprocess.check_call(vasp_cmd, stdout=f_std, stderr=f_err)


import os
from pymatgen.io.vasp import Poscar, Incar, Potcar, Kpoints

# from pymatgen.io.vasp.sets import MPRelaxSet

VASP_CMD = ["mpirun", "-machinefile", "/share/home/skk/host", "-np", "16", "vasp_std"]
in_dir= r"/share/home/skk/wcx/dielectricsPe.files/"
out_dir = r"/share/home/skk/wcx/dielectricsPeOut.files/"

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
INCAR.update({"KSPACING": 0.3, "IBRION": -1, "ISIF": 2, "NELM": 200, "LELF": True, "LAECHG": True, 'LREAL': "Auto"})

"INCAR"
def main():
    files = os.listdir(in_dir)  # 读入文件夹
    num_png = len(files)
    for i in range(3):

        vi = VaspInput.from_directory(os.path.join(in_dir, str(files[i])))
        vi["INCAR"]=INCAR
        vi.run_vasp(os.path.join(out_dir, str(files[i])), vasp_cmd=VASP_CMD)

        print(i)
if __name__ == "__main__":
    main()
