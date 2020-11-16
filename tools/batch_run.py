# -*- coding: utf-8 -*-
import os

from tqdm import tqdm

lsf_vasp_240 = """
#BSUB -q normal
#BSUB -n 32
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -R "span[ptile=24]"
PATH=/share/app/vasp.5.4.1-2018/bin/:$PATH
source /share/intel/intel/bin/compilervars.sh intel64
mpirun -np 36 vasp_std > log
"""

pbs_vasp_238_239 = """
#SBATCH -N 1
#SBATCH -n 36
#SBATCH --ntasks-per-node=36
#SBATCH --output=%j.out
#SBATCH --error=%j.err
source /data/home/qian1/intel/bin/compilervars.sh intel64
source /data/home/qian1/intel/mkl/bin/mklvars.sh intel64
export PATH=$PATH:/data/home/qian1/app/vasp/vasp.5.4.4/bin
scontrol show hostname $SLURM_JOB_NODELIST > host
mpirun -np 36 vasp_std >log
"""

pbs_python_template = """
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --ntasks-per-node=36
#SBATCH --output=%j.out
#SBATCH --error=%j.err
python test.py run > log
"""

lsf_python_template = """
#BSUB -q normal
#BSUB -n 16
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -R "span[ptile=24]"
python test.py run > log
"""

static = """
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

temps = {"lsf_python": lsf_python_template, "pbs_python": pbs_python_template,
         "pbs_vasp": pbs_vasp_238_239, "lsf_vasp": lsf_vasp_240}


def upload(run_tem=None, pwd=None, filter_in=None,
           filter_out=None, job_type=None, existed_run_tem=None, **kwargs):
    """
    产生批处理提交文件。
    最方便用法: 把该文件放到算例同一文件夹下，并提供 pbs or lsf 模板给 run_tem 参数.
    Args:
        run_tem: str, file,None
            pbs,lsf 模板。可以是文件，也可以是复制过来的字符串,该文件会复制到每个子文件夹.

        existed_run_tem: str,None
            1.不需要批量复制时, 使用每个子文件下已经存在 pbs,lsf 模板 该模板的名字必须提供 （run_tem=None, existed_run ！= None）。
            2.使用批量复制时, existed_run 为每个文件夹下新 pbs,lsf 模板名字 （run_tem！=None, existed_run != None）
            3.使用批量复制时, 每个文件夹下新 pbs,lsf 模板名字默认使用源文件名字, 可能会覆盖！ （run_tem!=None, existed_run == None）
            3.不使用批量复制, 也不提供 pbs,lsf 模板名字,你用个p（run_tem==None, existed_run == None）

        pwd: path,None
            数据路径

        filter_in: list,str,None
            文件夹过滤条件，默认全选子文件夹
            如果为字符串，数据路径的子文件夹中包含该字符串被选择
            如果为列表，数据路径的子文件夹名在列表中的被选择

        filter_out: list,str,None
            如果为字符串，数据路径的子文件夹中包含该字符串被忽略
            如果为列表，数据路径的子文件夹名在该列表中的被忽略

        job_type: str,None
            "pbs" or "lsf"，作业提交系统, 默认根据模板判断

        **kwargs:
            INCAR:str
                若提供，在每个子文件夹下加入该 INCAR 文件

    Returns:
        ./bash.sh 文件
    """
    if "INCAR" in kwargs:
        INCAR = kwargs["INCAR"]
    else:
        INCAR = None

    if INCAR is not None:
        if os.path.isfile(INCAR):
            try:
                a = open(INCAR)
                con = a.readlines()
                INCAR = "".join(con)
            except IOError:

                raise IOError("We cannot import message from {} file".format(INCAR))

    if run_tem is None:
        assert existed_run_tem is not None, \
            "Use -r, -e or both of them !!!.\n" \
            "若把模板文件复制到所有子文件夹，如：\n" \
            "python ***.py -r \***\pbs.run\n" \
            "若子文件夹已经存在模板，提供使用它们的文件名及类型，如：\n" \
            "python ***.py -e lsf.run -t lsf \n"

    if run_tem is not None:
        if os.path.isfile(run_tem):
            try:
                a = open(run_tem)
                con = a.readlines()
                run_tem = "".join(con)
            except IOError:

                raise IOError("We cannot import message from {} file".format(run_tem))
    elif run_tem in temps:
        run_tem = temps[run_tem]
    if run_tem is not None:
        if job_type is None:
            if "SBATCH" in run_tem:
                job_type = "pbs"
            else:
                job_type = "lsf"
    else:
        if job_type is None:
            raise NotImplemented("The job_type should be defined.>>> 'pbs' or 'lsf'")
    if pwd is None:
        pwd = os.getcwd()
    os.chdir(pwd)
    files = os.listdir(pwd)  # 读入文件夹
    if isinstance(filter_in, str):
        files = [i for i in files if filter_in in i]
    if isinstance(filter_out, str):
        files = [i for i in files if filter_out not in i]

    if isinstance(filter_in, (list, tuple)):
        files = filter_in
    if isinstance(filter_out, (list, tuple)):
        [files.remove(i) for i in filter_out]

    if len(files) == 0:
        raise FileNotFoundError("There is no directory left after filtering")
    for filei in tqdm(files):
        if run_tem is not None:
            try:
                existed_run_tem = existed_run_tem if existed_run_tem else "run.lsfpbs"
                os.chdir(os.path.join(pwd, str(filei)))
                pwdpath = os.getcwd()
                f = open(existed_run_tem, "w")
                f.write(run_tem)
                f.close()
                if INCAR is not None:
                    f = open("INCAR", "w")
                    f.write(INCAR)
                    f.close()
            except NotADirectoryError:
                print(filei, "is a file and be filtered")
                files.remove(filei)
        else:
            try:
                existed_run_tem = existed_run_tem if existed_run_tem else "run.lsfpbs"
                os.chdir(os.path.join(pwd, str(filei)))
                # pwdpath = os.getcwd()
                # assert os.path.isfile(existed_run_tem), "老夫一瞅，你的{}下没有你说的这个{}文件".format(os.getcwd(),existed_run_tem)
                assert os.path.isfile(os.path.join(os.getcwd(), existed_run_tem)), \
                    "There is no {} in your {}".format(existed_run_tem, os.getcwd())
                if INCAR is not None:
                    f = open("INCAR", "w")
                    f.write(INCAR)
                    f.close()
            except NotADirectoryError:
                print(filei, "is a file and filtered")
                files.remove(filei)

    if job_type == "lsf":
        job_type = "bsub < {}".format(existed_run_tem)
    else:
        job_type = "qsub {}".format(existed_run_tem)

    os.chdir(pwd)

    files = ["./" + i for i in files]
    batch_str = """#!/bin/bash
echo $dirname
for i in {}

do
cd $i
{}
cd ..
done
    """.format(files, job_type, )
    batch_str = batch_str.replace("'", "")
    batch_str = batch_str.replace("[", "")
    batch_str = batch_str.replace("]", "")
    batch_str = batch_str.replace(",", "")
    bach = open("batch.sh", "w")
    bach.write(batch_str)
    print("###################################################")
    print("OK")


################################################################################################################
if __name__ == '__main__':
    # 命令行模式
    import argparse

    parser = argparse.ArgumentParser(description='产生任务批处理文件,请保证 -r,-e 至少存在一个.\n'
                                                 '最方便用法: \n'
                                                 '1.把该文件放到和算例同一文件夹下，并提供 pbs or lsf 模板给 -r 参数.\n'
                                                 "python ***.py -r \***\pbs.run\n"
                                                 "2.若算例中已经存在pbs or lsf 模板，请提供模板名字及种类给 -e, -t 参数.\n"
                                                 "python ***.py -e pbs.run -t lsf \n")
    parser.add_argument('-r', dest='run_tem', default=None,
                        help='{pbs,lsf}模板文件地址')
    parser.add_argument('-p', dest='pwd', default=None,
                        help='所有算例地址, 默认当前地址')
    parser.add_argument('-e', dest='existed_run_tem', default=None,
                        help='当子文件夹已经存在{pbs,lsf}模板文件，直接使用它。提供该统一模板文件的名字')
    parser.add_argument('-t', dest='job_type', default=None,
                        help='作业提交系统, 默认根据模板判断，当没有-r时，需要被提供')
    parser.add_argument('-i', dest='filter_in', default=None,
                        help='过滤条件，当文件夹包含该字符串被选中')
    parser.add_argument('-o', dest='filter_in', default=None,
                        help='过滤条件，当文件夹包含该字符串被忽略')
    args = parser.parse_args()

    upload(run_tem=args.run_tem, pwd=args.pwd, existed_run_tem=args.existed_run_tem, job_type=args.job_type)
##############################################################################################################
# print(upload.__doc__)
# upload(run_tem="/share/home/skk/wcx/cam3d/Instance/Instance1/others/run.lsf", pwd="/share/home/skk/wcx/test/")
