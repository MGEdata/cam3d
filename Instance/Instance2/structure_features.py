# -*- coding: utf-8 -*-

# @Time    : 2020/11/27 11:27
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause
#!/usr/bin/python3.7
# -*- coding: utf-8 -*-

# @Time   : 2019/7/29 19:47
# @Author : Administrator
# @Software: PyCharm
# @License: BSD 3-Clause

"""
# Just a copy from xenonpy
"""

from itertools import zip_longest

import pandas as pd
from pymatgen import Structure

from tqdm import tqdm

from pymatgen.ext.matproj import MPRester

from mgetool.exports import Store

def data_fetcher(api_key, mp_ids, elasticity=True):
    """fetch file from pymatgen"""
    print('Will fetch %s inorganic compounds from Materials Project' % len(mp_ids))

    def grouper(iterable, n, fillvalue=None):
        """"
        split requests into fixed number groups
        eg: grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
        Collect data_cluster into fixed-length chunks or blocks"""
        args = [iter(iterable)] * n
        return zip_longest(fillvalue=fillvalue, *args)

    # the following props will be fetched
    mp_props = [
        'band_gap',
        'density',
        "icsd_ids"
        'volume',
        'material_id',
        'pretty_formula',
        'elements',
        "energy",
        'efermi',
        'e_above_hull',
        'formation_energy_per_atom',
        'final_energy_per_atom',
        'unit_cell_formula',
        'spacegroup',
        'nelements'
        "nsites",
        "final_structure",
        "cif",
        "piezo",
        "diel"
    ]
    if elasticity:
        mp_props.append("elasticity")

    entries = []
    mpid_groups = [g for g in grouper(mp_ids, 20)]

    with MPRester(api_key) as mpr:
        for group in tqdm(mpid_groups):
            mpid_list = [ids for ids in filter(None, group)]
            chunk = mpr.query({"material_id": {"$in": mpid_list}}, mp_props)
            entries.extend(chunk)

    df = pd.DataFrame(entries, index=[e['material_id'] for e in entries])
    # df = df.drop('material_id', axis=1)
    df = df.rename(columns={'unit_cell_formula': 'composition'})

    df = df.reindex(columns=sorted(df.columns))
    df = df.T

    return df


if __name__ == "__main__":

    # dff = data_fetcher("Di2IZMunaeR8vr9w", ["mp-12271"])
    # st = Store()
    # st.to_csv(dff, "id_structure",)
    dff = pd.read_csv("id_structure.csv")
    dff.index=dff.iloc[:,0]
    dff = dff.drop(columns='Unnamed: 0',axis=1)
    fs = dff.loc["cif"]
    structures = fs[0]

    structures = Structure.from_str(structures,"cif")



