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




def get_ids(api_key="Di2IZMunaeR8vr9w"):
    """
    support_proprerity = ['energy', 'energy_per_atom', 'volume', 'formation_energy_per_atom', 'nsites',
    'unit_cell_formula','pretty_formula', 'is_hubbard', 'elements', 'nelements', 'e_above_hull', 'hubbards',
    'is_compatible', 'spacegroup', 'task_ids',  'band_gap', 'density', 'icsd_id', 'icsd_ids', 'cif',
    'total_magnetization','material_id', 'oxide_type', 'tags', 'elasticity']
    """
    """
    $gt	>,  $gte >=,  $lt <,  $lte <=,  $ne !=,  $in,  $nin (not in),  $or,  $and,  $not,  $nor ,  $all	
    """
    m = MPRester(api_key)
    ids = m.query(
        criteria=
        # "**O3",
        {
        # 'pretty_formula': {"$in": name_list},
        # 'nelements': {"$lt": 3, "$gte": 3},
        # 'spacegroup.number': {"$in": [225]},
        # 'nsites': {"$lt": 20},
        # 'formation_energy_per_atom': {"$lt": 0},
        # "elements": {"$all": "O"},
        "piezo":{"$ne": None}
        # "elements": {"$all": "O"},
        # "elements": {"$in": list(combinations(["Al", "Co", "Cr", "Cu", "Fe", 'Ni'], 5))}
    },
        properties=["material_id"])
    print("number %s" % len(ids))
    return ids


if __name__ == "__main__":

    idss = get_ids(api_key="Di2IZMunaeR8vr9w")
    idss1 = [i['material_id'] for i in idss]
    dff = data_fetcher("Di2IZMunaeR8vr9w", idss1)
    st = Store(r"E:\pyma")
    st.to_csv(dff, "id_structure")