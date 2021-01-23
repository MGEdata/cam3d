# -*- coding: utf-8 -*-

# @Time   : 2019/5/19 16:25
# @Author : Administrator
# @Project : feature_preparation
# @FileName: neighborfeaturizer.py
# @Software: PyCharm

"""
this is a description
"""
import numpy as np
from abc import ABC
from copy import deepcopy

from featurebox.featurizers.base import BaseFeaturizer


class Neighborizer(BaseFeaturizer, ABC):

    def __init__(self, tol=1e-3, *, n_jobs=-1, on_errors='raise', return_type='any',neg_tol = 5e-2):
        """

        Parameters
        ----------
        tol:
        n_jobs: int
            The number of jobs to run in parallel for both _fit and Fit. Set -1 to use all cpu cores (default).
        on_errors: string
            How to handle exceptions in feature calculations. Can be 'nan', 'keep', 'raise'.
            When 'nan', return a column with ``np.nan``.
            The length of column corresponding to the number of feature labs.
            When 'keep', return a column with exception objects.
            The default is 'raise' which will raise up the exception.
        return_type: str
            Specific the return type.
            Can be ``any``, ``array`` and ``df``.
            ``array`` and ``df`` force return type to ``np.ndarray`` and ``pd.DataFrame`` respectively.
            If ``any``, the return type dependent on the input0 type.
            Default is ``any``
        """

        super().__init__(n_jobs=n_jobs, on_errors=on_errors, return_type=return_type)

        self.tol = tol
        self.neg_tol = neg_tol
        self.__authors__ = ['TsumiNa']

    def featurize(self, structure, sites=None, r_max=7):
        """
        Args:
            structure: Pymatgen Structure object.
            r_max:
            sites: specific center type
        """
        if not structure.is_ordered:
            raise ValueError("Disordered structure support not built yet")

        # Get the distances between all atoms
        # 如果 sites, 包含自身对，否则不包含自身对。
        neighbors = structure.get_neighbor_list(r_max, sites=sites, numerical_tol=self.tol)
        self.sites = sites
        self.neighbors = neighbors
        return neighbors

    def find_neighbor_from_center(self, c, n_nei=1, neighbors=None, nei_index=None):
        """

        Parameters
        ----------
        nei_index: np.ndarray
            specific neighbor type
        neighbors:
            specific center type
        c:center index
        n_nei:
            number of neighbor types.
            The first ,second ...

        Returns
        -------
        (sites,distances)
        """
        if neighbors is None:
            if hasattr(self, 'neighbors'):
                neighbors = self.neighbors
            else:
                raise NotImplemented("Please featurize first")

        dec = int(-np.log10(self.tol))
        centers, neighbors, _, distances = deepcopy(neighbors)
        distances = np.round(distances, decimals=dec)

        index = np.where(centers == c)
        neighbors = neighbors[index]
        distances = distances[index]

        # 指定对象
        if nei_index is not None:
            index2 = np.zeros_like(neighbors)
            for i in nei_index:
                wh = np.where(neighbors == i)
                index2[wh] = 1
            index2 = index2 > 0
            neighbors = neighbors[index2]
            distances = distances[index2]

        dis = list(set(distances))
        dis.sort()
        # 忽略自身对
        if 0 in dis:
            dis.remove(0)  # 忽略自身对

        # tol delete
        cc = np.diff(dis)
        cc = np.append(cc, np.inf)
        d_index = np.where(abs(cc) > self.neg_tol)[0]
        dis = [dis[i] for i in d_index]

        n_nei_index = [np.where(abs(distances - i) < self.neg_tol) for i in dis[:n_nei]]
        r_nei = [neighbors[i] for i in n_nei_index]

        # 删除重复统计。
        r_nei = [np.array(list(set(list(i)))) for i in r_nei]
        # 没有删除重复统计距离，用来检验。
        rd = [distances[i] for i in n_nei_index]

        # 周期性图像偏移，所以不同紧邻结果位置可能重复。
        return r_nei, rd
