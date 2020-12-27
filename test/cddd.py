# -*- coding: utf-8 -*-

# @Time    : 2020/12/10 15:54
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause
import subprocess

import cdt
from cdt import RPackages
from cdt.utils.R import DefaultRPackages
from cdt.utils.Settings import ConfigSettings, SETTINGS
import networkx as nx
SETTINGS.rpath = r"C:\Program Files\R\R-4.0.3\bin\x64\Rscript"


data, graph = cdt.data.load_dataset('sachs')
print(data.head())
glasso = cdt.independence.graph.Glasso()
skeleton = glasso.predict(data)
print(skeleton)
new_skeleton = cdt.utils.graph.remove_indirect_links(skeleton, alg='aracne')
model = cdt.causality.graph.GES()
output_graph = model.predict(data, new_skeleton)
print(nx.adjacency_matrix(output_graph).todense())
