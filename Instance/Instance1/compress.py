# -*- coding: utf-8 -*-

# @Time    : 2020/10/31 11:36
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause
from pymatgen.io.vasp import Chgcar

chgcar =Chgcar.from_file(r'/home/iap13/wcx/cx_flies/0/CHGCAR')
total = chgcar.data["total"]
total = total[::2,::2,::2]
import plotly.graph_objects as go
import numpy as np

np.random.seed(1)

N = 70

fig = go.Figure([total])

# fig.update_layout(
#     scene=dict(
#         xaxis=dict(nticks=4, range=[-100, 100], ),
#         yaxis=dict(nticks=4, range=[-50, 100], ),
#         zaxis=dict(nticks=4, range=[-100, 100], ), ),
#     width=700,
#     margin=dict(r=20, l=10, b=10, t=10))

