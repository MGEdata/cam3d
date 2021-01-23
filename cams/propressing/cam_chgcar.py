from collections import Counter
from copy import deepcopy

import scipy

from mgetool.imports import BatchFile
import numpy as np

from cams.propressing.Neighborfeaturizer import Neighborizer
from cams.propressing.data_rotate import rotation_axis_by_angle, rote_index
from cams.propressing.electro import ChgCar, ElePlot

# bf = BatchFile(r"/home/iap13/wcx/CHG")
from cams.propressing.spilt_tri import relative_location, spilt_tri_prism_z

bf = BatchFile(r"C:\Users\user\Desktop\CHG")
bf.filter_dir_name(includ="Mo")
bf.filter_file_name(exclud="png")
files = bf.merge()
filesi = files[0]
elfcar = ChgCar.from_file(filesi)
elf_data = elfcar.elf_data
structure = elfcar.structure
sites = structure.sites

# 找到中心原子，中心原子位置
counts = Counter([i.species_string for i in sites])
counts2 = list(counts.items())[-1]
assert counts2[-1] == 1
name = counts2[0]
center_index = [n for n, i in enumerate(sites) if i.species_string == name]

ner = Neighborizer()
neighbors_lst = ner.featurize(structure, sites=sites)

frac_coords = elfcar.structure.frac_coords

##########################Osite#######################################################################
sites_index = [n for n, i in enumerate(sites) if i.species_string == "O"]  # 关心的位置(相对于初始)
Ofrac_coords = frac_coords[sites_index, :]  # 所有o位置

Oz_frac_coords = frac_coords[sites_index, :][:, -1]  # 所有o位置 y值
Cz_frac_coords = frac_coords[center_index, :][0][-1]  # center位置 y值
diff_OC = Cz_frac_coords-Oz_frac_coords

min_diff_OC_index = np.argmin(diff_OC)
min_diff_OC = diff_OC[min_diff_OC_index]
sam_layer_O_index = np.where(abs(diff_OC - min_diff_OC) <= 1e-2)[0]

sites_index_sam_layer_O = np.array(sites_index)[sam_layer_O_index]

nei3O, nei_array3O = ner.find_neighbor_from_center(c=center_index, n_nei=3, nei_index=np.array(sites_index_sam_layer_O))

##########################Msite#######################################################################

sites = structure.sites
sites_index = [n for n, i in enumerate(sites) if i.species_string not in ["C", "N", "O", name]]  # 所有M(-C)位置
M_frac_coords = frac_coords[sites_index, :]  # 所有M(-C)位置
Mz_frac_coords = frac_coords[sites_index, :][:, -1]  # 所有M位置 y值

##########################M first site##########################
nei3O0 = nei3O[0]
print("#####first neighbors###########")
for centerO_index in nei3O0:
    Oz_frac_coords = frac_coords[centerO_index, :][-1]  # 中心O位置 y值
    diff_MO = Mz_frac_coords - Oz_frac_coords
    min_diff_MO_index = np.argmin(abs(diff_MO))
    min_diff_MO = diff_MO[min_diff_MO_index]
    sam_layer_M_index = np.array(sites_index)[np.where(abs(diff_MO - min_diff_MO) <= 1e-2)[0]]  # 相同的紧邻层

    sam_layer_M_index= np.append(sam_layer_M_index,[n for n, i in enumerate(sites) if i.species_string == name])

    nei3M, nei_array3M = ner.find_neighbor_from_center(c=centerO_index, n_nei=2,
                                                       nei_index=np.array(sam_layer_M_index))

    test_nei3M_list0=[]
    [test_nei3M_list0.extend(list(i)) for i in nei3M]
    nei3M_all = np.array(test_nei3M_list0)

    if nei3M_all.shape[0]!=3:
        nei3M, nei_array3M = ner.find_neighbor_from_center(c=centerO_index, n_nei=1,
                                                           nei_index=np.array(sam_layer_M_index))
        nei3M_all = nei3M[0]

    #test
    test_sites=[sites[_] for _ in nei3M_all]
    check_test = np.all([True for i in test_sites if i.species_string not in ["C", "N", "O"]])
    if check_test:
        print(centerO_index, nei3M_all, nei_array3M)
    else:
        print(centerO_index, nei3M_all, nei_array3M)
        raise TypeError("filesi is not pass")

##########################M first site##########################
nei3O0 = nei3O[1]
print("#####first neighbors###########")
for centerO_index in nei3O0:
    Oz_frac_coords = frac_coords[centerO_index, :][-1]  # 中心O位置 y值
    diff_MO = Mz_frac_coords - Oz_frac_coords
    min_diff_MO_index = np.argmin(abs(diff_MO))
    min_diff_MO = diff_MO[min_diff_MO_index]
    sam_layer_M_index = np.array(sites_index)[np.where(abs(diff_MO - min_diff_MO) <= 1e-2)[0]]  # 相同的紧邻层

    sam_layer_M_index= np.append(sam_layer_M_index,[n for n, i in enumerate(sites) if i.species_string == name])

    nei3M, nei_array3M = ner.find_neighbor_from_center(c=centerO_index, n_nei=1,
                                                       nei_index=np.array(sam_layer_M_index))

    nei3M_all =nei3M[0]

    # test

    test_sites = [sites[_] for _ in nei3M[0]]
    check_test = np.all([True for i in test_sites if i.species_string not in ["C", "N", "O"]])
    if check_test:
        print(centerO_index, nei3M_all, nei_array3M)
    else:
        print(centerO_index, nei3M_all, nei_array3M)
        raise TypeError("filesi is not pass")
#
#
nei3O0 = nei3O[2]
print("#####first neighbors###########")
for centerO_index in nei3O0:
    Oz_frac_coords = frac_coords[centerO_index, :][-1]  # 中心O位置 y值
    diff_MO = Mz_frac_coords - Oz_frac_coords
    min_diff_MO_index = np.argmin(abs(diff_MO))
    min_diff_MO = diff_MO[min_diff_MO_index]
    sam_layer_M_index = np.array(sites_index)[np.where(abs(diff_MO - min_diff_MO) <= 1e-2)[0]]  # 相同的紧邻层

    sam_layer_M_index= np.append(sam_layer_M_index,[n for n, i in enumerate(sites) if i.species_string == name])

    nei3M, nei_array3M = ner.find_neighbor_from_center(c=centerO_index, n_nei=1,
                                                       nei_index=np.array(sam_layer_M_index))

    nei3M_all = nei3M[0]

    # test
    test_sites = [sites[_] for _ in nei3M[0]]
    check_test = np.all([True for i in test_sites if i.species_string not in ["C", "N", "O"]])
    if check_test:
        print(centerO_index, nei3M_all, nei_array3M)
    else:
        print(centerO_index, nei3M_all, nei_array3M)
        raise TypeError("filesi is not pass")


# elf_data2 = rotation_axis_by_angle(elf_data, angles=(90, 90, 120), times=(2, 2, 2))
# point_ = elfcar.structure.frac_coords
# point = point_[(17, 14, 13), :]
# percent = rote_index(point, elf_data, data_init=True, angles=(90, 90, 120), times=(2, 2, 2), return_type="int")
# maxs = np.max(percent.astype(int), axis=0)
# mins = np.min(percent.astype(int), axis=0)
#
# data_target = elf_data2[mins[0]:maxs[0], mins[1]:maxs[1], :]
# relative = relative_location(percent[:, (0, 1)])
# site = relative * np.array(data_target.shape[:2])
# data_target_tri = spilt_tri_prism_z(data_target.shape, site, z_range=(0, data_target.shape[2]),
#                                     index_percent=False)
# data_result = data_target_tri * data_target
# #
# #
# #
# import matplotlib.pyplot as plt
# from matplotlib import animation
#
# fig = plt.figure()
# ax = plt.subplot()
# ims = []
# for i in np.arange(0, 600):
#     aa = data_result[:, :, i]
#     aa = aa.T
#     art = plt.imshow(aa, cmap="Reds",
#                      vmin=0,
#                      ).findobj()
#
#     ax.set_ylabel('y-lab')
#     ax.set_xlabel('x')
#     ims.append(art)
# fff = animation.ArtistAnimation(fig, ims, repeat=True, repeat_delay=2, interval=50,
#                                 )
# fff.save("test1.gif", writer='pillow')

# xy_coords =np.array([[0,0,0,],[1,0,0],[0,1,0],[1,1,0],[0.9,0.9,0]])
# xy_coords =np.array([[0,0,0,],[0.44444,0.55555,0.645]])
# #
# site = route_percent(xy_coords, angles=(90, 90, 120), times=(2, 2, 2))
# site2 = route_index(xy_coords, data=elf_data, angles=(90, 90, 120), times=(2, 2, 2),return_type="int")
#
# import matplotlib.pyplot as plt
# fig=plt.figure()
# ax = plt.subplot()
#
# ax.scatter(site[0], site[1])
# # ax.scatter(xy_coords3[0],xy_coords3[1])
#
# ax.axis("equal")
# ax.set_ylabel('Y_lab', fontdict={'size': 15, 'color': 'red'})
# ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
# # ax.set_xlim((0,2))
# # ax.set_ylim((0,2))
# plt.gca().invert_yaxis()
#
# plt.show()
#
# valuesa = rote_value(elfcar.structure.frac_coords, elf_data, angles=(90, 90, 120), times=(2, 2, 2), method="near",data_type="old")
# # valuesb = rote_value(elfcar.structure.frac_coords, elf_data, angles=(90, 90, 120), times=(2, 2, 2), method="inter")
# import matplotlib.pyplot as plt
# from matplotlib import animation
# fig=plt.figure()
# ax = plt.subplot()
# a = elf_data2[:,:,400]
# plt.imshow(a)
# plt.show()
#
# pl = ElePlot(elf_data2)
# pl.plot_field()
