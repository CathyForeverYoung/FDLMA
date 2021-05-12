# -*- coding: utf-8 -*-
import numpy as np
import scipy.sparse as sparse
import multiprocessing
from sklearn import preprocessing
import os
import sys

from Inputer import *
from incomplete_mixed_tool import *
from parameter_prepare import *


if __name__ == "__main__":
	cores = multiprocessing.cpu_count()

	inputer = ML100K()
	#inputer = ML1M()

	for fold in range(2,3):
		print("------the dataset running is {}; the fold is {}-------".format(inputer.dataset,fold))

		data_train,data_test = inputer.data_load_thisfold(fold)
		index_u_train, index_i_train, rate_train = inputer.data2list(data_train)
		index_u_test, index_i_test, rate_test = inputer.data2list(data_test)
		ui = np.array(sparse.coo_matrix((rate_train, (index_u_train, index_i_train)), shape=(inputer.m, inputer.n), dtype=float).todense())
		iu=ui.T

		"""第一次计算隐性因子向量"""
		nmf_pu,nmf_qi = get_NMF(ui,fold)
		nmf_pu,nmf_qi = np.array(nmf_pu)+0.0001, np.array(nmf_qi).T + 0.0001
	
		#一些要用的字典值
		true_dict_train = get_datadic(data_train)
		true_dict_test = get_datadic(data_test)
		datacount_test = get_datacount(data_test) #统计每个数据属于的子矩阵个数

		q=50
		anchorgetter = AnchorGetter(inputer.m, inputer.n)
		anchor_list, q = anchorgetter.get_anchorlist_asym(q, fold, nmf_pu, nmf_qi,auto=False)
		print(q)


		"""非对称相似度的邻域"""
		print("---------start constructing local matrices-----------")
		anchor_subdata_train = {}
		anchor_subdata_test = {}
		subdata_getter = SubdataGetter(inputer.m, inputer.n)
		multiprocessing.freeze_support()
		pool = multiprocessing.Pool(processes=cores - 2)
		nargs = [(data_train, data_test, nmf_pu, nmf_qi, q, i, anchor_list[i]) for i in range(q)]
		for y in pool.imap(subdata_getter.get_subdata_asym, nargs):
			subdata_train, subdata_test, i = y
			anchor_subdata_train[i] = subdata_train
			anchor_subdata_test[i] = subdata_test
			for u,v,r in subdata_test:
				datacount_test[u][v] += 1
			sys.stdout.write('have constructed %d/%d local matrices\r' % (i+1,q))
		pool.close()
		pool.join()

	
	
	
 