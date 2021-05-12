import numpy as np
import scipy
from sklearn.metrics import mean_squared_error,mean_absolute_error
import random
import decomposition as decomp
from math import sqrt,log
import pickle
from anchor_selection import *
from sklearn import preprocessing

class AnchorGetter(object):
	"""docstring for AnchorGetter"""
	def __init__(self,m,n):
		self.m = m
		self.n = n

	def get_anchorlist_asym(self,q,fold,pu,qi,auto = False):
		candiitem_num = candiuser_num = q

		anchorSelector = AsymAnchorSelector()
		"""挑选用户的锚点"""
		ut,it = 0.04,0.04
		user_dis = anchorSelector.get_dismat(pu, self.m)  # 是用因子矩阵计算距离还是用评分矩阵计算距离
		udc = anchorSelector.get_dc(user_dis, ut)
		udensity = anchorSelector.get_density(user_dis, udc)
		uq = anchorSelector.get_q(udensity)
		usigma, usn = anchorSelector.get_sigma(user_dis, uq)
		#对距离和密度分别进行归一化
		min_max_scaler = preprocessing.MinMaxScaler()
		usigma = min_max_scaler.fit_transform(usigma)
		udensity = min_max_scaler.fit_transform(udensity) 
		ugamma = usigma * udensity
		a = [(i,g) for i,g in enumerate(ugamma)]
		aa = sorted(a,key = lambda x:x[1],reverse=True)
		ugamma_index = [i[0] for i in aa[:100]]
		ugamma_val = [i[1] for i in aa[:100]]
		
		if auto:
			j = 0
			while True:
				h = ugamma_val[j] - np.mean(ugamma_val[j+1:j+11])
				if h>= 0.01:
					j+=1
				else:
					candiuser_num = j
					break

		"""挑选物品的锚点"""
		item_dis = anchorSelector.get_dismat(qi, self.n)
		idc = anchorSelector.get_dc(item_dis, it)
		idensity = anchorSelector.get_density(item_dis, idc)
		iq = anchorSelector.get_q(idensity)
		isigma, isn = anchorSelector.get_sigma(item_dis, iq)
		#对距离和密度分别进行归一化
		min_max_scaler = preprocessing.MinMaxScaler()
		isigma = min_max_scaler.fit_transform(isigma)
		idensity = min_max_scaler.fit_transform(idensity)
		igamma = isigma * idensity
		a = [(i,g) for i,g in enumerate(igamma)]
		aa = sorted(a,key = lambda x:x[1],reverse=True)
		igamma_index = [i[0] for i in aa[:100]]
		igamma_val = [i[1] for i in aa[:100]]
		
		if auto:        
			j = 0
			while True:
				h = igamma_val[j]-np.mean(igamma_val[j+1:j+11])
				if h>=0.01:
					j+=1
				else:
					candiitem_num = j
					break 

		anchor_num = max(candiuser_num,candiitem_num)
		candiuser = ugamma_index[:anchor_num]
		candiitem = igamma_index[:anchor_num]
		random.shuffle(candiuser)
		random.shuffle(candiitem)
		anchor_list = []
		for m,n in zip(candiuser,candiitem):
			anchor_list.append((m,n))  
		return (anchor_list,len(anchor_list))   



class SubdataGetter(object):
	def __init__(self,m,n):
		self.m = m
		self.n = n

	def get_subdata_asym(self,args):
		"""使用KL散度——非对称距离"""
		data_train, data_test, pu, qi, q, i, anchor = args
		neighuser = get_neighbortuple(anchor[0], pu, self.m)  # 得到邻域用户及与锚点的相似度
		neighitem = get_neighbortuple(anchor[1], qi, self.n)  # 得到邻域物品及与锚点的相似度
		subdata_train = get_submatrix(data_train, neighuser, neighitem)  # 得到这个范围内的训练集
		subdata_test = get_submatrix(data_test, neighuser, neighitem)
		return (subdata_train, subdata_test, i)


def get_datadic(data):
	true_dict={}
	for u,v,r in data:
		true_dict.setdefault(u, {})
		true_dict[u][v] = r
	return true_dict

def get_datacount(data):
	"""构建一个值为0的字典，用来统计每个评分点出现在子矩阵中的个数"""
	datacount={}
	for u,v,r in data:
		datacount.setdefault(u, {})
		datacount[u][v] = 0
	return datacount

def get_KL(p,q):
	"""
	scipy会自动归一化概率分布
	"""
	return scipy.stats.entropy(q,p)

def get_dis(p,q):
	cosval = np.dot(q.ravel(), p.ravel()) / (
			np.linalg.norm(q.ravel()) * np.linalg.norm(p.ravel()) + 0.001)
	return np.arccos(cosval)

def fill_pred_dict(dict_data,pred,test,len_q,q):
   """
   将预测的值填入预测字典
   """
   for i in range(len(test)):
	   dict_data.setdefault(test[i][0],{})
	   dict_data[test[i][0]].setdefault(test[i][1],np.zeros(len_q))
	   dict_data[test[i][0]][test[i][1]][q]=pred[i]

def get_neighbortuple(anchorid,mat,num):
	"""
	得到用户的邻域及邻域用户与锚点的相似度（或者改成距离）
	return: [(id,相似度),(),()]
	"""	
	dis = []
	for i in range(num):
		dis.append(get_KL(mat[anchorid],mat[i]))
   
	h = 8
	neighbor_sim = []
	for index,val in enumerate(dis):
		if val<h:
			neighbor_sim.append((index,val))
	return neighbor_sim

def get_submatrix(data,neighuser,neighitem):
	"""
	针对get_neighbortuple函数得到子矩阵
	"""
	neighuser_id = [i[0] for i in neighuser]
	neighitem_id = [i[0] for i in neighitem]
	subdata = []
	for i in data:
		if (i[0] in neighuser_id) and (i[1] in neighitem_id):
			subdata.append(i)
	return subdata

