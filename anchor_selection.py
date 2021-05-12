# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 15:06:31 2018
针对于选择锚点的类
@author: admin
"""

import numpy as np
import scipy.sparse as sparse
from incomplete_mixed_tool import *
import multiprocessing
import random

class AsymAnchorSelector(object):
    def get_dis(self,p,q):
        """
        KL散度，用q来近似p的信息量的损失
        scipy会自动归一化概率分布
        在这里p传锚点，q传邻域
        """
        return scipy.stats.entropy(q,p)

    def get_dismat(self,f,n):
        """得到非对称的距离矩阵"""
        dismat = np.zeros(n*n).reshape(n,n)
        for i in range(n):                
            for j in range(i+1,n):
                disij = self.get_dis(f[i],f[j])
                disji = self.get_dis(f[j],f[i])
                #距离矩阵中，在前的值为锚点
                dismat[i][j] = disij 
                dismat[j][i] = disji
        return dismat


    def get_dc(self,dismat,t):
        n = dismat.shape[0]
        M = n*(n-1)
        dis = np.array([])
        for i,line in enumerate(dismat):# 除掉对角线元素
            dis = np.append(dis,line[:i])
            dis = np.append(dis,line[i+1:])
        dis.sort()
        dc = dis[int(M*t)]
        return dc

    def get_density(self,dismat,dc):
        temp_mat = dismat.copy() - dc
        temp_mat[np.where(temp_mat>0)] = 0
        temp_mat[np.where(temp_mat<0)] = 1
        density = np.sum(temp_mat,axis=1) #实际上对角线的值不应该被计算进去，但是这里问题不大
        return density

    def get_q(self,density):
        """生成密度降序排列的下标序"""
        return np.argsort(density)[::-1]

    def get_sigma(self,dismat,q):
        n = dismat.shape[0]
        maxdis = np.max(dismat)
        s = np.zeros(n) #sigma
        sn = np.zeros(n) #sigma的下标
        for i in range(1,n):
            s[q[i]] = maxdis
            for j in range(i):
                avedis = 0.5*(dismat[q[i]][q[j]]+dismat[q[j]][q[i]])
                if avedis<s[q[i]]:
                    s[q[i]] = avedis
                    sn[q[i]] = q[j]
        s[q[0]]= np.max(s)
        return s,sn