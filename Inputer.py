class Inputer(object):
    def data2list(self,data):
        index_u = []
        index_i = []
        rate = []
        for i, j, k in data:
            index_u.append(i)
            index_i.append(j)
            rate.append(k)
        return (index_u, index_i, rate)


class ML1M(Inputer):
    def __init__(self):
        self.m = 6040
        self.n = 3952
        self.dataset = 'ML1M'

    def data_load_thisfold(self,i):
        with open("ml-1m/data/train"+str(i)+".txt",'r') as f:
            data_train = eval(f.read())
        with open("ml-1m/data/test"+str(i)+".txt",'r') as f:
            data_test = eval(f.read())
        return (data_train,data_test)


class ML100K(Inputer):
    def __init__(self):
        self.m = 943
        self.n = 1682
        self.dataset = 'ML100K'

    def data_load_thisfold(self,i):
        data_train = []
        file_path = "ml-100k/u"+str(i)+".base"
        with open(file_path) as f:
            for line in f:
                a = line.split("\t")
                data_train.append((int(a[0]) - 1, int(a[1]) - 1, int(a[2])))  # 此处已经把id变成索引（-1）
        
        data_test = []
        file_path = "ml-100k/u"+str(i)+".test"
        with open(file_path) as f:
            for line in f:
                a = line.split("\t")
                data_test.append((int(a[0]) - 1, int(a[1]) - 1, int(a[2])))  # 此处已经把id变成索引（-1）

        return (data_train,data_test)