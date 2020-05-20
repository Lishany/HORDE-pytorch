import torch
class ctxpair_loader(object):
    def __init__(self,alldata,batchsize,neg_num, degree_node, distortion):
        self.BATCHSIZE = batchsize
        self.neg_num = neg_num
        self.alldata = alldata
        self.count_alldata = len(self.alldata)
        self.n_batches = int(self.count_alldata / self.BATCHSIZE)
        self.degree_node = degree_node
        self.prob = torch.pow(degree_node,distortion)
        self.inputsample = self.alldata[:, 0]
        self.posisample = self.alldata[:, 1]
        self.itr = 0
        self.shuffle_indices()
    def shuffle_indices(self):
        self.itr = 0
        self.indices = torch.randperm(self.count_alldata)  
        self.negasample = torch.multinomial(self.prob,self.count_alldata*self.neg_num,replacement = True).view(-1,self.neg_num)
        id_temp = (self.posisample.unsqueeze(1) == self.negasample)
        neg_same_temp = self.negasample[id_temp]
        if len(neg_same_temp) > 0:
            temp = torch.multinomial(self.prob, len(neg_same_temp), replacement=True)
            pos_same = self.negasample[id_temp]
            neg_same = temp
            id_temp3 = (temp == pos_same)
            same_node_num = torch.sum(id_temp3).numpy()
            while same_node_num > 0:
                temp = torch.multinomial(self.prob, (int)(same_node_num), replacement=True)
                neg_same[id_temp3] = temp
                id_temp3 = (neg_same == pos_same)
                same_node_num = torch.sum(id_temp3).numpy()
            self.negasample[id_temp] = neg_same
    def is_end(self):
        finished = True if self.itr == self.n_batches else False
        return finished
    def _create_batch(self):
        if self.is_end():
            self.shuffle_indices()
        temp_id = self.indices[self.itr * self.BATCHSIZE:(self.itr + 1) * self.BATCHSIZE]
        inputbatch = self.inputsample[temp_id]
        posbatch = self.posisample[temp_id]
        negbatch = self.negasample[temp_id]
        self.itr += 1
        return inputbatch, posbatch, negbatch





class visit_loader(object):
    def __init__(self,allvisit_dict, batchsize, node_num, degree_node):
        self.BATCHSIZE = batchsize
        self.NODE_NUM = node_num
        self.allvisit = []
        self.visitslen = []
        self.degree_visit = []
        for id in allvisit_dict:
            self.allvisit.append(allvisit_dict[id])
            self.visitslen.append(len(allvisit_dict[id]))
            self.degree_visit.append([len(w) for w in allvisit_dict[id]])
        self.degree_node = degree_node #N
        self.visitslen = torch.LongTensor(self.visitslen)
        self.count_allvisit = len(self.allvisit)
        self.n_batches = int(self.count_allvisit / self.BATCHSIZE)
        self.itr = 0
        self.shuffle_indices()
    def is_end(self):
        finished = True if self.itr == self.n_batches else False
        return finished
    def shuffle_indices(self):
        self.itr = 0
        self.indices = torch.randperm(self.count_allvisit) 
    def _create_batch(self):
        if self.is_end():
            self.shuffle_indices()
        temp_id = self.indices[self.itr * self.BATCHSIZE:(self.itr + 1) * self.BATCHSIZE]
        visitbatch = [self.allvisit[w] for w in temp_id]
        degree_visitbatch = [self.degree_visit[w] for w in temp_id]
        visitslen_batch = self.visitslen[temp_id]
        maxlen = torch.max(visitslen_batch)
        batch_input = torch.zeros(self.BATCHSIZE,maxlen,self.NODE_NUM)
        batch_visitdegree = torch.zeros(self.BATCHSIZE, maxlen)
        batch_edge = torch.zeros(self.BATCHSIZE, maxlen, self.NODE_NUM)
        for i in range(self.BATCHSIZE):
            temp = visitbatch[i]
            for j in range(visitslen_batch[i]):
                batch_input[i][j][temp[j]] = 1
                batch_visitdegree[i][j] = degree_visitbatch[i][j]
                batch_edge[i][j] = batch_input[i][j]/torch.sqrt(degree_visitbatch[i][j]*self.degree_node)
        self.itr += 1
        return batch_input,batch_visitdegree,batch_edge





class test_visit_loader(object):
    def __init__(self,allvisit_dict, batchsize, node_num, degree_node):
        self.BATCHSIZE = batchsize
        self.NODE_NUM = node_num
        self.allvisit = []
        self.visitslen = []
        self.degree_visit = []
        for id in allvisit_dict:
            self.allvisit.append(allvisit_dict[id])
            self.visitslen.append(len(allvisit_dict[id]))
            self.degree_visit.append([len(w) for w in allvisit_dict[id]])
        self.degree_node = degree_node #N
        self.visitslen = torch.LongTensor(self.visitslen)
        self.count_allvisit = len(self.allvisit)
        self.n_batches = int(self.count_allvisit / self.BATCHSIZE)
        self.indices = torch.arange(self.count_allvisit)  # 句子的数量
    def _create_batch(self, k):
        if k==self.n_batches:
            end_id = self.count_allvisit
        else:
            end_id = (k + 1) * self.BATCHSIZE
        temp_id = self.indices[k * self.BATCHSIZE:end_id]
        temp_all_count = end_id - (k * self.BATCHSIZE)
        visitbatch = [self.allvisit[w] for w in temp_id]
        degree_visitbatch = [self.degree_visit[w] for w in temp_id]
        visitslen_batch = self.visitslen[temp_id]
        maxlen = torch.max(visitslen_batch)
        batch_input = torch.zeros(temp_all_count,maxlen,self.NODE_NUM)
        batch_visitdegree = torch.zeros(temp_all_count, maxlen)
        batch_edge = torch.zeros(temp_all_count, maxlen, self.NODE_NUM)
        for i in range(temp_all_count):
            temp = visitbatch[i]
            for j in range(visitslen_batch[i]):
                batch_input[i][j][temp[j]] = 1
                batch_visitdegree[i][j] = degree_visitbatch[i][j]
                batch_edge[i][j] = batch_input[i][j]/torch.sqrt(degree_visitbatch[i][j]*self.degree_node)
        visitslen_batch_list = [x.item() for x in visitslen_batch]
        return batch_input,batch_visitdegree,batch_edge,visitslen_batch_list
    def __iter__(self):
        for i in range(self.n_batches+1):
            if i == self.n_batches:
                if self.count_allvisit % self.BATCHSIZE !=0:
                    yield self._create_batch(i)
                raise StopIteration()
            yield self._create_batch(i)  # return tensor[] and corresponding length

