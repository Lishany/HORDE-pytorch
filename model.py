import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import math


class Inv_GCN_model(nn.Module):
    def __init__(self,num_nodes, feature_dim, gcn_l1_dim, output_dim,dropout_prob,edge_index,edge_weight, is_cuda = torch.cuda.is_available()):
        super(Inv_GCN_model, self).__init__()
        if is_cuda:
            #self.feature  = torch.eye(num_nodes).cuda()
            #self.edge_index = edge_index.cuda()
            #self.edge_weight = edge_weight.cuda()
            feature = torch.eye(num_nodes)
            self.register_buffer("feature",feature)
            self.register_buffer("edge_index",edge_index)
            self.register_buffer("edge_weight",edge_weight)
        else:
            self.feature  = torch.eye(num_nodes)
            self.edge_index = edge_index
            self.edge_weight = edge_weight
        self.conv1 = GCNConv(feature_dim, gcn_l1_dim,bias=False, cached = True)
        self.conv2 = GCNConv(gcn_l1_dim, output_dim,bias=False,cached = True)
        self.dropout_prob = dropout_prob
        self.num_nodes = num_nodes
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        self.dropout_layer = nn.Dropout(self.dropout_prob)
    def forward(self):
        #feature0 = nn.Dropout(self.dropout_prob)(self.feature)
        feature0 = self.dropout_layer(self.feature)
        #print(self.feature)
        #print(self.edge_index)
        #print(self.edge_weight)
        #print(feature0)
        x = self.conv1(feature0, self.edge_index, self.edge_weight)
        #print(x)
        x = F.relu(x)
        x = F.normalize(x,2,1)
        self.first_embedding = x
        x = self.dropout_layer(x)
        x = self.conv2(x, self.edge_index, self.edge_weight)
        x = torch.tanh(x)
        x = F.normalize(x, 2, 1)
        return x
    def get_firstembedding_and_fina_embed(self):
        #feature0 = self.dropout_layer(self.feature)
        x = self.conv1(self.feature, self.edge_index, self.edge_weight)
        x = F.relu(x)
        first_embed = F.normalize(x,2,1)
        x = self.dropout_layer(first_embed)
        x = self.conv2(x, self.edge_index, self.edge_weight)
        x = torch.tanh(x)
        x = F.normalize(x, 2, 1)
        return(first_embed,x)
    



class Var_GCN_model(nn.Module):
    def __init__(self, input_dim, output_dim,visit_dim, num_nodes, dropout_prob, act, deg_node_normalized,forget_bias, is_cuda = torch.cuda.is_available()):
        super(Var_GCN_model, self).__init__()
        self.num_nodes = num_nodes
        self.dropout_prob = dropout_prob
        self.linear =nn.Linear(input_dim, visit_dim,bias=False)
        self.lstm = nn.LSTM(visit_dim, output_dim,batch_first=True)# , dropout=dropout_prob)
        self.output_dim = output_dim
        self.deg_node_normalized = deg_node_normalized
        self.act = act
        self.bias = nn.Linear(1,self.num_nodes,bias=False)
        #for layer_name in self.lstm._all_weight:
        #   for weight_name in layer_name:
        #        if 'weight' in weight_name:
        #           nn.init.xavier_uniform_(self.lstm.__getattr__(weight_name))
        #elif 'bias' in weight_name:
        #   stdv = math.sqrt(6.0/(output*4))
        #    self.lstm.__getattr__(weight_name).data.uniform_(-stdv,stdv)
        #nn.init.xavier_uniform_(self.bias.weight)
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "weight" in n, names):
                weight = getattr(self.lstm,name)
                nn.init.xavier_uniform_(weight)
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(forget_bias)
        nn.init.xavier_uniform_(self.linear.weight)
        stdv = math.sqrt(6.0/(self.num_nodes))
        self.bias.weight.data.uniform_(-stdv,stdv)
        self.dropout_layer = nn.Dropout(self.dropout_prob)
        if is_cuda:
            #self.onetensor = torch.cuda.FloatTensor([1])
            self.register_buffer("onetensor",torch.cuda.FloatTensor([1]))
        else:
            self.onetensor = torch.FloatTensor([1])
                
    def forward(self, data, feature, ti_embed, edge_normalized, degree_visit):
        size_o = data.size() # B * L * N
        label = data[:,1:,:].contiguous().view(-1,size_o[2])
        feature = self.dropout_layer(feature)
        degree_i_temp = degree_visit[:,1:].contiguous().view(-1)# 下一次visit的相对的degree，可以用来筛选有效的当前visit。 L-1
        mask_flag = degree_i_temp>0
        edge_normalized = edge_normalized.view(-1, self.num_nodes)
        x = torch.matmul(edge_normalized, feature) # (B*L)*E 此处没问题 x为BL* E
        x = self.linear(x) # BL * E2  ## 这里在数据预处理的时候应该就要注意 ，如果visit没有entity就提前去掉
        x = x.view(size_o[0],size_o[1],-1) # x 为B * L * E ## LSTM 的输入
        out, hidden = self.lstm(x) # out: B*L*E  是否需要再进行一个连接层，这样lstm的output_dim就可以不必和 INV_GCN出来的dim一样了
        out = self.act(out)
        out = out[:,:-1,:].contiguous()
        out = out.view(-1,self.output_dim) #B L-1 * E -> S E
        select_out = out[mask_flag] # w(x_v,x_j)>0 S*E
        bias = self.bias(self.onetensor)
        xwb = torch.matmul(select_out,torch.transpose(ti_embed,0,1)) + bias
        #print(bias.size())
        softmax_matrix = torch.softmax(xwb,1) # S*N
        label = label[mask_flag]  # S*N
        mask_flag = label>0
        loss = -torch.sum(torch.log(softmax_matrix[mask_flag]))
        #print(loss)
        return loss
    def get_softmax_matrix(self, data, feature, ti_embed, edge_normalized, degree_visit):
        size_o = data.size() # B * L * N
        edge_normalized = edge_normalized.view(-1, self.num_nodes) #BL*E
        x = torch.matmul(edge_normalized, feature) # (B*L)*E 此处没问题 x为BL* E
        x = self.linear(x) # BL * E2  ## 这里在数据预处理的时候应该就要注意 ，如果visit没有entity就提前去掉
        x = x.view(size_o[0],size_o[1],-1) # x 为B * L * E2 ## LSTM 的输入
        out, hidden = self.lstm(x) # out: B*L*E2  是否需要再进行一个连接层，这样lstm的output_dim就可以不必和 INV_GCN出来的dim一样了
        out = self.act(out) 
        out = out.contiguous()
        out = out.view(-1, self.output_dim) # BL * E2  ti_embed N * E2
        bias = self.bias(self.onetensor) # N
        xwb = torch.matmul(out,torch.transpose(ti_embed,0,1)) + bias  #xw + b  BL * N
        softmax_matrix = torch.softmax(xwb,1).view(size_o[0], size_o[1], -1)
        #print(softmax_matrix)
        return softmax_matrix



class Inv_GCN_RW_Optimizer(nn.Module):
    def __init__(self, negsample_num):
        super(Inv_GCN_RW_Optimizer, self).__init__()
        self.negsample_num = negsample_num
    def forward(self, input, pos_input,neg_input, Embedding): #input: B, pos_input: B, neg_input: B * #(nega)
        neg_input = neg_input.view(-1)
        input_emb = Embedding[input]
        pos_emb = Embedding[pos_input]
        pos = torch.sum(F.logsigmoid(torch.sum(input_emb * pos_emb, 1)))  # B
        input_emb = input_emb.unsqueeze(1) # B*E -> B*1*E
        neg_emb = Embedding[neg_input].view(input.size()[0],self.negsample_num,-1)
        neg = torch.sum(F.logsigmoid(torch.sum(input_emb*neg_emb,2))) # B* neg_num
        loss = neg-pos
        #print(loss)
        return loss



class Whole_Model(nn.Module):
    def __init__(self, node_model, visit_model, node_optimizer):
        super(Whole_Model, self).__init__()
        self.node_model = node_model
        self.visit_model = visit_model
        self.node_optimizer = node_optimizer
    def forward(self, inputbatch, posbatch, negbatch,batch_visit,batch_visit_normalized_edge,batch_visitdegree): #input: B, pos_input: B, neg_input: B * #(nega)
        #out1 = self.node_model.forward()
        out1 = self.node_model()
        #self.embed = out1 #(make no sense in the eval session, we can't get this one with dropout issue in eval, we need to run node_model again to get the one without dropout session)
        #print(self.embed)
        #print(self.node_model.conv1.weight)
        loss1 = self.node_optimizer(inputbatch, posbatch, negbatch, out1)
        loss2 = self.visit_model(batch_visit,self.node_model.first_embedding, out1,batch_visit_normalized_edge,batch_visitdegree)
        return torch.unsqueeze(loss1+loss2,0)