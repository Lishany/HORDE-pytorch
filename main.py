import numpy as np
import dataloader as dataloader
import model as model
import torch
import metrics as metrics
import os
import time
start = time.time()

if torch.cuda.is_available():
    is_cuda = True
else:
    is_cuda = False

#is_cuda = False
    
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"
main_device_id = 0
torch.cuda.set_device(main_device_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device_ids = [0]
device_ids = [0,1,2] #remark, we need to put the main_device_id at first, because I don't set output_device = main_device_id in dataparallel

if not is_cuda:
    device = "cpu"
    device_ids = []

print(device)
#output_write.write(device+"\n")
'''#just to test the code
BATCHSIZE_TI = 5
BATCHSIZE_TV = 5
BATCHSIZE_test = 2
NEG_NUM = 2
distortion = 0.75
gcn_l1_dim = 5
ti_embed_dim = 5
hidden_lstm_dim = 5
visit_embed_dim = 5
dropout_prob_TI = 0.3
dropout_prob_TV = 0.3
NUM_ITR = 100
learning_rate = 0.01
lambda_weight_decay = 0.1
n_printiters = 5
recall_at = 5
label_file = ""
input_dir = './data/'
output_dir = "./embed_result/"
'''
# mimic data
BATCHSIZE_TI = 1024 # 512
BATCHSIZE_TV = 64 # 32
BATCHSIZE_test = 400
NEG_NUM = 1
distortion = 0.75
gcn_l1_dim = 256
ti_embed_dim = 256
hidden_lstm_dim = 256
visit_embed_dim = 256
dropout_prob_TI = 0.1 # 0.3
dropout_prob_TV =0.1 # 0.3
NUM_ITR = 50000
learning_rate = 0.0005 #0.001
lambda_weight_decay = 0.001
n_printiters = 2000 #2000
recall_at = 20
label_file = ""
input_dir = './data/'
#output_dir = "./embed_result_addition1/"
#output_log = "./embed_result_addition1/log.txt"
#output_dir = "./embed_result/"
#output_log = "./embed_result/log.txt"
output_dir = "./embed_result_allgpu/"
output_log = "./embed_result_allgpu/log.txt"
output_write = open(output_log,'w')
if is_cuda:
    output_write.write("cuda\n")
else:
    output_write.write("cpu\n")

if os.path.isfile(label_file):
    input_nodelabels = np.load(label_file)
else:
    input_nodelabels = None

    
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

torch.backends.cudnn.benchmark = True
    
input_nodes, input_edges, input_stats = np.load( input_dir+'graph.npy',allow_pickle=True)
input_ctxpairs = np.load(input_dir+'ctxpairs.npy',allow_pickle=True)
input_seqdict = np.load(input_dir+'patients.npy',allow_pickle=True).item()
test_seqdict = np.load(input_dir + 'testpatients.npy',allow_pickle=True).item()

input_num_nodes, input_num_events, input_num_concepts = input_stats

edge_index = input_edges[0]
edge_weight = input_edges[1]
row = edge_index[0]

def scatter_add(edge_weight, row,dim_size=input_num_nodes):
    result = np.zeros(dim_size)
    for (i,weight) in zip(row,edge_weight):
        result[i] += weight
    return result



degree_node =torch.Tensor(scatter_add(edge_weight, row, input_num_nodes) + 1).float()

ctxpari_loader = dataloader.ctxpair_loader(torch.LongTensor(input_ctxpairs),BATCHSIZE_TI,NEG_NUM, degree_node, distortion)
visit_loader = dataloader.visit_loader(input_seqdict,BATCHSIZE_TV,input_num_nodes,degree_node)
test_loader = dataloader.test_visit_loader(test_seqdict,BATCHSIZE_test,input_num_nodes,degree_node)

edge_index = torch.LongTensor(edge_index)
edge_weight = torch.FloatTensor(edge_weight)

node_model = model.Inv_GCN_model(input_num_nodes, input_num_nodes, gcn_l1_dim, ti_embed_dim,dropout_prob_TI,edge_index,edge_weight, is_cuda = is_cuda)
node_optimizer = model.Inv_GCN_RW_Optimizer(NEG_NUM)

visit_model = model.Var_GCN_model(ti_embed_dim, hidden_lstm_dim,visit_embed_dim, input_num_nodes, dropout_prob_TV,  lambda x: x,torch.pow(degree_node.to(device),-0.5),1, is_cuda = is_cuda)
whole_model = model.Whole_Model(node_model,visit_model,node_optimizer).to(device)

if len(device_ids)>1:
    whole_model = torch.nn.DataParallel(whole_model, device_ids=device_ids,output_device=main_device_id)
    
optimizer = torch.optim.Adam(whole_model.parameters(), lr=learning_rate, weight_decay=lambda_weight_decay)
whole_model.train()

if len(device_ids)>1:
    node_model_test = model.Inv_GCN_model(input_num_nodes, input_num_nodes, gcn_l1_dim, ti_embed_dim,dropout_prob_TI,edge_index,edge_weight)
    node_model_test.eval()
    visit_model_test = model.Var_GCN_model(ti_embed_dim, hidden_lstm_dim,visit_embed_dim, input_num_nodes, dropout_prob_TV,  lambda x: x,torch.pow(degree_node,-0.5),1)
    visit_model_test.onetensor.data = torch.FloatTensor([1])
    visit_model_test.eval()

train_cost = 0
##check the training
train_input_visit =[]
train_visitdegree = []
train_visitedge = []
# end

for n_itr in range(NUM_ITR):
    torch.cuda.empty_cache()
    inputbatch, posbatch, negbatch = ctxpari_loader._create_batch()
    batch_visitinput, batch_visitdegree, batch_visitedge = visit_loader._create_batch()
    
    #check train_input_visit
    if n_itr== 0:
        train_input_visit.append(batch_visitinput)
        train_visitdegree.append(batch_visitdegree)
        train_visitedge.append(batch_visitedge)
    if (n_itr % n_printiters) > (n_printiters-20):
        train_input_visit.append(batch_visitinput)
        train_visitdegree.append(batch_visitdegree)
        train_visitedge.append(batch_visitedge)
    # end
    
    optimizer.zero_grad()
    inputbatch, posbatch, negbatch = inputbatch.to(device), posbatch.to(device), negbatch.to(device)
    batch_visitinput, batch_visitdegree, batch_visitedge = batch_visitinput.to(device), batch_visitdegree.to(device), batch_visitedge.to(device)
    loss = whole_model(inputbatch, posbatch, negbatch,batch_visitinput,batch_visitedge,batch_visitdegree)
    #print(loss)
    loss = loss.sum()
    train_cost += loss.data
    loss.backward()
    optimizer.step()
    if n_itr % n_printiters ==0:
        if len(device_ids)>1:
            node_model_test.conv1.weight.data = whole_model.module.state_dict()['node_model.conv1.weight'].data.cpu()
            node_model_test.conv2.weight.data = whole_model.module.state_dict()['node_model.conv2.weight'].data.cpu()
            dict2visit_mode = {}
            for x in visit_model_test.state_dict():
                dict2visit_mode[x] =  whole_model.module.state_dict()['visit_model.'+x].data.cpu()
            visit_model_test.load_state_dict(dict2visit_mode)
            first_embed, ti_embed = node_model_test.get_firstembedding_and_fina_embed()
            
            '''#测试是否一致，结果：一致
            whole_model.eval()
            first_embed1, ti_embed2 = whole_model.module.node_model.get_firstembedding_and_fina_embed()
            print(first_embed)
            print(first_embed1)
            print(ti_embed)
            print(ti_embed2)
            '''
            
            visit_label = []
            softmax_list = []
            maxlen_list = []
            patient_num = []
            visitlen_list = []
            for i,(batch_visitinput, batch_visitdegree, batch_visitedge, visitslen_batch) in enumerate(test_loader,0):
                #print(torch.sum(batch_visitinput,2))
                #print(visitslen_batch)
                softmax_matrix =visit_model_test.get_softmax_matrix(batch_visitinput,first_embed,ti_embed,batch_visitedge,batch_visitdegree)
                visit_label.append(batch_visitinput)
                softmax_list.append(softmax_matrix)
                maxlen_list.append(len(batch_visitdegree[0]))
                patient_num.append(len(batch_visitdegree))
                visitlen_list.append(visitslen_batch)
            if len(visit_label) == 1:
                tv_labels = batch_visitinput
                tv_outputs = softmax_matrix
                maxlen = maxlen_list[0]
                tv_visitlen = visitslen_batch
            else:
                maxlen = max(maxlen_list)
                tv_outputs = torch.zeros(test_loader.count_allvisit,maxlen,len(ti_embed))
                tv_labels = torch.zeros(test_loader.count_allvisit, maxlen, len(ti_embed))
                start_id = 0
                end_id = 0
                tv_visitlen = []
                for index in range(len(visit_label)):
                    end_id = end_id+patient_num[index]#len(patient_num)
                    tv_labels[start_id:end_id,:maxlen_list[index],:] =  visit_label[index]
                    tv_outputs[start_id:end_id, :maxlen_list[index], :] = softmax_list[index]
                    start_id = end_id
                    tv_visitlen += visitlen_list[index]
            tv_labels_test = tv_labels.data.numpy()
            tv_outputs_test = tv_outputs.data.numpy()
            ti_embed_test = ti_embed.data.numpy()
            #print(input_nodelabels)
            #recall = metrics.compute_recall_k(tv_labels_test[:, :, :input_num_events], tv_outputs_test[:, :, :input_num_events], tv_visitlen, recall_at)
        else:
            whole_model.eval()
            first_embed, ti_embed = whole_model.node_model.get_firstembedding_and_fina_embed()
            #ti_embed = whole_model.module.embed # need to use the one of node_model,  whole_model.module.embed with dropout session, so we will re run node_model() in eval() and get the embed and first_embedding without dropout session
            #if is_cuda:
            #    ti_embed = whole_model.module.node_model()
            #else:
            #    ti_embed = whole_model.node_model()

            
            #check training 
            train_visit_label = []
            train_softmax_list = []
            train_maxlen_list = []
            train_patient_num = []
            for i in range(len(train_input_visit)):
                softmax_matrix = whole_model.visit_model.get_softmax_matrix(train_input_visit[i].to(device),first_embed,ti_embed,train_visitedge[i].to(device),train_visitdegree[i].to(device))
                #print(train_input_visit[i].to(device))
                #print(train_visitedge[i].to(device))
                #print(train_visitdegree[i].to(device))
                train_visit_label.append(train_input_visit[i])#b l n
                train_softmax_list.append(softmax_matrix)
                train_maxlen_list.append(len(train_visitdegree[i][0])) #l
                train_patient_num.append(len(train_visitdegree[i]))#b
            if len(train_visit_label) == 1:
                train_tv_labels = train_input_visit[0]
                train_tv_outputs = softmax_matrix
                maxlen = train_maxlen_list[0]
            else:
                maxlen = max(train_maxlen_list)
                train_tv_outputs = torch.zeros(sum(train_patient_num),maxlen,len(ti_embed))
                train_tv_labels = torch.zeros(sum(train_patient_num), maxlen, len(ti_embed))
                start_id = 0
                end_id = 0
                for index in range(len(train_visit_label)):
                    end_id = end_id+ train_patient_num[index]#len(patient_num)
                    train_tv_labels[start_id:end_id,:train_maxlen_list[index],:] =  train_visit_label[index]
                    train_tv_outputs[start_id:end_id, :train_maxlen_list[index], :] = train_softmax_list[index]
                    start_id = end_id

            tv_labels_test = train_tv_labels.data.numpy()
            if is_cuda:
                tv_outputs_test = train_tv_outputs.cpu().data.numpy()
                ti_embed_test = ti_embed.cpu().data.numpy()
            else:
                tv_outputs_test = train_tv_outputs.data.numpy()
                ti_embed_test = ti_embed.data.numpy()
            #print(input_nodelabels)
            tv_labels_test_sum = np.sum(tv_labels_test,2)
            visitslen_batch = np.sum(tv_labels_test_sum>0,1)
            #print(tv_labels_test[:, :, :input_num_events])
            #print(tv_labels_test[:, :, :input_num_events].shape)
            #print(tv_outputs_test[:, :, :input_num_events])
            #print(tv_outputs_test[:, :, :input_num_events].shape)
            #print(visitslen_batch)
            recall = metrics.compute_recall_k(tv_labels_test[:, :, :input_num_events], tv_outputs_test[:, :, :input_num_events], visitslen_batch,recall_at)
            #metrics.plot_lowdim_space(ti_embeds[:input_num_events], ti_embeds[input_num_events:], itr)
            #print("Minibatch-iter : %d\tMinibatch-cost : %f\tRecall@%d : %f" % (n_itr, train_cost, recall_at, recall))
            if input_nodelabels is not None:
                nmi = metrics.compute_clustering_nmi(ti_embed_test[:input_num_events], input_nodelabels[:input_num_events])
                print("Training: Minibatch-iter : %d\tMinibatch-cost : %f\tClustering NMI : %f\tRecall@%d : %f" % (n_itr, train_cost, nmi, recall_at, recall))
                #output_write.write("Minibatch-iter : "+str(n_itr)+"\tMinibatch-cost : "+str(train_cost)+"\tClustering NMI :"+str(nmi)+"\tRecall@"+str(recall_at)+" : "+str(recall)+"\n")
            else:
                print("Training: Minibatch-iter : %d\tMinibatch-cost : %f\tRecall@%d : %f" % (n_itr, train_cost, recall_at, recall))
                #output_write.write("Minibatch-iter : "+str(n_itr)+"\tMinibatch-cost : "+str(train_cost)+"\tRecall@"+str(recall_at)+" : "+str(recall)+"\n")
                
            train_visit_label = []
            train_softmax_list = []
            train_maxlen_list = []
            train_patient_num = []
            
            
            train_input_visit =[]
            train_visitdegree = []
            train_visitedge = []
            # end train...
            
            
            
            
            
            torch.cuda.empty_cache()
            
            visit_label = []
            softmax_list = []
            maxlen_list = []
            patient_num = []
            visitlen_list = []
            
            for i,(batch_visitinput, batch_visitdegree, batch_visitedge, visitslen_batch) in enumerate(test_loader,0):
                torch.cuda.empty_cache()
                #print(torch.sum(batch_visitinput,2))
                #tttmp = torch.sum(batch_visitinput,2)
                #print(torch.sum(tttmp>0,1))
                #print(visitslen_batch)
                #softmax_matrix = whole_model.module.visit_model.get_softmax_matrix(batch_visitinput.to(device),whole_model.module.node_model.first_embedding,ti_embed,batch_visitedge.to(device),batch_visitdegree.to(device))
                softmax_matrix = whole_model.visit_model.get_softmax_matrix(batch_visitinput.to(device),first_embed,ti_embed,batch_visitedge.to(device),batch_visitdegree.to(device))
                visit_label.append(batch_visitinput)#b l n
                softmax_list.append(softmax_matrix)
                maxlen_list.append(len(batch_visitdegree[0])) #l
                patient_num.append(len(batch_visitdegree))#b
                visitlen_list.append(visitslen_batch)
            #print(patient_num)
            #print(len(visit_label))
            if len(visit_label) == 1:
                tv_labels = batch_visitinput
                tv_outputs = softmax_matrix
                maxlen = maxlen_list[0]
                tv_visitlen = visitslen_batch
            else:
                maxlen = max(maxlen_list)
                tv_outputs = torch.zeros(test_loader.count_allvisit,maxlen,len(ti_embed))
                tv_labels = torch.zeros(test_loader.count_allvisit, maxlen, len(ti_embed))
                start_id = 0
                end_id = 0
                tv_visitlen = []
                for index in range(len(visit_label)):
                    end_id = end_id+ patient_num[index]#len(patient_num)
                    tv_labels[start_id:end_id,:maxlen_list[index],:] =  visit_label[index]
                    tv_outputs[start_id:end_id, :maxlen_list[index], :] = softmax_list[index]
                    start_id = end_id
                    tv_visitlen += visitlen_list[index]
                #print(visitlen_list)
                #print(tv_visitlen)

            tv_labels_test = tv_labels.data.numpy()
            if is_cuda:
                tv_outputs_test = tv_outputs.cpu().data.numpy()
                ti_embed_test = ti_embed.cpu().data.numpy()
            else:
                tv_outputs_test = tv_outputs.data.numpy()
                ti_embed_test = ti_embed.data.numpy()
            #print(input_nodelabels)
        recall = metrics.compute_recall_k(tv_labels_test[:, :, :input_num_events], tv_outputs_test[:, :, :input_num_events], tv_visitlen,recall_at)
        tv_labels_test_sum = np.sum(tv_labels_test,2)
        visitslen_batch_temp = np.sum(tv_labels_test_sum>0,1)
        #print(tv_visitlen[1200:])
        #print(tv_labels_test_sum[1200:])
        #print(visitslen_batch_temp[1200:])
        #metrics.plot_lowdim_space(ti_embeds[:input_num_events], ti_embeds[input_num_events:], itr)
        #print("Minibatch-iter : %d\tMinibatch-cost : %f\tRecall@%d : %f" % (n_itr, train_cost, recall_at, recall))
        visit_label = []
        softmax_list = []
        maxlen_list = []
        patient_num = []
        visitlen_list = []
        if input_nodelabels is not None:
            nmi = metrics.compute_clustering_nmi(ti_embed_test[:input_num_events], input_nodelabels[:input_num_events])
            print("Minibatch-iter : %d\tMinibatch-cost : %f\tClustering NMI : %f\tRecall@%d : %f" % (n_itr, train_cost, nmi, recall_at, recall))
            output_write.write("Minibatch-iter : "+str(n_itr)+"\tMinibatch-cost : "+str(train_cost)+"\tClustering NMI :"+str(nmi)+"\tRecall@"+str(recall_at)+" : "+str(recall)+"\n")
        else:
            print("Minibatch-iter : %d\tMinibatch-cost : %f\tRecall@%d : %f" % (n_itr, train_cost, recall_at, recall))
            output_write.write("Minibatch-iter : "+str(n_itr)+"\tMinibatch-cost : "+str(train_cost)+"\tRecall@"+str(recall_at)+" : "+str(recall)+"\n")
        whole_model.train()
    if n_itr % (n_printiters * 10) == 0:
        whole_model.eval()
        if is_cuda:
            if len(device_ids)>1:
                node_model_test.conv1.weight.data = whole_model.module.state_dict()['node_model.conv1.weight'].data.cpu()
                node_model_test.conv2.weight.data = whole_model.module.state_dict()['node_model.conv2.weight'].data.cpu()
                _, ti_embed = node_model_test.get_firstembedding_and_fina_embed()
                ti_embed = ti_embed.data.numpy()
            else:
                _, ti_embed = whole_model.node_model.get_firstembedding_and_fina_embed()
                ti_embed = ti_embed.cpu().data.numpy()
            np.save("%s/embedding_%d.npy" % (output_dir, n_itr), ti_embed)
            torch.save(whole_model.state_dict(), output_dir+'params_'+str(n_itr)+'.pkl')
        else:
            _, ti_embed = whole_model.node_model.get_firstembedding_and_fina_embed()
            np.save("%s/embedding_%d.npy" % (output_dir, n_itr), ti_embed.data.numpy())
            torch.save(whole_model.state_dict(), output_dir+'params_'+str(n_itr)+'.pkl')
        #if is_cuda:
        #    np.save("%s/embedding_%d.npy" % (output_dir, n_itr), whole_model.module.embed.cpu().data.numpy())
        #else:
        #    np.save("%s/embedding_%d.npy" % (output_dir, n_itr), whole_model.module.embed.data.numpy())
        #print(ti_embed)
        whole_model.train()
    
    train_cost = 0

print("Optimization finished!")
output_write.write("Optimization finished!")
# Save the embedding vectors of events and concepts
whole_model.eval()
if is_cuda:
    if len(device_ids)>1:
        node_model_test.conv1.weight.data = whole_model.module.state_dict()['node_model.conv1.weight'].data.cpu()
        node_model_test.conv2.weight.data = whole_model.module.state_dict()['node_model.conv2.weight'].data.cpu()
        _, ti_embed = node_model_test.get_firstembedding_and_fina_embed()
        ti_embed = ti_embed.data.numpy()
    else:
        _, ti_embed = whole_model.node_model.get_firstembedding_and_fina_embed()
        ti_embed = ti_embed.data.cpu().numpy()
    np.save("%s/embedding_%d.npy" % (output_dir, n_itr), ti_embed)
    torch.save(whole_model.state_dict(), output_dir+'params_'+str(n_itr)+'.pkl')
else:
    _, ti_embed = whole_model.node_model.get_firstembedding_and_fina_embed()
    np.save("%s/embedding_%d.npy" % (output_dir, n_itr), ti_embed.data.numpy())
    torch.save(whole_model.state_dict(), output_dir+'params_'+str(n_itr)+'.pkl')

    
end = time.time()
print(end-start)
output_write.write(str(end-start))
output_write.close()
#if is_cuda:
#    np.save("%s/embedding_final.npy" % output_dir, whole_model.module.embed.cpu().data.numpy())
#else:
#    np.save("%s/embedding_final.npy" % output_dir, whole_model.module.embed.data.numpy())
