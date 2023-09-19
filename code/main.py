import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp
import dgl
from  load_data import *
from model import *
from utils import *
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
from mlflow.tracking import MlflowClient
import time
import mlflow
import warnings
warnings.filterwarnings('ignore')

seed = 13
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

torch.autograd.set_detect_anomaly(True)

n_his = 12
n_pred = 1
batch_size = 1
epochs = 50
lr = 0.0005

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
traffic_filename = 'data/traffic.csv'
traffic_mx = pd.read_csv(traffic_filename,header = None)
# print(traffic_mx.shape)
traffic_mx = np.log10(traffic_mx.values + 1)
traffic_train = traffic_mx[:,:1950]  #1950-2366-
traffic_val = traffic_mx[:,1950:2366]
traffic_test = traffic_mx[:,2366:]

embedding_graph_filename = 'data/base_kg.csv' # spatial KG
embedding_graph = pd.read_csv(embedding_graph_filename,header = None)
embedding_graph = embedding_graph.values
g_base = dgl.graph((embedding_graph[:,0],embedding_graph[:,1])).to(device)
etype_base = torch.tensor(embedding_graph[:,2].reshape(-1)).to(device)

 
embedding_filename = 'data/kg_embedding.csv'  #pre-train from TuckER
emd_mx = pd.read_csv(embedding_filename,header = None)
emd_mx = emd_mx.values
emd_mx = torch.tensor(emd_mx).to(torch.float32).to(device)

embedding_graph_filename = 'data/nanjing_kg.csv' #augmented spatial KG
embedding_graph = pd.read_csv(embedding_graph_filename,header = None)
embedding_graph = embedding_graph.values

g_urban = dgl.graph((embedding_graph[:,0],embedding_graph[:,2])).to(device)
etype_urban = torch.tensor(embedding_graph[:,1].reshape(-1)).to(device)

num_nodes = traffic_train.shape[0]

traffic_train = traffic_train.T
traffic_val = traffic_val.T
traffic_test = traffic_test.T
scaler = StandardScaler()
traffic_train = scaler.fit_transform(traffic_train)
traffic_val = scaler.transform(traffic_val)
traffic_test = scaler.transform(traffic_test)
traffic_train = traffic_train.T
traffic_val = traffic_val.T
traffic_test = traffic_test.T

x_train,y_train = data_transform(traffic_train,n_his,n_pred)
x_val,y_val = data_transform(traffic_val,n_his,n_pred)
x_test,y_test = data_transform(traffic_test,n_his,n_pred)
# print(x_train.shape,y_train.shape)

train_data = torch.utils.data.TensorDataset(x_train, y_train)
train_iter = torch.utils.data.DataLoader(train_data, batch_size, shuffle = True)
val_data = torch.utils.data.TensorDataset(x_val, y_val)
val_iter = torch.utils.data.DataLoader(val_data, batch_size)
test_data = torch.utils.data.TensorDataset(x_test, y_test)
test_iter = torch.utils.data.DataLoader(test_data, batch_size)

hidden_dim = 16
gcn_emb_dim = 64
trans_emb_dim = 64
n_trans_layers = 2
n_rgcn_layers = 1
num_bases = 4
kernal_size = 3


experiment_name = 'multi-relational_knowledge_graph_convolutional_network'
mlflow.set_tracking_uri('./mlflow_output')
client = MlflowClient( )
try:
    EXP_ID = client.create_experiment(experiment_name)
    print('Initial Create!', EXP_ID)
except:
    experiments = client.get_experiment_by_name(experiment_name)
    EXP_ID = experiments.experiment_id
    print('Experiment Exists, Continuing', EXP_ID)
with mlflow.start_run(experiment_id = EXP_ID):
    archive_path = mlflow.get_artifact_uri( )
    params = {'dataset': 'nanjing', 'n_trans_layers': n_trans_layers,
              'hidden_dim': hidden_dim,'num_bases': num_bases,'lr':lr,'kernal_size':kernal_size,
              'gcn_emb_size': gcn_emb_dim, 'trans_emb_size': trans_emb_dim}
    mlflow.log_params(params)

    save_path = 'models/multi-relational_knowledge_graph_convolutional_network.pt'  ##########################################


    loss = nn.MSELoss()
    loss = loss.to(device)
    model = MRKGCN().to(device) ################################
    optimizer = torch.optim.Adam(model.parameters(),lr)
    min_val_loss = np.inf

    for epoch in range(1,epochs):
        start_train = time.time( )
        print('epoch:',epoch)
        total_train_step = 0
        l_sum, n = 0.0, 0
        model.train( )
        for x,y in train_iter:
            x = x.to(device)
            y = y.to(device)

            y_pred = model(x,g_base,etype_base,emd_mx,g_urban,etype_urban)
            l = loss(y_pred,y)
            optimizer.zero_grad( )
            l.backward( )
            optimizer.step( )
            l_sum += l.item( ) * y.shape[0]
            n += y.shape[0]
            total_train_step += 1


            # if total_train_step % 10 == 0:
            #     print('train_step:{},loss = {}'.format(total_train_step,l.item()))
            #
        val_loss,MAE,RMSE,r2 = evaluate_mdoel(model,loss,val_iter,g_base,etype_base,emd_mx,g_urban,etype_urban,scaler,device)
        # print("epoch", epoch, ", train loss:", l_sum / n, ", validation loss:", val_loss)
        # print("MAE:", MAE, ", RMSE:", RMSE, ", R2:", r2)
        mlflow.log_metrics(
            {'train_time': time.time( ) - start_train, 'train_loss': l_sum / n, 'validation loss': val_loss,
             'MAE': MAE, 'RMSE': RMSE, 'R2': r2},
            step = epoch)

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict( ), save_path)


best_model = MRKGCN().to(device)##############################
best_model.load_state_dict(torch.load(save_path))
test_loss, MAE, RMSE, r2 = evaluate_mdoel(best_model,loss,test_iter,g_base,etype_base,emd_mx,g_urban,etype_urban,scaler,device)
print("test loss:", test_loss, "\nMAE:", MAE, ", RMSE:", RMSE, ", R2:", r2)
