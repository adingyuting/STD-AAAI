import torch
import torch.utils.data
from data.dataprovider import CustomProvider

data_dict = {
    'D1': CustomProvider,
    'D2': CustomProvider,
    'D3': CustomProvider,
    'D4': CustomProvider,
}

def data_loader(dataset, batch_size, shuffle=True, drop_last=True):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)
    return dataloader


def load_data(dataset,batch_size, sample_len,output_len, window_size, \
              input_dim , output_dim ,\
               train_ratio, val_ratio, data_path , adj_path ,target_strategy, few_shot = 1, node_shuffle_seed = None):

    dataprovider = data_dict[dataset](data_path, adj_path,dataset,node_shuffle_seed)

    train_set, val_set, test_set = dataprovider.getdataset(sample_len=sample_len,output_len=output_len,window_size=window_size, \
                                                           input_dim = input_dim , output_dim = output_dim,
                                                           train_ratio=train_ratio,val_ratio=val_ratio,target_strategy=target_strategy, few_shot = few_shot)

    train_loader = data_loader(train_set, batch_size=batch_size)

    val_loader = data_loader(val_set, batch_size=batch_size)

    test_loader = data_loader(test_set, batch_size=batch_size, shuffle=False, drop_last=False)


    scaler = dataprovider.scaler
    node_num, features = dataprovider.node_num, dataprovider.features

    adj_mx, distance_mx = dataprovider.getadj()

    return train_loader, val_loader, test_loader,\
           scaler,  node_num, features , \
           adj_mx, distance_mx
