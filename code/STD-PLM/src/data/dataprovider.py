import torch
import numpy as np
import torch.utils.data
from utils.utils import get_randmask, linear_interpolate, get_block_mask
import pandas as pd
from data.scaler import StandardScaler
from pathlib import Path
from scipy.io import loadmat
from sklearn.metrics.pairwise import haversine_distances


def load_table(path: Path):
    """Read .mat/.csv/.txt automatically."""
    if path.suffix.lower() == ".mat":
        mat = loadmat(path)
        key = next(k for k in mat if not k.startswith("__"))
        return mat[key]
    return pd.read_csv(path, sep="::", engine="python", header=None).values

def build_adj_from_pos(pos: np.ndarray):
    """Generate adjacency and distance matrices from coordinates."""
    coords = np.radians(pos)
    dist = haversine_distances(coords) * 6371000
    sigma = dist[dist > 0].std()
    adj = np.exp(-(dist ** 2) / (2 * sigma ** 2))
    adj[adj < 1e-2] = 0
    np.fill_diagonal(adj, 1)
    return adj.astype(np.float32), dist.astype(np.float32)

def generate_sample_by_sliding_window(data, sample_len, step=1):

    sample = []

    for i in range(0, data.shape[0] - sample_len, step):

        sample.append(torch.unsqueeze(data[i:i+sample_len] , 0))
    
    if (data.shape[0] - sample_len) % step !=0 :
        sample.append(torch.unsqueeze(data[-sample_len:] , 0))

    sample = torch.concat(sample,dim=0)

    return sample

class BasicDataset(torch.utils.data.Dataset):

    history  : torch.Tensor #(B,sample_len,node_num,features)
    target   : torch.Tensor #(B,(sample_len)+predict_len,node_num,features)
    timestamp: torch.Tensor #(B,sample_len*2, 4 ) 12(months) + 31(day) + 7(week) + 24(day)
    cond_mask: torch.Tensor #(B,sample_len,node_num)
    ob_mask  : torch.Tensor #(B,sample_len,node_num)

    def __init__(self, history, target, timestamp,cond_mask,ob_mask,training=False) -> None:
        
        self.history = history
        self.target = target
        self.timestamp = timestamp
        self.cond_mask = cond_mask
        self.ob_mask = ob_mask
        self.training = training

    def __len__(self):

        return self.history.shape[0]
    
    def __getitem__(self, index):

        # cond_mask = self.cond_mask[index]
        # if self.training:
        #     cond_mask = get_block_mask(self.ob_mask[index].cpu(), target_strategy='hybrid',min_seq=3, max_seq=12).cuda()
        
        return (self.history[index], self.target[index], self.timestamp[index], self.cond_mask[index], self.ob_mask[index])


class DataProvider():

    node_num : int
    features : int
    data  : torch.Tensor #(T, node_num,features)
    timestamp: torch.Tensor #(T,sample_len+predict_len, 5 ) 12(months) + 31(day) + 7(week) + 24(hours) +60(minutes)
    mask:torch.Tensor #(T, node_num,features) 
    mask_eval:torch.Tensor #(T, node_num,features) position for evaluate, mask_eval&mask=mask

    def __init__(self, data_path, adj_path ,dataset, node_shuffle_seed=None) -> None:

        self.dataset = dataset

        self.data, self.node_num, self.features, \
        self.adj_mx, self.distance_mx, \
        self.timestamp,self.mask,self.mask_eval =  self.read_data(data_path, adj_path)

        if node_shuffle_seed is not None:
            rdm = np.random.RandomState(node_shuffle_seed)
            idx = np.arange(self.node_num)
            rdm.shuffle(idx)
            idx = torch.from_numpy(idx)
            self.data = self.data[:,idx,:]
            self.adj_mx = self.adj_mx[idx,:][:,idx]


    def getdataset(self, sample_len,output_len, window_size, \
                    input_dim , output_dim , \
                   train_ratio, val_ratio,target_strategy, few_shot = 1):

        self.data = self.data.float().cuda()

        self.timestamp = self.timestamp.cuda()
        self.mask = self.mask.float().cuda()
        self.mask_eval = self.mask_eval.float().cuda()

        all_len = self.data.shape[0]
        train_len = int(all_len * train_ratio)
        val_len = int(all_len * val_ratio)

        train_range = [0,int(train_len * few_shot)]
        val_range = [train_len, train_len+val_len]
        test_range = [train_len+val_len, all_len]

        scaler_mask = self.mask[train_range[0]:train_range[1]]!=0
        scaler_data = self.data[train_range[0]:train_range[1]]
        dim = scaler_data.shape[-1]
        mean = [scaler_data[...,i:i+1][scaler_mask[...,i:i+1]].mean() for i in range(dim)]
        std = [scaler_data[...,i:i+1][scaler_mask[...,i:i+1]].std() for i in range(dim)]
        self.scaler = self.getscalerclass()(mean,std)
        #self.data = self.scaler.transform(self.data)

        train_data = self.data[train_range[0]:train_range[1]]
        train_te = self.timestamp[train_range[0]:train_range[1]]
        train_mask, train_mask_eval = self.mask[train_range[0]:train_range[1]], self.mask_eval[train_range[0]:train_range[1]]
        train_sample = generate_sample_by_sliding_window(train_data, sample_len=window_size)
        train_x, train_y = train_sample[:,:sample_len,...][...,:input_dim],  train_sample[:,-output_len:,...][...,:output_dim]
        train_x = self.scaler.transform(train_x)
        train_te = generate_sample_by_sliding_window(train_te, sample_len=window_size)
        train_ob_mask = generate_sample_by_sliding_window(train_mask, sample_len=window_size)[...,:input_dim]
        if target_strategy=='random':
            train_cond_mask = get_randmask(train_ob_mask[:,:sample_len],0,1).cuda()[...,:input_dim]
        else :
            train_len = train_ob_mask.shape[0]
            train_cond_mask = torch.concat([get_block_mask(train_ob_mask[i,:sample_len].cpu(), target_strategy='hybrid',min_seq=3, max_seq=12).cuda().unsqueeze(0) for i in range(train_len)])[...,:input_dim]
        #train_x[train_cond_mask==0] = 0
        #train_x[train_cond_mask==0] = torch.nan
        #B,T,N,F = train_x.shape
        #train_x = linear_interpolate(train_x.view(B,T,N,F,1).permute(0,2,3,1,4).contiguous()).view(B,N,F,T).permute(0,3,2,1).contiguous()
        train_dataset = BasicDataset(history=train_x, target=train_y, timestamp=train_te,cond_mask=train_cond_mask,ob_mask=train_ob_mask,training=True)

        val_data = self.data[val_range[0]:val_range[1]]
        val_te = self.timestamp[val_range[0]:val_range[1]]
        val_mask, val_mask_eval = self.mask[val_range[0]:val_range[1]], self.mask_eval[val_range[0]:val_range[1]]
        val_sample = generate_sample_by_sliding_window(val_data, sample_len=window_size)
        val_x, val_y = val_sample[:,:sample_len,...][...,:input_dim],  val_sample[:,-output_len:,...][...,:output_dim]
        val_x = self.scaler.transform(val_x)
        val_te = generate_sample_by_sliding_window(val_te, sample_len=window_size)

        val_ob_mask = generate_sample_by_sliding_window(val_mask_eval, sample_len=window_size)[...,:input_dim]
        val_cond_mask = generate_sample_by_sliding_window(val_mask, sample_len=window_size)[:,:sample_len][...,:input_dim]
        #val_x[val_cond_mask==0] = 0
        #val_x[val_cond_mask==0] = torch.nan
        #B,T,N,F = val_x.shape
        #val_x = linear_interpolate(val_x.view(B,T,N,F,1).permute(0,2,3,1,4).contiguous()).view(B,N,F,T).permute(0,3,2,1).contiguous()
        val_dataset = BasicDataset(history=val_x, target=val_y, timestamp=val_te, cond_mask=val_cond_mask,ob_mask=val_ob_mask)

        test_data = self.data[test_range[0]:test_range[1]]
        test_te = self.timestamp[test_range[0]:test_range[1]]
        test_mask, test_mask_eval = self.mask[test_range[0]:test_range[1]], self.mask_eval[test_range[0]:test_range[1]]
        test_sample = generate_sample_by_sliding_window(test_data, sample_len=window_size)
        test_x, test_y = test_sample[:,:sample_len,...][...,:input_dim],  test_sample[:,-output_len:,...][...,:output_dim]
        test_x = self.scaler.transform(test_x)
        test_te = generate_sample_by_sliding_window(test_te, sample_len=window_size)

        test_ob_mask = generate_sample_by_sliding_window(test_mask_eval, sample_len=window_size)[...,:input_dim]
        test_cond_mask = generate_sample_by_sliding_window(test_mask, sample_len=window_size)[:,:sample_len][...,:input_dim]
        #test_x[test_cond_mask==0] = 0
        #test_x[test_cond_mask==0] = torch.nan
        #B,T,N,F = test_x.shape
        #test_x = linear_interpolate(test_x.view(B,T,N,F,1).permute(0,2,3,1,4).contiguous()).view(B,N,F,T).permute(0,3,2,1).contiguous()
        test_dataset = BasicDataset(history=test_x, target=test_y, timestamp=test_te, cond_mask=test_cond_mask,ob_mask=test_ob_mask)


        return train_dataset, val_dataset, test_dataset
                

    def getadj(self):

        return self.adj_mx, self.distance_mx
    
    def getscalerclass(self):
        
        return StandardScaler


def generatetimestamp(start, periods, freq):

    time = pd.date_range(start=start,periods=periods,freq=freq)

    month = np.reshape(time.month, (-1, 1))
    dayofmonth = np.reshape(time.day, (-1, 1))
    dayofweek = np.reshape(time.weekday, (-1, 1))
    hour = np.reshape(time.hour, (-1, 1))
    minute = np.reshape(time.minute, (-1, 1))

    timestamp = np.concatenate((month, dayofmonth, dayofweek, hour, minute), -1)

    timestamp = torch.tensor(timestamp)

    return timestamp

timestampfun = {
    'D1': lambda T: generatetimestamp(start='2014-05-01 00:00:00', periods=T, freq='1D'),
    'D2': lambda T: generatetimestamp(start='2014-05-01 00:00:00', periods=T, freq='1D'),
    'D3': lambda T: generatetimestamp(start='1981-09-14 00:00:00', periods=T, freq='1W'),
    'D4': lambda T: generatetimestamp(start='2010-01-01 00:00:00', periods=T, freq='1D'),
}


class CustomProvider(DataProvider):
    """Load datasets stored as dense matrices with optional positions.

    ``data_path`` should point to a table of shape ``(node, time)`` saved as
    ``.mat`` or text/CSV file separated by ``::``. The loader transposes the
    array to ``(time, node, 1)`` and constructs a mask where non-NaN values are
    marked as observed. If ``adj_path`` is supplied, a position file with
    longitude/latitude is used to build adjacency and distance matrices
    on-the-fly; otherwise identity matrices are returned.
    """

    def read_data(self, data_path, adj_path=None):
        """Read dense matrix and optional position file.

        If ``data_path`` or ``adj_path`` is ``None`` they are inferred from
        ``datasets/<dataset>/`` following the naming convention
        ``data.*`` and ``position.*`` respectively.
        """
        ds_dir = Path("datasets") / self.dataset
        if data_path is None:
            try:
                data_path = next(ds_dir.glob("data.*"))
            except StopIteration as e:
                raise FileNotFoundError(f"No data file found under {ds_dir}") from e
        table = load_table(Path(data_path))
        data = torch.from_numpy(table.T[..., None].astype(np.float32))
        mask = (~torch.isnan(data)).float()
        data = torch.nan_to_num(data)

        if adj_path is None:
            adj_path = next(ds_dir.glob("position.*"), None)

        if adj_path is not None:
            pos = load_table(Path(adj_path))
            adj_mx, distance_mx = build_adj_from_pos(pos)
        else:
            node_num = data.shape[1]
            adj_mx = np.eye(node_num, dtype=np.float32)
            distance_mx = adj_mx

        T, node_num, features = data.shape
        timestamp = timestampfun[self.dataset](T)

        return data, node_num, features, adj_mx, distance_mx, timestamp, mask, mask.clone()



