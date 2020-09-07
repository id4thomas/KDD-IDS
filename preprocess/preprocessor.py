import os
import h5py

import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.utils import shuffle


COLS = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]

FEATURES=[
    "duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","count","serror_rate","srv_serror_rate",
    "same_srv_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_src_port_rate","dst_host_serror_rate","dst_host_srv_serror_rate"
]

def to_numeric(df,service_list,flag_list,labeled=False):
     # Protocol
    print('Protocol -> Numeric')
    df["protocol_type"].replace(['tcp', 'udp', 'icmp'], range(3), inplace=True)

    # Service
    print('Service -> Numeric')
    df["service"].replace(service_list, range(len(service_list)), inplace=True)

    # Flag
    print('Flag -> Numeric')
    df["flag"].replace(flag_list, range(len(flag_list)), inplace=True)

    df_data=df.loc[:,FEATURES]
    if labeled:
        df_label=df.loc[:,"label"]
        df_label = df_label.map(lambda x: 0 if x == 'normal.' else 1)  # normal 0, attack 1
        print(df_label.head)
        label=df_label.to_numpy()
        print("data {} label {}".format(df_data.shape,label.shape))
    else:
        label=[]
    return df_data,label

def preprocess(df,service_list,flag_list,labeled=False):
    print(df.shape)
    df_data,label=to_numeric(df,service_list,flag_list,labeled=labeled)

    scaler = MinMaxScaler()
    enc = OneHotEncoder(categories=[range(3), range(len(service_list)), range(len(flag_list))])
    numericDataDesc = df_data.loc[:, ["duration","src_bytes", "dst_bytes"]].describe()

    # Duration,src_bytes,dst_bytes
    attr_name = ["duration","src_bytes", "dst_bytes"]
    for i in attr_name:
        print('Processing - {}'.format(i))
        iqr = (numericDataDesc[i].values[6] - numericDataDesc[i].values[4])
        std = numericDataDesc[i].values[6] + iqr * 1.5  # IQR upper fence = Q3 + 1.5 * IQR
        if std == 0:
            df_data[i] = df_data[i].map(lambda x: 1 if x > 0 else 0)
        else:
            df_data[i] = df_data[i].map(lambda x: std if x > std else x)

    scaler.fit(df_data[attr_name].values)
    df_data[attr_name]= scaler.transform(df_data[attr_name].values)

    print('Processing - count')
    df_data["count"] = df_data["count"]/100

    print('Processing - dst host count')
    df_data["dst_host_count"] = df_data["dst_host_count"]/100

    print('Processing - dst host srv count')
    df_data["dst_host_srv_count"] = df_data["dst_host_srv_count"]/100

    #To Onehot
    enc.fit(df_data[["protocol_type","service","flag"]].values)
    oneHotEncoding = enc.transform(df_data[["protocol_type","service","flag"]].values).toarray()
    df_data.drop(["protocol_type","service","flag"],axis=1,inplace=True)

    df_final = np.concatenate((df_data.values, oneHotEncoding), axis=1)
    df_final = pd.DataFrame(df_final)

    print(df_final.shape)
    return df_final,label

if __name__ == "__main__":
    kdd_path='../../kdd_data'
    #Read Service, Flag
    service = open(kdd_path+'/service.txt', 'r')
    serviceData = service.read().split('\n')
    service.close()

    flag = open(kdd_path+'/flag.txt', 'r')
    flagData = flag.read().split('\n')
    flag.close()

    #Create Processed Folder
    if not os.path.exists(kdd_path+'/processed'):
        os.makedirs(kdd_path+'/processed')

    #Read Data
    df = pd.read_csv(kdd_path+'/kddcup.data', names=COLS)
    print(df['label'].unique())
    print(df['label'].value_counts())

    df_final,label=preprocess(df,serviceData,flagData,labeled=True)

    #Save Data
    with h5py.File(kdd_path+'/processed/kddcup.hdf5', 'w') as hdf:
        print('Saving file : {}'.format(kdd_path+'/processed/kddcup.hdf5'))
        hdf['x'] = df_final.values[:]

    #Save Label
    np.save(kdd_path+'/processed/kddcup_label.npy',label)
