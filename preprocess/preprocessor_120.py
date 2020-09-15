import os
import h5py

import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler, Normalizer
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

#Selected Features
FEATURES=[
    "duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","count","serror_rate","srv_serror_rate",
    "same_srv_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_src_port_rate","dst_host_serror_rate","dst_host_srv_serror_rate"
]

#All Features
#num_outbound_cmds & is_host_login redundant
CONT_FEATURES=[
    "duration","src_bytes","dst_bytes","wrong_fragment","urgent","hot",
    "num_failed_logins","num_compromised","root_shell",
    "su_attempted","num_root","num_file_creations","num_shells",
    "num_access_files","count","srv_count",
    "serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
    "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count",
    "dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
    "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate"
]

#land ~ is_guest_login : 0/1 - already int64
CAT_FEATURES=[
    "protocol_type","service","flag","land","logged_in","is_guest_login"
]


def to_numeric(df,service_list,flag_list,labeled=False):
    # Protocol
    print('Protocol -> Numeric')
    df["protocol_type"].replace(['tcp', 'udp', 'icmp'], range(3), inplace=True)

    # # Service
    print('Service -> Numeric')
    df["service"].replace(service_list, range(len(service_list)), inplace=True)

    # # Flag
    print('Flag -> Numeric')
    df["flag"].replace(flag_list, range(len(flag_list)), inplace=True)

    # df_data=df.loc[:,FEATURES]
    df_data=df.loc[:,CONT_FEATURES+CAT_FEATURES]

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
    print(len(service_list))
    print(len(flag_list))
    print('Selected',df_data.shape)
    scaler=Normalizer()

    #Continuous Data
    data_cont=df_data[CONT_FEATURES].values
    data_cont=scaler.fit_transform(data_cont)

    #Binary Category Data
    data_bin=df_data[CAT_FEATURES[3:]].values

    #Categorical Data
    enc = OneHotEncoder(categories=[range(3), range(len(service_list)), range(len(flag_list))])
    enc.fit(df_data[["protocol_type","service","flag"]].values)
    oneHotEncoding = enc.transform(df_data[["protocol_type","service","flag"]].values).toarray()

    print(oneHotEncoding.shape)
    df_final = np.concatenate((data_cont, oneHotEncoding,data_bin), axis=1)
    df_final = pd.DataFrame(df_final)
    print(df_final.shape)
    return df_final,label

def make_cat(df,kdd_path):
    lenc=LabelEncoder()
    lenc.fit_transform(df["service"].values)
    service_list=lenc.classes_
    f=open(kdd_path+'/service.txt','w')
    service_list='\n'.join(service_list)
    f.write(service_list)
    f.close()

    lenc=LabelEncoder()
    lenc.fit_transform(df["flag"].values)
    flag_list=lenc.classes_
    f=open(kdd_path+'/flag.txt','w')
    flag_list='\n'.join(flag_list)
    f.write(flag_list)
    f.close()


if __name__ == "__main__":
    kdd_path='../../kdd_data'

    #Load DF
    df = pd.read_csv(kdd_path+'/kddcup.data_10_percent', names=COLS)

    #Make Category
    make_cat(df,kdd_path)

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
    print(df['label'].unique())
    print(df['label'].value_counts())

    df_final,label=preprocess(df,serviceData,flagData,labeled=True)

    save_path='../../kdd_120'
    #Save Data
    with h5py.File(save_path+'/processed/kddcup_10.hdf5', 'w') as hdf:
        print('Saving file : {}'.format(save_path+'/processed/kddcup_10.hdf5'))
        hdf['x'] = df_final.values[:]

    #Save Label
    np.save(save_path+'/processed/kddcup_10_label.npy',label)
