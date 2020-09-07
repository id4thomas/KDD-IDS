import os
import numpy as np

from preprocessor import *
from sklearn.model_selection import train_test_split

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

#Preprocess Data
df_final,label=preprocess(df,serviceData,flagData,labeled=True)

print(label.shape)
#Combine Label
df_final = np.concatenate((df_final.values, np.expand_dims(label, axis=1)), axis=1)
df_final = pd.DataFrame(df_final)

#Save Data
with h5py.File(kdd_path+'/processed/kddcup.hdf5', 'w') as hdf:
    print('Saving file : {}'.format(kdd_path+'/processed/kddcup.hdf5'))
    hdf['x'] = df_final.values[:]

#Save Label
# np.save(kdd_path+'/processed/kddcup_label.npy',label)

#Split Train/Test
train, test = train_test_split(df_final, test_size=0.5)

#114 + 1 (Label)
print("Train {} Test {}".format(train.shape,test.shape))

with h5py.File(kdd_path+'/processed/train.hdf5', 'w') as hdf:
    print('Saving file : {}'.format(kdd_path+'/processed/train.hdf5'))
    hdf['x'] = df_final.values[:]

with h5py.File(kdd_path+'/processed/test.hdf5', 'w') as hdf:
    print('Saving file : {}'.format(kdd_path+'/processed/test.hdf5'))
    hdf['x'] = df_final.values[:]
