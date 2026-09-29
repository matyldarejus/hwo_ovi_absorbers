import h5py
import numpy as np

def write_dict_to_h5(mydict, h5file):
    
    with h5py.File(h5file, 'a') as f:
        for key in mydict.keys():
            
            if type(mydict[key]) == dict:
                new_group = f.create_group(key) 
                for sub_key in mydict[key].keys():
                    new_group.create_dataset(sub_key, data=np.array(mydict[key][sub_key]))
            
            else:
                if not key in list(f.keys()):
                    f.create_dataset(key, data=np.array(mydict[key]))


def read_h5_into_dict(h5file):

    mydict = {}
    with h5py.File(h5file, 'r') as f:
        for key in f.keys():

            if type(f[key]) == h5py._hl.group.Group:
                mydict[key] = {}

                for sub_key in f[key].keys():

                    if len(f[key][sub_key].shape) == 0:
                        mydict[key][sub_key] = f[key][sub_key][()]
                    else:
                        mydict[key][sub_key] = f[key][sub_key][:]

            else:
                if len(f[key].shape) == 0:
                    mydict[key] = f[key][()]
                else:
                    mydict[key] = f[key][:]

                for attr_k in f[key].attrs.keys():
                    mydict[attr_k] = f[key].attrs[attr_k]

    return mydict

