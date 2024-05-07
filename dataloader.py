from config import *
import h5py
import torch.utils.data as data
import glob
import random

class H5Dataset(data.Dataset):

    def __init__(self, root_path,patch_size,Whole_val, mode='train', ILD = False,pos_x=0,pos_y=0,pos_z=0):
        self.hdf5_list = [x for x in glob.glob(os.path.join(root_path, '*.h5'))]
        self.patches_size = patch_size
        self.mode = mode
        self.ILD = ILD
        self.whole_val = Whole_val
        self.indx_x = pos_x
        self.indx_y = pos_y
        self.indx_z = pos_z
        if (self.mode == 'train'):
            self.hdf5_list =self.hdf5_list + self.hdf5_list + self.hdf5_list + self.hdf5_list 
        
        if ILD == True:
            chn_cmbs = 2**4  #combinaciones de canales  
            self.channels_to_use = []
            for data_in in self.hdf5_list:          
                self.channels_to_use.append(random.randint(1,chn_cmbs))
            


    def __getitem__(self, index):
        h5_file = h5py.File(self.hdf5_list[index],"r")
        
        self.data = h5_file.get('data')
        self.label = h5_file.get('label')      
        self.label=self.label[:,0,...]        
        _, _, C, H, W = self.data.shape
        #print(self.data.shape)
        if (self.mode=='train'):
            cx = self.indx_x
            cy = self.indx_y
            cz = self.indx_z
            #print("tamaño img:", np.shape(self.data))
            self.data_crop  = self.data [:, :, cx: cx + self.patches_size[0], cy: cy + self.patches_size[1], cz: cz + self.patches_size[2]]
            self.label_crop = self.label[:,  cx: cx + self.patches_size[0], cy: cy + self.patches_size[1], cz: cz + self.patches_size[2]]
            #print(np.shape(self.data_crop))
            if self.ILD == True:             
                chnl_tu = self.channels_to_use[index]
                #print(chnl_tu) 
                chnl_tu = bin(chnl_tu)[2:].zfill(4) 
                #print(chnl_tu)
                for chnl_drp in range(0,4):
                    if int(chnl_tu[chnl_drp]) == 0:
                      self.data_crop[:,chnl_drp,:,:,:] = np.copy(self.data_crop[:,chnl_drp,:,:,:])*0
                 #     print("deleting channel ok")
                return (torch.from_numpy(self.data_crop[0,:,:,:,:]).float(),
                    torch.from_numpy(self.label_crop[0,:,:,:]).long(),chnl_tu)
                
            else:
                return (torch.from_numpy(self.data_crop[0,:,:,:,:]).float(),
                    torch.from_numpy(self.label_crop[0,:,:,:]).long())
            
            
            
        elif (self.mode == 'val'):
            cx = self.indx_x
            cy = self.indx_y
            cz = self.indx_z
            #print(self.data)
            if self.whole_val == False:
                 self.label_crop = self.label[:,  cx: cx + self.patches_size[0], cy: cy + self.patches_size[1], cz: cz + self.patches_size[2]]
                 self.data_crop  = self.data [:, :, cx: cx + self.patches_size[0], cy: cy + self.patches_size[1], cz: cz + self.patches_size[2]]
                 if self.ILD == True:
                     chnl_tu = self.channels_to_use[index]
                     chnl_tu = bin(chnl_tu)[2:].zfill(4)
                     for chnl_drp in range(0,4): 
                      if int(chnl_tu[chnl_drp]) == 0:
                         self.data_crop[:,chnl_drp,:,:,:] = np.copy(self.data_crop[:,chnl_drp,:,:,:])*0  
                     return(torch.from_numpy(self.data_crop[0,:,:,:,:]).float(), torch.from_numpy(self.label_crop[0,:,:,:]).long(), chnl_tu)

                 else:
                   return(torch.from_numpy(self.data_crop[0,:,:,:,:]).float(),
                         torch.from_numpy(self.label_crop[0,:,:,:]).long())
            
            
            else:
                 self.label_crop = self.label
                 #self.data_crop  = self.data [:, :, cx: cx + self.patches_size[0], cy: cy + self.patches_size[1], cz: cz + self.patches_size[2]]
                 self.data_crop = self.data [:, :,:, :,:]#np.copy(self.data)
                 #print(self.data_crop)
                 if self.ILD == True:
                     chnl_tu = self.channels_to_use[index]
                     chnl_tu = bin(chnl_tu)[2:].zfill(4)
                     for chnl_drp in range(0,4):
                       if int(chnl_tu[chnl_drp]) == 0:                    
                        self.data_crop[:,chnl_drp,:,:,:] = np.copy(self.data_crop[:,chnl_drp,:,:,:])*0     
                 
                     return (torch.from_numpy(self.data_crop[0,:,:,:,:]).float(),
                             (self.label_crop[0,:,:,:]),chnl_tu)
           
                 else:
                     return (torch.from_numpy(self.data_crop[0,:,:,:,:]).float(),
                         (self.label_crop[0,:,:,:]))
            
                



    def __len__(self):
        return len(self.hdf5_list)
