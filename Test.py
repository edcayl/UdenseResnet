# -*- coding: utf-8 -*-
"""
Created on Mon May 23 11:46:19 2022

@author: PC-FONDECYT-1
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 19 16:20:17 2022

@author: PC-FONDECYT-1
"""

from config import *
import time
from loss_func import dice
import SimpleITK as sitk
import tqdm
import glob
import os
import numpy as np
import pathlib
from datetime import datetime
from os import listdir
from Transformacion_img import Obtener_bordes, Quitar_bordes, resize_ski, margenes
import h5py
#op_dir = './iSeg-2019-Testing-labels'




def read_med_image(file_path, dtype):
    #----Transformar imagen------
    img_stk = sitk.ReadImage(file_path)
    img_np = sitk.GetArrayFromImage(img_stk)
    img_np = img_np.astype(dtype)
    return img_np, img_stk

def convert_label(label_img):
    label_processed=np.zeros(label_img.shape[0:]).astype(np.uint8)
    for i in range(label_img.shape[2]):
        label_slice=label_img[:, :, i]
        label_slice[label_slice == 1 ] = 1# 10] = 1
        label_slice[label_slice == 2 ] = 2 #150] = 2
        label_slice[label_slice == 4 ] = 3#250] = 3
        label_processed[:, :, i]=label_slice
    return label_processed

def convert_label_submit(label_img):
    label_processed=np.zeros(label_img.shape[0:]).astype(np.uint8)
    for i in range(label_img.shape[2]):
        label_slice=label_img[:, :, i]
        label_slice[label_slice == 1] = 1#10
        label_slice[label_slice == 2] = 2#150
        label_slice[label_slice == 3] = 4#250
        label_processed[:, :, i]=label_slice
    return label_processed

def get_seg(net, op_dir, chnl_tu):

    onlyfiles = listdir(test_path)
    test_subj = []
    for n in onlyfiles:
        if n[-3:] == ".h5":
            test_subj.append(n)    
            
       
    for subj in tqdm.tqdm(test_subj,disable = silencetqdm):
        subj_id = config_db.get_subj_id(subj,instance ="test")
        print(subj_id)
        
        
        time_start = time.perf_counter()
        
        #---get original image spacing--------------------------------------
        trainset_image = config_db.raw_data_location(raw_data,subj_id,"t1ce")
        print(trainset_image)
        if os.path.exists(trainset_image):
            sitk_T1 = sitk.ReadImage(trainset_image)
            #sitk_T1 = sitk.DICOMOrient(sitk_T1, '%s' %img_or)  #reorientar imagen 
            spacing = sitk_T1.GetSpacing()
            print(spacing)
            
        else:
            spacing = config_db.spacing()
        #-------------------------------------------------------------------
        

                 
        img_h5 = h5py.File("%s/%s" %(test_path,subj),"r")
        t1_h5 =  img_h5["data"][0,0,:,:,:]
        t1_h5 = np.transpose(t1_h5, (2,1,0))
        t2_h5 = img_h5["data"][0,1,:,:,:]
        t2_h5 = np.transpose(t2_h5, (2,1,0))
        t1ce_h5 = img_h5["data"][0,2,:,:,:]
        t1ce_h5 = np.transpose(t1ce_h5, (2,1,0))
        flair_h5 = img_h5["data"][0,3,:,:,:]
        flair_h5 = np.transpose(flair_h5, (2,1,0))
        #seg_h5 = img_h5["label"][0,0,:,:,:]
        #seg_h5 = np.transpose(seg_h5, (0,2,1))
        img_h5.close()
               
        input1 = t1_h5[:, :, :, None]
        input2 = t2_h5[:, :, :, None]
        input3 = t1ce_h5[:, :, :, None]
        input4 = flair_h5[:, :, :, None]
        print(np.shape(t1_h5))
                
        chnl_tu = str(chnl_tu)
        
        if int(chnl_tu[0]) == 0:
            input1 = np.copy(input1)*0
        if int(chnl_tu[1]) == 0:
            input2 = np.copy(input2)*0
        if int(chnl_tu[2]) == 0:
            input3 = np.copy(input3)*0
        if int(chnl_tu[3]) == 0:
            input4 = np.copy(input4)*0
        

        inputs = np.concatenate((input1, input2, input3, input4), axis=3)
        #print(np.unique(inputs))
        inputs = inputs[None, :, :, :, :]
        image = inputs.transpose(0, 4, 1, 3, 2)               
        image = torch.from_numpy(image).float().to(device)   

        _, _, C, H, W = image.shape
        print(image.shape)
        deep_slices   = margenes(C,patches_size[0],solapado)
        height_slices = margenes(H,patches_size[1],solapado)
        width_slices  = margenes(W,patches_size[2],solapado)
        print(deep_slices)
        print(height_slices)
        print(width_slices)
        whole_pred = np.zeros((1,)+(num_classes,) + image.shape[2:])
        count_used = np.zeros((image.shape[2], image.shape[3], image.shape[4])) + 1e-5

        with torch.no_grad():             
                       
            for i in range(len(deep_slices)):
                for j in range(len(height_slices)):
                    for k in range(len(width_slices)):
                        deep = deep_slices[i]
                        height = height_slices[j]
                        width = width_slices[k]
                        image_crop = image[:, :, deep   : deep   + patches_size[0],
                                                    height : height +patches_size[1],
                                                    width  : width  + patches_size[2]]
                        

                        outputs = net(image_crop)
                        whole_pred[slice(None), slice(None), deep: deep + patches_size[0],
                                    height: height + patches_size[1],
                                    width: width + patches_size[2]] += outputs.data.cpu().numpy()

                        count_used[deep: deep + patches_size[0],
                                    height: height + patches_size[1],
                                    width: width + patches_size[2]] += 1
            
            
        
        
        whole_pred = whole_pred[0, :, :, :, :]
        print(np.unique(whole_pred))
        whole_pred = np.argmax(whole_pred, axis=0)
        print(np.unique(whole_pred))        
        whole_pred = whole_pred.transpose(0,2,1)
        
        time_elapsed = (time.perf_counter() - time_start)
        f= open("%s/output_time.txt" %(out_dir),"a+")
        f.write("subject_id / time_elapsed \n")
        f.write("Subject_%s %.16f\n" % (subj_id, time_elapsed))
        print("Subject_%s %.16f\n" % (subj_id, time_elapsed))

        f_pred = os.path.join( op_dir, "subject-%s_label.nii"  % subj_id )
        whole_pred = (t2_h5 != 0) * whole_pred
        print(np.shape(whole_pred))
        whole_pred = convert_label_submit(whole_pred)
        print(np.unique(whole_pred))
        whole_pred_itk = sitk.GetImageFromArray(whole_pred.astype(np.uint8))
        whole_pred_itk.SetSpacing(spacing)
        #whole_pred_itk = sitk.DICOMOrient(whole_pred_itk, "PLS") #por defecto LPS
        #whole_pred_itk = sitk.DICOMOrient(whole_pred_itk, "SPL")
        whole_pred_itk = sitk.DICOMOrient(whole_pred_itk, "PSL")
        whole_pred_itk.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        whole_pred_itk.SetOrigin((0,-230,0))
        sitk.WriteImage(whole_pred_itk, f_pred)

if __name__ == '__main__':
    from global_config import init_parser, db_config
    
    hora = datetime.now()
    mes = hora.strftime(("%m-%Y"))
    hora = hora.strftime(("%d-%m-%Y_%H-%M"))

    parser_config = init_parser()
    #args = parser_config.parse_args(["-ts", 0, "-vs", 0, "-db" ,"HCvBv2" ])
    args = parser_config.parse_args()
    init_config_db = db_config(args.database,args.framework)
    config_db = init_config_db.db
    
    if args.framework == "cluster":
        silencetqdm = True
    else:
        silencetqdm = False

    raw_data 	= config_db.dir_path("raw_data")              
    test_path  = config_db.dir_path("test_dir")
    out_path = config_db.dir_path("results_dir")
    out_subdir = config_db.out_subdir("test",args.test_model,args.database)
    out_dir = "%s/%s/%s/Imgout_%s" %(out_path,out_subdir,mes,hora)
    
    patches_size = config_db.train_config("patches_size")
    solapado = config_db.train_config("solapado_test")
    num_classes = config_db.train_config("num_classes")

    if not os.path.exists(out_dir):
         os.makedirs(out_dir)
    
    img_or = config_db.image_orientation()
    img_in_size = config_db.input_size()
    
    
    model_checkpoint = config_db.trained_model(args.test_model,out_mode="checkpoint",model_state = args.model_checkpoint)
    f= open("%s/output_time.txt" %(out_dir),"a+")
    f.write("Model: %s \n" % (model_checkpoint))
    f.close()
    print(model_checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = DenseResNet(num_init_features=32, growth_rate=16, block_config=(4, 4, 4, 4),num_classes=4).to(device)
    
    saved_state_dict = torch.load( '%s' %model_checkpoint)  
    net.load_state_dict(saved_state_dict)
    net.eval()
    

    if args.ILD:
      for kpt_channel in range(1,16):
       channels_to_use = bin(kpt_channel)[2:].zfill(4)
       save_dir = "%s/w_chnl%s" %(out_dir,channels_to_use)
       if not os.path.exists(save_dir):
         os.makedirs(save_dir)

       d = get_seg(net,save_dir,channels_to_use)

    else: 
     d = get_seg(net,out_dir,1111)
     
    f_log = open("net_log.txt", "a+")
    f_log.write("%s \t ,test \n" %hora)
    f_log.close()
  
