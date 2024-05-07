import numpy as np
import os
import glob
import h5py
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from skimage.transform import resize
import SimpleITK as sitk
import tqdm

#from config import dlt_bgr, rs_ds

#agregar rutas al archivo de configuracion y reemplazar
hora = datetime.now()
hora = hora.strftime(("%d-%m-%Y_%H-%M-%S"))
os.makedirs("D:/EduardoCavieres/Resultados/prep_datos_%s" %hora)
archivo = "D:/EduardoCavieres/Resultados/prep_datos_%s/dataset_%s.txt" %(hora,hora)
f_out = open(archivo, "a+")
#f_out.write("nro sujetos:  %s  nro sujetos entrenamiento: %s nro sujetos validacion: %s  nro sujetos test: %s \n" %(nrosujetos, s_train, s_validacion, s_test))
f_salida = open("D:/EduardoCavieres/Resultados/prep_datos_%s/out_%s.txt" %(hora,hora),"a+")
f_mask = open("D:/EduardoCavieres/Resultados/prep_datos_%s/mask_size%s.txt" %(hora,hora),"a+")
#---------------------------------------------------------------------------------
def convert_label(label_img):

    label_processed = np.where(label_img==1, 1, label_img)
    label_processed = np.where(label_processed==2, 2, label_processed)
    label_processed = np.where(label_processed==4, 3, label_processed)
    return label_processed


def masking(IMref,IM):  
    '''
    determina en que slices la imagen esta vacia
    para poder eliminar el fondo
    '''
    
    out_y = (np.sum(IMref,axis=0)).astype(int)
    out_y = (np.sum(out_y,axis=1) == 0).astype(int)
    img_enmascarada = np.delete(IM, np.argwhere(out_y==1),1)
    
    
    out_x = (np.sum(IMref,axis=1)).astype(int)
    out_x = (np.sum(out_x,axis=0) == 0).astype(int)
    img_enmascarada = np.delete(img_enmascarada, np.argwhere(out_x==1),2)
        
    out_z = (np.sum(IMref,axis=2)).astype(int)
    out_z = (np.sum(out_z,axis=1) == 0).astype(int)
    img_enmascarada = np.delete(img_enmascarada, np.argwhere(out_z==1),0)
   
    return(out_x,out_y,out_z)
    
def padding(input_image,out_size,pad_value=None):
    '''
    cambia la dimension de la imagen utilizando un padding, con la informacion en el centro    
    '''
    if pad_value == None:
        pad_val = input_image[0,0,0] #valor de fondo
    else: 
        pad_val = pad_value    
        
    padding = np.array(out_size)-np.shape(input_image)  #diferencia de tamaÃ±o entre imagen de entrada e imagen objetivo
    #print(padding)
    #print(out_size)
    #if np.shape(input_image)[0] > 22:
        #print(sb_id)  
    #    print(np.shape(input_image))
    pad_image = np.pad(input_image[:,:,:], pad_width=[(padding[0]//2, padding[0] - padding[0]//2),(padding[1]//2, padding[1] - padding[1]//2),(padding[2]//2, padding[2] - padding[2]//2)], mode='constant', constant_values=pad_val)
    
    return pad_image    
    
        

class data_split:
    def __init__(self,train_size = None, val_size = 59, test_size = 74, random_state=42):
        if not train_size == None:
          if "." in train_size:
            self.train_size = float(train_size)
          else:
            self.train_size = int(train_size)
        
        if not val_size == None:
          if "." in val_size:
            self.val_size = float(val_size)
          else:
            self.val_size = int(val_size) 
        
        if not test_size == None:
          if "." in test_size:
            self.test_size = float(test_size)
          else:
            self.test_size = int(test_size)

        self.random_state = int(random_state)
            
            
            
    def main(self,X):
        if self.train_size != 0 and self.val_size != 0 and self.test_size != 0:
            #print("train: %s val:: %s test: %s" %(self.train_size,self.val_size,self.test_size))
            #print(type(self.train_size))
            #print(type(self.val_size))
            #print(type(self.test_size))
            X_train, X_test = train_test_split(X, test_size= self.test_size, random_state=self.random_state)
            x_train, x_val, y_train, y_val = train_test_split(X_train, train_size=self.train_size ,test_size=self.val_size, random_state=self.random_state)   
            x_train = np.array(x_train)
            x_val = np.array(x_val)
            x_test = np.array(X_test)
            
        elif (self.train_size != 0 and self.val_size != 0) or (self.train_size != 0 and self.test_size != 0) or (self.test_size != 0 and self.val_size != 0):
            if (self.train_size != 0 and self.val_size != 0):
              X_train, X_val = train_test_split(X, train_size=self.train_size, test_size=self.val_size, random_state=self.random_state)
              x_train = np.array(X_train)
              x_val = np.array(X_val)
              x_test = np.array([None])
              
            elif (self.train_size != 0 and self.test_size != 0):
              X_train, X_test= train_test_split(X, train_size=self.train_size, test_size=self.test_size, random_state=self.random_state)
              x_train = np.array(X_train)
              x_val = np.array([None])
              x_test = np.array(X_test)
            
            elif (self.test_size != 0 and self.val_size != 0):
              X_val, X_test = train_test_split(X, train_size= self.val_size ,test_size=self.test_size, random_state=self.random_state)  
              x_train = np.array([None])
              x_val = np.array(x_val)
              x_test = np.array(X_test)
              
        elif self.train_size == 0 and self.val_size == 0 and self.test_size == 0:
            pass 
        
        else: 
            if self.train_size != 0:
              x_train = np.array(X)
              x_val = np.array([None])
              x_test = np.array([None])
            elif self.val_size != 0:
              x_val = np.array(X)
              x_train = np.array([None])
              x_test = np.array([None])  
            elif self.test_size != 0:
              x_test = np.array(X)
              x_val = np.array([None])
              x_train = np.array([None])  
             
              
              
              
                
        return(x_train,x_val,x_test)


    def excel_import(self,excel_path):
        data_info = pd.read_excel('/home4/eduardo.cavieres/UdenseResnet/Separacionsujetos.xlsx',skiprows=1,usecols="A,B,D")
        X = data_info["Sujeto"]
        Y = data_info["Izq"]
        if self.train_size != 0 and self.val_size != 0 and self.test_size != 0:
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= self.test_size, random_state=self.random_state,stratify=Y)
            x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, train_size=self.train_size ,test_size=self.val_size, random_state=self.random_state,stratify=Y_train)
            x_train = np.array(x_train)
            x_val = np.array(x_val)
            x_test = np.array(X_test)
    
            
        elif (self.train_size != 0 and self.val_size != 0) or (self.train_size != 0 and self.test_size != 0) or (self.test_size != 0 and self.val_size != 0):
            if (self.train_size != 0 and self.val_size != 0) :
                X_train, X_val, Y_train, Y_val = train_test_split(X, Y,train_size=self.train_size, test_size=self.val_size, random_state=self.random_state,stratify=Y)
                x_train = np.array(x_train)
                x_val = np.array(x_val)
                x_test = np.array(X_test)
                
            elif (self.train_size != 0 and self.test_size != 0):
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=self.train_size, test_size=self.test_size, random_state=self.random_state,stratify=Y)
                x_train = np.array(x_train)
                x_val = np.array(x_val)
                x_test = np.array(X_test)
                
            elif (self.test_size != 0 and self.val_size != 0):
                X_val, X_test, Y_val, Y_test = train_test_split(X, Y, train_size= self.val_size ,test_size=self.test_size, random_state=self.random_state,stratify=Y)
                x_train = np.array(x_train)
                x_val = np.array(x_val)
                x_test = np.array(X_test)
  
            
        elif self.train_size == 0 and self.val_size == 0 and self.test_size == 0:
            pass 
        else:  
            pass
        return(x_train,x_val,x_test)

    
    def load_previous(self,txt_path):
        file = open(txt_path)
        text = file.readlines()
        x_train = []
        x_test = []
        x_val = []
        toggle = "off"
        write_subj = False
        for i in text:
            
            if "entrenamiento: [" in i:
                toggle = "train"
                #print(i)
            elif "validacion: [" in i:
                toggle = "val"
                #print(i)
            elif "test: [" in i:
                toggle = "test"
                #print(i)

            if toggle == "train":
                
                subjs = i.split()
                #break
                for subj in subjs:
                    if "]" in subj:
                        write_subj = False
                        x_train.append((subj.strip("]").zfill(3)))
                    if write_subj:
                        x_train.append(subj.zfill(3))
                    if "[" in subj:
                        write_subj = True
                        x_train.append(subj.strip("[").zfill(3))
                    
            if toggle == "test":
                
                subjs = i.split()
                for subj in subjs:
                    if "]" in subj:
                        write_subj = False
                        x_test.append(subj.strip("]").zfill(3))
                    if write_subj:
                        x_test.append(subj.zfill(3))
                    if "[" in subj:
                        write_subj = True
                        x_test.append(subj.strip("[").zfill(3))            


            if toggle == "val":
                #break
                
                subjs = i.split()
                for subj in subjs:
                    if "]" in subj:
                        write_subj = False
                        x_val.append(subj.strip("]").zfill(3))

                    if write_subj:
                        x_val.append(subj.zfill(3))
                    if "[" in subj:
                        write_subj = True
                        x_val.append(subj.strip("[").zfill(3))
                  

        print("train:")
        print(x_train)
        print("val:")
        print(x_val)
        print("test")
        print(x_test)

        return(x_train,x_val,x_test)


class preprocessing():
    def __init__(self):
        pass




def build_h5_dataset(training_size,validation_size,test_size,splitmode):
    '''
    Build HDF5 Image Dataset.
    '''
    #---eliminar archivos antiguos----------------------
    if not (training_size == 0 or training_size == "0"):
        filelist = glob.glob(os.path.join(train_path,"*"))
        for f in filelist:
            os.remove(f)
    if not (test_size == 0 or test_size == "0"):        
        filelist = glob.glob(os.path.join(test_path,"*"))	
        for f in filelist:
            os.remove(f)
    if not (validation_size == 0 or validation_size =="0"):				
        filelist = glob.glob(os.path.join(val_path,"*"))	
        for f in filelist:
            os.remove(f)
    #-------------------------------------------------

    #--listar sujetos en base de datos---------------
    db_subjects = os.listdir(raw_data) 
    #db_subjects = db_subjects.sort()
    imagenes = config_db.sequences()

    #print(raw_data)    
    #recuperar id de sujetos 
    sujects_id = []
    for sujeto in db_subjects:
        subj_id = config_db.get_subj_id(sujeto)
        sujects_id.append(subj_id)


    #------cargar separador de datos--------------------   
    
    data_splitter = data_split(train_size = training_size, val_size = validation_size, test_size = test_size, random_state=42)
    split_mode = splitmode
    if split_mode == "main":
            x_train, x_val, x_test = data_splitter.main(sujects_id)
    elif split_mode == "load_excel":
            x_train, x_val, x_test = data_splitter.excel_import
    elif split_mode == "load_txt":
            x_train, x_val, x_test = data_splitter.load_previous("/home4/eduardo.cavieres/UdenseResnet/split_subj.txt")
    else:
            x_train, x_val, x_test = data_splitter.load_previous
    
    

    for db_subject in tqdm.tqdm(db_subjects,disable = silencetqdm):

    # clasificar sujeto segun grupo asignado
      
      sb_id = config_db.get_subj_id(db_subject) 
      #print(sb_id)  
      if sb_id in x_train:
        grupo = "train"  
        target_path = train_path
      elif sb_id in x_val:
        target_path = val_path
        grupo = "val" 
      elif sb_id in x_test: 
        target_path = test_path
        grupo = "test" 
      
        
      #----abrir carpeta de sujetos y cargar imagenes-------
      subject_images = config_db.sequences()
      img_in_size = config_db.input_size()
      if (os.path.exists(config_db.raw_data_location(raw_data,sb_id,"t2")))  and (os.path.exists(config_db.raw_data_location(raw_data,sb_id,"t1ce"))):
          imagenes = [0,1,1,0,]
          dummy_img = sitk.ReadImage(config_db.raw_data_location(raw_data,sb_id,"t2"))
          img_in_size = np.shape(sitk.GetArrayFromImage(dummy_img))
      else: 
          if (os.path.exists(config_db.raw_data_location(raw_data,sb_id,"t2"))):
              imagenes = [0,1,0,0]
              dummy_img = sitk.ReadImage(config_db.raw_data_location(raw_data,sb_id,"t2"))
              img_in_size = np.shape(sitk.GetArrayFromImage(dummy_img))
          else:
              imagenes = [0,0,1,0]
              dummy_img = sitk.ReadImage(config_db.raw_data_location(raw_data,sb_id,"t1ce"))
              img_in_size = np.shape(sitk.GetArrayFromImage(dummy_img))




      for subdir in subject_images: 
        img_or = config_db.image_orientation() #cargar configuracion de orientacion de la imagen
#---------cargar imagenes--------------------------------------------------
        if imagenes[0] == 1 :  
            f_T1 = config_db.raw_data_location(raw_data,sb_id,"t1")
            sitk_T1 = sitk.ReadImage(f_T1)
            #sitk.DICOMOrient(sitk_T1, '%s' %img_or)  #reorientar imagen 
            img_T1 = sitk.GetArrayFromImage(sitk_T1)
            img_in_size = np.shape(img_T1)
        else:
            img_T1 = np.zeros((img_in_size))
        if imagenes[1] == 1: 
            f_T2 = config_db.raw_data_location(raw_data,sb_id,"t2")
            sitk_T2 = sitk.ReadImage(f_T2)
            #sitk.DICOMOrient(sitk_T2, '%s' %img_or)  #reorientar imagen
            img_T2 = sitk.GetArrayFromImage(sitk_T2)
        else:
            img_T2 = np.zeros((img_in_size))  
        if imagenes[2] == 1: 
            f_T1ce = config_db.raw_data_location(raw_data,sb_id,"t1ce")
            sitk_T1ce = sitk.ReadImage(f_T1ce)  
            #sitk.DICOMOrient(sitk_T1ce, '%s' %img_or)  #reorientar imagen
            img_T1ce = sitk.GetArrayFromImage(sitk_T1ce)
        else:
            img_T1ce = np.zeros((img_in_size))    
        if imagenes[3] == 1:
            f_Flair = config_db.raw_data_location(raw_data,sb_id,"flair")
            sitk_Flair = sitk.ReadImage(f_Flair)
            #sitk.DICOMOrient(sitk_Flair, '%s' %img_or)  #reorientar imagen
            img_Flair = sitk.GetArrayFromImage(sitk_Flair)
        else:
            img_Flair = np.zeros((img_in_size))    
        if config_db.label_avb() == True:
            f_l = config_db.raw_data_location(raw_data,sb_id,"seg")
            sitk_label = sitk.ReadImage(f_l)
            #sitk.DICOMOrient(sitk_label, '%s' %img_or)  #reorientar imagen
            labels = sitk.GetArrayFromImage(sitk_label)
        else:
            labels = np.zeros((img_in_size))

#---------preprocesamiento------------------------------------------------------------------

####################################################
 
       	inputs_T1 = img_T1.astype(np.float32)
       	inputs_T2 = img_T2.astype(np.float32)
       	inputs_T1ce = img_T1ce.astype(np.float32)
       	inputs_Flair = img_Flair.astype(np.float32)
        labels = labels.astype(np.uint8)
        labels=convert_label(labels)
        mask=inputs_T1>0#labels>0
        # Normalization
        if imagenes[0] == 1:
            mask=inputs_T1>0
            inputs_T1_norm = (inputs_T1 - inputs_T1[mask].mean()) / inputs_T1[mask].std()
        else:
            inputs_tmp_T1 = inputs_T1
        if imagenes[1] == 1:
            mask=inputs_T2>0
            inputs_T2_norm = (inputs_T2 - inputs_T2[mask].mean()) / inputs_T2[mask].std()
        else:
            inputs_tmp_T2 = inputs_T2            
        if imagenes[2] == 1:
            mask=inputs_T1ce>0
            inputs_T1ce_norm = (inputs_T1ce - inputs_T1ce[mask].mean()) / inputs_T1ce[mask].std()
        else:
            inputs_tmp_T1ce = inputs_T1ce
        if imagenes[3] == 1:
            mask=inputs_Flair>0
            inputs_Flair_norm = (inputs_Flair - inputs_Flair[mask].mean()) / inputs_Flair[mask].std()
        else:
            inputs_tmp_Flair = inputs_Flair  
           
       
        #enmascarar
        
    # transformaciones dimensionales    
    #########################
        if config_db.image_transformation("dlt_bgr"):                 #eliminar fondo
        
                if imagenes[0] == 1:
                    out_x_t1 ,out_y_t1 ,out_z_t1 = masking(img_T1, img_T1)
                   
                    inputs_tmp_T1 = np.delete(inputs_T1_norm, np.argwhere(out_y_t1==1),1)
                    inputs_tmp_T1 = np.delete(inputs_tmp_T1, np.argwhere(out_x_t1==1),2)
                    inputs_tmp_T1 = np.delete(inputs_tmp_T1, np.argwhere(out_z_t1==1),0) 
                    shape_mask = inputs_tmp_T1.shape
                    
                if imagenes[1] == 1:
                    out_x_t1 ,out_y_t1 ,out_z_t1 = masking(img_T2, img_T2)
                    inputs_tmp_T2 = np.delete(inputs_T2_norm, np.argwhere(out_y_t1==1),1)
                    inputs_tmp_T2 = np.delete(inputs_tmp_T2, np.argwhere(out_x_t1==1),2)
                    inputs_tmp_T2 = np.delete(inputs_tmp_T2, np.argwhere(out_z_t1==1),0) 
                
                if imagenes[2] == 1:
                    out_x_t1 ,out_y_t1 ,out_z_t1 = masking(img_T1ce, img_T1ce)
                    inputs_tmp_T1ce = np.delete(inputs_T1ce_norm, np.argwhere(out_y_t1==1),1)
                    inputs_tmp_T1ce = np.delete(inputs_tmp_T1ce, np.argwhere(out_x_t1==1),2)
                    inputs_tmp_T1ce = np.delete(inputs_tmp_T1ce, np.argwhere(out_z_t1==1),0)

                if imagenes[3] == 1:         
                    out_x_t1 ,out_y_t1 ,out_z_t1 = masking(img_Flair, img_Flair)
                    inputs_tmp_Flair = np.delete(inputs_Flair_norm, np.argwhere(out_y_t1==1),1)
                    inputs_tmp_Flair = np.delete(inputs_tmp_Flair, np.argwhere(out_x_t1==1),2)
                    inputs_tmp_Flair = np.delete(inputs_tmp_Flair, np.argwhere(out_z_t1==1),0)
                if config_db.label_avb() == True:
                    labels_tmp = np.delete(labels, np.argwhere(out_y_t1==1),1)
                    labels_tmp = np.delete(labels_tmp, np.argwhere(out_x_t1==1),2)
                    labels_tmp = np.delete(labels_tmp, np.argwhere(out_z_t1==1),0)     
                else:
                    labels_tmp = labels    
        
        else:
                if imagenes[0]==1:
                    inputs_tmp_T1 = inputs_T1_norm
                if imagenes[1] == 1:                
                    inputs_tmp_T2 = inputs_T2_norm
                if imagenes[2] == 1:
                    inputs_tmp_T1ce = inputs_T1ce_norm
                if imagenes[3] == 1:                
                    inputs_tmp_Flair = inputs_Flair_norm
                if config_db.label_avb() == True:
                    labels_tmp = labels
                
        
        
        
        if config_db.image_transformation("padding"): 
      #  pad_val = -3.7915964  #valor con el que se rellenan los bordes (ver que sea el mismo valor que el del fondo)
          if imagenes[0] == 1: 
            inputs_tmp_T1 = padding(inputs_tmp_T1,img_in_size)
          if imagenes[1] == 1:   
            inputs_tmp_T2 = padding(inputs_tmp_T2,img_in_size)
          if imagenes[2] == 1:   
            inputs_tmp_T1ce = padding(inputs_tmp_T1ce,img_in_size)
          if imagenes[3] == 1:   
            inputs_tmp_Flair = padding(inputs_tmp_Flair,img_in_size)
          if config_db.label_avb() == True: 
            labels_tmp = padding(labels_tmp,img_in_size)
    
       
        
        
       ##############resize#################
        if config_db.image_transformation("rs_ds"):
            if imagenes[0] == 1: 
                inputs_tmp_T1 = resize(inputs_tmp_T1, img_in_size, order=1, mode='reflect', cval=0, clip=True, preserve_range=True, anti_aliasing=False, anti_aliasing_sigma=False)
            if imagenes[1] == 1: 
                inputs_tmp_T2 = resize(inputs_tmp_T2, img_in_size, order=1, mode='reflect', cval=0, clip=True, preserve_range=True, anti_aliasing=False, anti_aliasing_sigma=False)
       	    if imagenes[2] == 1: 
                   inputs_tmp_T1ce = resize(inputs_tmp_T1ce, img_in_size, order=1, mode='reflect', cval=0, clip=True, preserve_range=True, anti_aliasing=False, anti_aliasing_sigma=False)
            if imagenes[3] == 1: 
                inputs_tmp_Flair = resize(inputs_tmp_Flair, img_in_size, order=1, mode='reflect', cval=0, clip=True, preserve_range=True, anti_aliasing=False, anti_aliasing_sigma=False)
            if config_db.label_avb() == True:
                labels_tmp = resize( labels_tmp, img_in_size, order=0, mode='reflect', cval=0, clip=True, preserve_range=True, anti_aliasing=False, anti_aliasing_sigma=False)
    


        if config_db.image_transformation("extract_slices"):

            slice_set = np.arange(7,155,7)
            inputs_tmp_T1 = inputs_tmp_T1[slice_set]
            inputs_tmp_T2 = inputs_tmp_T2[slice_set]
            inputs_tmp_T1ce = inputs_tmp_T1ce[slice_set]
            inputs_tmp_Flair =  inputs_tmp_Flair[slice_set]
            if config_db.label_avb() == True:
                labels_tmp = labels_tmp[slice_set]

        if config_db.image_transformation("merge_slices"):

            slice_set = np.arange(0,145,5)
            mask_set = np.arange(2,145,5)
            T1_merge = np.copy(inputs_tmp_T1)[slice_set]
            T2_merge = np.copy(inputs_tmp_T2)[slice_set]
            T1ce_merge = np.copy(inputs_tmp_T1ce)[slice_set]
            FLAIR_merge = np.copy(inputs_tmp_Flair)[slice_set]
            
            for s_n, slice_n in enumerate(slice_set):
                T1_merge[s_n] = np.mean(np.copy(inputs_tmp_T1)[slice_n:slice_n+5],axis=0)
                T2_merge[s_n] = np.mean(np.copy(inputs_tmp_T2)[slice_n:slice_n+5],axis=0)
                T1ce_merge[s_n] = np.mean(np.copy(inputs_tmp_T1ce)[slice_n:slice_n+5],axis=0)
                FLAIR_merge[s_n] = np.mean(np.copy(inputs_tmp_Flair)[slice_n:slice_n+5],axis=0)
                


            inputs_tmp_T1 = T1_merge
            inputs_tmp_T2 = T2_merge
            inputs_tmp_T1ce = T1ce_merge
            inputs_tmp_Flair =  FLAIR_merge
            if config_db.label_avb() == True:
                labels_tmp = labels_tmp[mask_set]



#########################    
       
    
    
    
    
        
           
 ##############################################
        
####################crear archivo h5###############################  

        inputs_tmp_T1 = inputs_tmp_T1[:, :, :, None]
        inputs_tmp_T2 = inputs_tmp_T2[:, :, :, None]
        inputs_tmp_T1ce = inputs_tmp_T1ce[:, :, :, None]
        inputs_tmp_Flair =  inputs_tmp_Flair[:, :, :, None]
        if config_db.label_avb() == True:
            labels_tmp =  labels_tmp[:, :, :, None]         
        inputs = np.concatenate((inputs_tmp_T1, inputs_tmp_T2, inputs_tmp_T1ce, inputs_tmp_Flair), axis=3)
        
        
        #print ('Subject:', subject_name, 'Input:', inputs.shape, 'Labels:', labels_tmp.shape, 'group:', group)
        #i_correct = i + 1 
        #f_salida.write('Subject: %s ; Input: %s ; Labels: %s ; group: %s \n' %(subject_name,inputs.shape,labels_tmp.shape,group))
        #f_mask.write('Subject: %s ; shape: %s ; group: %s \n' %(i_correct,shape_mask,group))
        f_salida.flush()
        f_mask.flush()
        
        
        inputs_caffe = inputs[None, :, :, :, :]
        
        inputs_caffe = inputs_caffe.transpose(0,4,1,2,3)#(0, 4, 3, 1, 2)
        if config_db.label_avb() == True:
            labels_caffe = labels_tmp[None, :, :, :, :]
            labels_caffe = labels_caffe.transpose(0,4,1,2,3)#(0, 4, 3, 1, 2)
        
        out_name = config_db.h5_file_name(sb_id,grupo) 
        with h5py.File(os.path.join(target_path, '%s.h5' % (out_name)), 'w') as f:
            f['data'] = inputs_caffe  # c x d x h x w
            if config_db.label_avb() == True:
                f['label'] = labels_caffe
            
    f_salida.close()
    f_out.close() 
    f_mask.close
    
    f_log = open("net_log.txt","a+") 
    f_log.write("%s \t ,generate_h5 \n" %hora)
    f_log.close()
          

if __name__ == '__main__':
    from global_config import init_parser, db_config
    parser_config = init_parser()
    #args = parser_config.parse_args(["-ts", 0, "-vs", 0, "-db" ,"HCvBv2" ])
    args = parser_config.parse_args()
    s_m = args.datasplit
    t_s = args.train_size
    v_s = args.val_size
    test_s = args.test_size
    
    if args.framework == "cluster":
        silencetqdm = True
    else:
        silencetqdm = False
    
    
    init_config_db = db_config(args.database,args.framework)
    config_db = init_config_db.db
    
   

    raw_data 	= config_db.dir_path("raw_data")
    train_path 	= config_db.dir_path("train_dir")                 # Path to save hdf5 data.
    val_path 	= config_db.dir_path("val_dir")                   # Path to save hdf5 data.
    test_path  = config_db.dir_path("test_dir")                   # Path to save hdf5 data.
    
    hora = datetime.now()
    mes = hora.strftime(("%m-%Y"))
    hora = hora.strftime(("%d-%m-%Y_%H-%M"))    

    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(val_path):
        os.makedirs(val_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    
    out_path = config_db.dir_path("results_dir")
    out_subdir = config_db.out_subdir("prep_data",args.test_model,args.database)
    out_dir = "%s/%s/%s/prep_data_%s" %(out_path,out_subdir,mes,hora)
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    archivo = "%s/dataset_%s.txt" %(out_dir,hora)
    f_out = open(archivo, "a+")
    #f_out.write("nro sujetos:  %s  nro sujetos entrenamiento: %s nro sujetos validacion: %s  nro sujetos test: %s \n" %(nrosujetos, s_train, s_validacion, s_test))
    f_salida = open("%s/out_%s.txt" %(out_dir,hora),"a+")
    f_mask = open("%s/mask_size%s.txt" %(out_dir,hora),"a+")


    build_h5_dataset(t_s,v_s,test_s,s_m)

else:
    build_h5_dataset()
