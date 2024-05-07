# -*- coding: utf-8 -*-
"""
Created on Mon May  8 11:48:05 2023

@author: PC-FONDECYT-1
"""

import argparse
import os
overdrive_parse = False

#------argparse--------------------------------------------------------------

def init_parser(): 

    parser = argparse.ArgumentParser(
                        prog='DL Brain Tumor Segmentation',
                        description='What the program does',
                        epilog='Configuraciones mas especificas deben realizarse en el archivo global config')

# define espacio de trabajo 
    parser.add_argument('--framework','-fm', action='store',
                        default="desktop", required=False,
                        help='espacio de trabajo utilizado computador/cluser (desktop por defecto)')

#define que base de datos se utilizara
    parser.add_argument('--database','-db', action='store',
                        required=False, default="BraTS2020",
                        help='base de datos utilizada (Brats 2020 por defecto)) \n BraTS2020 ; BraTS2023 ; HCvB ; HCvBv2')

#define si se utiliza ILD
    parser.add_argument('--ILD',action='store_true',
                        required=False,
                        help='aplicar input layer drop ')
    
    
#----como separar los datos (preparacion de datos)
    parser.add_argument('--datasplit', '-dbsp',action='store', default='main',
                        required=False,
                        help='forma en la que se dividiran los datos \n train ; test ; val ; train_test ; train_val ; val_test ; train_val_test \n ; load_excel ; load_txt')
    
    parser.add_argument('--train_size', '-ts',action='store', default=None,
                        required=False,
                        help='')


    parser.add_argument('--val_size', '-vs',action='store', default=None,
                        required=False,
                        help='')

    parser.add_argument('--test_size', action='store', default=None,
                            required=False,
                            help='')

#-------Entrenamiento-----------------------------------------------------------
    

    parser.add_argument('--train_submodel','-train_sm' , action='store', default="main",
                            required=False,
                            help='')

#-------Test-------------------------------------------------------------------

    parser.add_argument('--test_model', action='store', default="default",
                            required=False,
                            help='Especifica que modelo utilizar para el test, por defecto se utiliza el definido por la base' 
                            ' de datos e ILD. Para utilizar otro modelo se debe escribir la base de datos acompañada por "_ILD'
                            ' si que utiliza el entrenamiento con ILD')
    
    parser.add_argument('--model_checkpoint','-model', action='store', default="best",
                            required=False,
                            help='')


#----define si se debe preparar datos, entrenar o realizar test---------------    
    parser.add_argument('--train' ,action='store_true',
                        required=False,
                        help='entrenar red')
    
    parser.add_argument('--test','-t' ,action='store_true',
                        required=False,
                        help='realizar segmentacion')


    parser.add_argument('--preparedata','-pd' ,action='store_true',
                        required=False,
                        help='Separar datos')

#---------------------------------------------------------------------------
                    


    #args = parser.parse_args()


    return parser 

#---------------------------------------------------------------------

if overdrive_parse == True:
    framework = "desktop" #desktop cluster
    database = "HCvB"     #brats


#---------------global config-------------------------------------------

#-------configuracion base de datos------------------------------------------

class db_config:
    def __init__(self,database,framework):
        
        if database == "BraTS2020" or database == "brats":
            self.db = self.BraTS2020(framework)
        elif database == "hcvb" or database == "HCvB":
            self.db = self.HCvB(framework)
        elif database == "hcvbv2" or database == "HCvBv2":
            self.db = self.HCvBv2(framework)    
        elif database == "BraTS2023" or database == "brats2023":
            self.db = self.BraTS2023(framework)
        if database == "BraTS2020_2d" or database == "brats_2d":
            self.db = self.BraTS2020_2d(framework)
        elif database == "hcvbv3" or database == "HCvBv3":
            self.db = self.HCvBv3(framework)

    
    class BraTS2020:
        def __init__(self,framework):
            #self.db = database
            self.framework = framework
            if framework == "cluster":
                self.main_dir = "/home4/eduardo.cavieres/"
            elif framework == "desktop":
                self.main_dir = "D:/EduardoCavieres/TesisMagister"
            elif framework == "linux":
                self.main_dir = "/home/edo/Documents"

        def raw_data_location(self,raw_data,sujeto,imagen):
            if imagen == "t1":
                rdl = os.path.join("%s/BraTS20_Training_%s/BraTS20_Training_%s_t1" %(raw_data,sujeto,sujeto) + '.nii.gz')
            elif imagen == "t2":
               rdl = os.path.join("%s/BraTS20_Training_%s/BraTS20_Training_%s_t2" %(raw_data,sujeto,sujeto) + '.nii.gz')
            elif imagen == "t1ce":
               rdl = os.path.join("%s/BraTS20_Training_%s/BraTS20_Training_%s_t1ce" %(raw_data,sujeto,sujeto) + '.nii.gz')
            elif imagen == "flair":
               rdl = os.path.join("%s/BraTS20_Training_%s/BraTS_Training_%s_flair" %(raw_data,sujeto,sujeto) + '.nii.gz')
            elif imagen == "seg":
               rdl = os.path.join("%s/BraTS20_Training_%s/BraTS20_Training_%s_seg" %(raw_data,sujeto,sujeto) + '.nii.gz')
            return rdl

                
        def dir_path(self,directorio):
            if self.framework == "cluster":
                if directorio == "raw_data":
                    dir_path = "%s/MICCAI_BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData" %(self.main_dir) 
                elif directorio == "train_dir":
                    dir_path = "%s/data_train" %(self.main_dir)
                elif directorio == "val_dir":
                    dir_path =  "%s/data_val" %(self.main_dir)
                elif directorio == "test_dir":
                    dir_path = "%s/data_test" %(self.main_dir)
                elif directorio == "results_dir":
                    dir_path = "%s/Resultados/UdenseResnet" %(self.main_dir)
                    
            elif self.framework == "desktop":
                if directorio == "raw_data":
                    dir_path = "%s/Bateria_Datos/MICCAI_BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData" %(self.main_dir) 
                elif directorio == "train_dir":
                    dir_path = "%s/PrepData/BraTS_2020/data_train" %(self.main_dir)
                elif directorio == "val_dir":
                    dir_path =  "%s/PrepData/BraTS_2020/data_val" %(self.main_dir)
                elif directorio == "test_dir":
                    dir_path = "%s/PrepData/BraTS_2020/data_val" %(self.main_dir)
                elif directorio == "results_dir":
                    dir_path = "%s/Resultados/UdenseResnet" %(self.main_dir)
                    
            return dir_path


        def out_subdir(self,instance,trained_model,database):

            if trained_model == "default":
                train_db = "BraTS2020"
                submodel = "Main"

            else:
                if "-" in trained_model or "_" in trained_model:
                    train_db = trained_model.split("_")[0]
                    if len(trained_model.split("_")) == 2:
                        submodel = trained_model.split("_")[1]
                    else:
                        submodel = "_".join(trained_models.split("_")[1:])
                else:
                    train_db = trained_model
                    submodel = "Main"

                if train_db == "BraTS2020" or train_db == "brats" or train_db == "brats2020":
                    train_db = "BraTS2020"
                elif train_db == "BraTS2023" or train_db == "brats2023":
                    train_db = "BraTS2023"


            if instance == "train":
                dir_path = "UdenseResnet/%s/%s/%s" %(submodel,train_db,instance)
            elif instance == "test":
                dir_path = "UdenseResnet/%s/%s/%s/%s" %(submodel,train_db,instance,database)
            return dir_path

    #-------image-info------------------------------------------------------
        def sequences(self):
            seq = [1,1,1,1]  #secuencias disponibles en base de datos, en este orden T1,T2,T1ce,Flair
            return seq

        def label_avb(self): #si existe segmentacion disponible
            lbl = True
            return lbl

        def image_orientation(self):
            orientation = "RAI"
            return orientation

        def input_size(self):
            img_sz = (155,240,240)
            return img_sz

        def spacing(self):
            img_spc = (1, 1, 1)
            return img_spc

    #------preprocessing--------------------------------------
        def image_transformation(self, op):
            dlt_bgr = False #
            rs_ds = False #
            padding = False #

            if op == "dlt_bgr":
                return dlt_bgr
            elif op == "resize":
                return rs_ds
            elif op == "padding":
                return padding
    #------------------------------------------------------------------
        def get_subj_id(self,sujeto,instance="generate_db"):
            if instance =="generate_db":
                subj_id = sujeto.split("_")[2]
            elif instance == "train":
                subj_id = sujeto[12:3]
            elif instance =="test":
                subj_id = sujeto[12:-3]

            return subj_id

        def h5_file_name(self,sujeto,grupo):

            h5fn = "%s_BraTS2020_%s" %(grupo,sujeto)
            return h5fn

        def trained_model(self,model,out_mode="model_path",model_state="best"):
            if model == "default":
                train_db = "BraTS2020"
                submodel = "Main"

            else:
                if "-" in model or "_" in model:
                    modelo = model.split("_")[0]
                    if len(model.split("_"))==2:
                       submodel = model.split("_")[1]
                    else:
                       submodel = "_".join(model.split("_")[1:])
                else:
                    modelo = model.split("-")[0]
                    submodel = "Main"

                if modelo == "BraTS2020" or modelo == "brats" or modelo == "brats2020":
                    train_db = "BraTS2020"
                if modelo == "BraTS2023" or modelo == "brats2023":
                    train_db = "BraTS2023"

            if out_mode == "model_path":
                out = "%s/Trained_Models/UdenseResnet/%s/%s" %(self.main_dir,submodel,train_db)
            if out_mode == "checkpoint":
                if model_state == "best":
                    modelstate = "best_model"
                else:
                    modelstate = str(model).zfill(5)


                out = "%s/Trained_Models/UdenseResnet/%s/%s/%s.pth" %(self.main_dir,submodel,train_db,modelstate)

            return out        


        def train_config(self,config):
            if self.framework == "desktop":
                if config == "loss_weights":
                    config_value = [1,1.5,1,2]
                elif config == "num_classes":
                    config_value = 4
                elif config == "num_epoch":
                    config_value = 300
                elif config == "lr_init":
                    config_value = 2e-4
                elif config == "lr_stepsize":
                    config_value = 30
                elif config == "batch_train":
                    config_value = 2
                elif config == "patches_size":
                    config_value = (128,128,128)
                elif config == "solapado":
                    config_value = 16
                elif config == "whole_val":
                    config_value = True

            elif self.framework == "cluster":
                if config == "loss_weights":
                    config_value = [1,1.5,1,1]
                elif config == "num_classes":
                    config_value = 4
                elif config == "num_epoch":
                    config_value = 300
                elif config == "lr_init":
                    config_value = 2e-4
                elif config == "lr_stepsize":
                    config_value = 30
                elif config == "batch_train":
                    config_value = 8
                elif config == "patches_size":
                    config_value = (128,128,128)
                elif config == "solapado":
                    config_value = 100
                elif config == "solapado_val":
                    config_value = 1
                elif config == "solapado_test":
                    config_value = 100
                elif config == "whole_val":
                    config_value = True


            return config_value

    class BraTS2023:
        def __init__(self,framework):
            self.framework = framework
            if framework == "cluster":
                self.main_dir = "/home4/eduardo.cavieres"
            elif framework == "desktop":
                self.main_dir = "D:/EduardoCavieres"
            elif framework == "linux":
                self.main_dir = "/home/edo/Documents"
    #-------------- path config-------------------------------------
        def raw_data_location(self,raw_data,sujeto,imagen):
            if imagen == "t1":
                rdl = os.path.join("%s/BraTS-GLI-%s/BraTS-GLI-%s-t1n" %(raw_data,sujeto,sujeto) + '.nii.gz') 
            elif imagen == "t2":   
               rdl = os.path.join("%s/BraTS-GLI-%s/BraTS-GLI-%s-t2w" %(raw_data,sujeto,sujeto) + '.nii.gz') 
            elif imagen == "t1ce":   
               rdl = os.path.join("%s/BraTS-GLI-%s/BraTS-GLI-%s-t1c" %(raw_data,sujeto,sujeto) + '.nii.gz')                
            elif imagen == "flair":   
               rdl = os.path.join("%s/BraTS-GLI-%s/BraTS-GLI-%s-t2f" %(raw_data,sujeto,sujeto) + '.nii.gz') 
            elif imagen == "seg":   
               rdl = os.path.join("%s/BraTS-GLI-%s/BraTS-GLI-%s-seg" %(raw_data,sujeto,sujeto) + '.nii.gz') 
            return rdl          
                
        def dir_path(self,directorio):
            if self.framework == "cluster":
                if directorio == "raw_data":
                    #dir_path = "%s/Bateria_Datos/BraTS2023/BraTS2023_ValidationData" %(self.main_dir) #TrainingData" %(self.main_dir)
                    dir_path = "%s/Bateria_Datos/BraTS2023/BraTS2023_TrainingData" %(self.main_dir) 
                elif directorio == "train_dir":
                    dir_path = "%s/Bateria_Datos/preproc/BraTS2023/train" %(self.main_dir)
                elif directorio == "val_dir":
                    dir_path =  "%s/Bateria_Datos/preproc/BraTS2023/val" %(self.main_dir)
                elif directorio == "test_dir":
                    dir_path = "%s/Bateria_Datos/preproc/BraTS2023/test" %(self.main_dir)
                elif directorio == "results_dir":
                    dir_path = "%s/Resultados" %(self.main_dir)
                    
            elif self.framework == "desktop":
                if directorio == "raw_data":
                    dir_path = "%s/Bateria_Datos/MICCAI_BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData" %(self.main_dir) 
                elif directorio == "train_dir":
                    dir_path = "%s/PrepData/BraTS2023/train" %(self.main_dir)
                elif directorio == "val_dir":
                    dir_path =  "%s/PrepData/BraTS2023/val" %(self.main_dir)
                elif directorio == "test_dir":
                    dir_path = "%s/PrepData/BraTS2023/test" %(self.main_dir)
                elif directorio == "results_dir":
                    dir_path = "%s/Resultados" %(self.main_dir)

            elif self.framework == "laptop":
                pass

            return dir_path

        def out_subdir(self,instance,trained_model,database):
            
            if trained_model == "default":
                train_db = "BraTS2023"
                submodel = "Main"#"weighted2ET"#"Main"
                
            else:
                if "-" in trained_model or "_" in trained_model:
                    train_db = trained_model.split("_")[0]
                    submodel = trained_model.split("_")[1]
                else:
                    train_db = trained_model
                    submodel = "Main"
                    
                if train_db == "BraTS2020" or train_db == "brats" or train_db == "brats2020":
                    train_db = "BraTS2020"
                elif train_db == "BraTS2023" or train_db == "brats2023":
                    train_db = "BraTS2023"

            
            if instance == "train":                        
                dir_path = "UdenseResnet/%s/%s/%s" %(submodel,train_db,instance)   
            elif instance == "test":                        
                dir_path = "UdenseResnet/%s/%s/%s/%s" %(submodel,train_db,instance,database)
            return dir_path
            
    #-------image-info------------------------------------------------------
        def sequences(self):
            seq = [1,1,1,1]  #secuencias disponibles en base de datos, en este orden T1,T2,T1ce,Flair
            return seq
        
        def label_avb(self): #si existe segmentacion disponible
            lbl = True
            return lbl            
        
        def image_orientation(self):
            orientation = "RAI" 
            return orientation    

        def input_size(self):
            img_sz = (155,240,240)
            return img_sz

        def spacing(self):
            img_spc = (1, 1, 1)
            return img_spc    
    
    #------preprocessing--------------------------------------
        def image_transformation(self, op):
            dlt_bgr = False #
            rs_ds = False #
            padding = False # 
            
            if op == "dlt_bgr":
                return dlt_bgr
            elif op == "resize":
                return rs_ds
            elif op == "padding":
                return padding
    #------------------------------------------------------------------
        def get_subj_id(self,sujeto,instance="generate_db"):
            if instance =="generate_db":
                subj_id = sujeto[10:19]     
            elif instance =="test":
                subj_id = sujeto[10:-3] 
                
            return subj_id

        def h5_file_name(self,sujeto,grupo):
                
            h5fn = "%s_BraTS2023_%s" %(grupo,sujeto)
            return h5fn   

        def trained_model(self,model,out_mode="model_path",model_state="best"):
            if model == "default":
                train_db = "BraTS2023"
                submodel = "Main"#"weighted2ET"#"Main"
                
            else:
                if "-" in model or "_" in model:
                    train_db = model.split("_")[0]
                    submodel = model.split("_")[1]
                else:
                    train_db = model.split("-")[0]
                    submodel = "Main"
                    
                if train_db  == "BraTS2020" or train_db == "brats" or train_db == "brats2020":
                    train_db = "BraTS2020"
                if train_db == "BraTS2023" or train_db == "brats2023":
                    train_db = "BraTS2023"   
                
            if out_mode == "model_path":
                out = "%s/Trained_Models/UdenseResnet/%s/%s" %(self.main_dir,submodel,train_db)
            if out_mode == "checkpoint":
                if model_state == "best":
                    modelstate = "best_model"
                else:    
                    modelstate = str(model).zfill(5)
                    
                
                out = "%s/Trained_Models/UdenseResnet/%s/%s/%s.pth" %(self.main_dir,submodel,train_db,modelstate)
            
            return out  


        def train_config(self,config):
            if self.framework == "desktop":
                if config == "loss_weights":
                    config_value = [1,1.5,1,1] 
                elif config == "num_classes":
                    config_value = 4
                elif config == "num_epoch":
                    config_value = 300
                elif config == "lr_init":
                    config_value = 2e-4
                elif config == "lr_stepsize":
                    config_value = 30
                elif config == "batch_train":
                    config_value = 2
                elif config == "patches_size":
                    config_value = (128,128,128)
                elif config == "solapado":
                    config_value = 16    
                elif config == "whole_val": 
                    config_value = True

            elif self.framework == "cluster":
                if config == "loss_weights":
                    config_value = [1,1.5,1,1]#[1,1.5,1,2] 
                elif config == "num_classes":
                    config_value = 4
                elif config == "num_epoch":
                    config_value = 300
                elif config == "lr_init":
                    config_value = 2e-4
                elif config == "lr_stepsize":
                    config_value = 30
                elif config == "batch_train":
                    config_value = 8
                elif config == "patches_size":
                    config_value = (128,128,128)
                elif config == "solapado":
                    config_value = 100     
                elif config == "solapado_val":
                    config_value = 1
                elif config == "solapado_test":
                    config_value = 100                
                elif config == "whole_val": 
                    config_value = True


            return config_value



    class HCvB:
        def __init__(self,framework):
            self.framework = framework
            if framework == "cluster":
                self.main_dir = "/home4/eduardo.cavieres"
            elif framework == "desktop":
                self.main_dir = "D:/EduardoCavieres/TesisMagister"
            elif framework == "linux":
                self.main_dir = "/home/edo/Documents"


        def dir_path(self,directorio):
            if self.framework == "cluster":
                if directorio == "raw_data":
                    dir_path = "%s/" %(self.main_dir) 
                elif directorio == "train_dir":
                    dir_path = "%s/" %(self.main_dir)
                elif directorio == "val_dir":
                    dir_path =  "%s/" %(self.main_dir)
                elif directorio == "test_dir":
                    dir_path = "%s/" %(self.main_dir)
                elif directorio == "results_dir":
                    dir_path = "%s/" %(self.main_dir)  
                    
            elif self.framework == "desktop":
                if directorio == "raw_data":
                    dir_path = "%s/Bateria_Datos/HCvB_db" %(self.main_dir) 
                elif directorio == "train_dir":
                    dir_path = "%s/PrepData/HCvB/train" %(self.main_dir)
                elif directorio == "val_dir":
                    dir_path =  "%s/PrepData/HCvB/val" %(self.main_dir)
                elif directorio == "test_dir":
                    dir_path = "%s/PrepData/HCvB/test" %(self.main_dir)
                elif directorio == "results_dir":
                    dir_path = "%s/" %(self.main_dir)
            
            elif self.framework == "laptop":
                if directorio == "raw_data":
                    dir_path = "%s/HCvB_BET" %(self.main_dir) 
                elif directorio == "train_dir":
                    dir_path = "%s/HCvB_train" %(self.main_dir)
                elif directorio == "val_dir":
                    dir_path =  "%s/HCvB_val" %(self.main_dir)
                elif directorio == "test_dir":
                    dir_path = "%s/HCvB_test" %(self.main_dir)
                elif directorio == "results_dir":
                    dir_path = "%s/HCvB_Results" %(self.main_dir)

            return dir_path          
        
        def trained_model(self,model):
            if "-" in model or "_" in model:
                modelo = model.split("-")[0]
                submodel = model.split("-")[1]
            else:
                modelo = model.split("-")[0]
                submodel = "MAIN"
                
            if modelo == "BraTS2020" or modelo == "brats" or modelo == "brats2020":
                modelo = "BraTS2020"
                
            if modelo == "BraTS2020" or modelo == "brats" or modelo == "brats2020":
                modelo = "BraTS2020"    
                
        
        def sequences(self):
            seq = [0,1,0,0] #solo existe t1

        def image_orientation(self):
            orientation = "SPR"

        def input_size(self):
            img_sz = (157,189,156)
     
        def label_avb(self):
            lbl = False
            return lbl
        
        def raw_data_location(self,raw_data,sujeto,imagen):
            if imagen == "t1":
                rdl = os.path.join("%s/P%s/P%s_T1FLAIR" %(raw_data,sujeto,sujeto) + '.nii.gz') 
            elif imagen == "flair":   
               rdl = os.path.join("%s/P%s/P%s_T2FLAIR" %(raw_data,sujeto,sujeto) + '.nii.gz') 
            return rdl  

        def get_subj_id(sujetos):
            subj_id = []
            for subj in sujetos:
                subj_id.append(subj[1:])
            return subj_id

        
        def h5_file_name(self,sujeto,grupo):
            if grupo == "Training":
                subset = "train"  
            elif grupo == "Validation":
                subset = "val"
            elif grupo == "Test" :
                subset = "test"
                
            h5fn = "%s_HCvB_%s" (subset,sujeto)
            return h5fn     


    class HCvBv2:
        def __init__(self,framework):
            self.framework = framework
            if framework == "cluster":
                self.main_dir = "/home4/eduardo.cavieres"
            elif framework == "desktop":
                self.main_dir = "D:/EduardoCavieres"
            elif framework == "linux":
                self.main_dir = "/home/edo/Documents"


        def raw_data_location(self,raw_data,sujeto,imagen):
            if imagen == "t1":
                rdl = os.path.join("%s/P%s/P%s_T1FLAIR" %(raw_data,sujeto,sujeto) + '.nii.gz') 
            elif imagen == "flair":   
               rdl = os.path.join("%s/P%s/P%s_T2LAIR" %(raw_data,sujeto,sujeto) + '.nii.gz') 
            elif imagen == "t2":
               rdl = os.path.join("%s/P%s/P%s_T2" %(raw_data,sujeto,sujeto) + '.nii.gz')
            return rdl 
        
        
        def dir_path(self,directorio):
            if self.framework == "cluster":
                if directorio == "raw_data":
                    dir_path = "%s/Bateria_Datos/HCvBv2_skullstrip" %(self.main_dir) 
                elif directorio == "train_dir":
                    dir_path = "%s/Bateria_Datos/preproc/HCvB/HCvBv2/train" %(self.main_dir)
                elif directorio == "val_dir":
                    dir_path =  "%s/Bateria_Datos/preproc/HCvB/HCvBv2/val" %(self.main_dir)
                elif directorio == "test_dir":
                    dir_path = "%s/Bateria_Datos/preproc/HCvB/HCvBv2/test" %(self.main_dir)
                elif directorio == "results_dir":
                    dir_path = "%s/Resultados" %(self.main_dir)  
            
            elif self.framework == "desktop":
                if directorio == "raw_data":
                    dir_path = "%s/Bateria_Datos/HCvBv2_bet-hd" %(self.main_dir) 
                elif directorio == "train_dir":
                    dir_path = "%s/PrepData/HCvB/HCvBv2/train" %(self.main_dir)
                elif directorio == "val_dir":
                    dir_path =  "%s/PrepData/HCvB/HCvBv2/val" %(self.main_dir)
                elif directorio == "test_dir":
                    dir_path = "%s/PrepData/HCvB/HCvBv2/test" %(self.main_dir)
                elif directorio == "results_dir":
                    dir_path = "%s/Resultados" %(self.main_dir)

            
            elif self.framework == "laptop":
                if directorio == "raw_data":
                    dir_path = "%s/HCvB_BET" %(self.main_dir) 
                elif directorio == "train_dir":
                    dir_path = "%s/HCvB_train" %(self.main_dir)
                elif directorio == "val_dir":
                    dir_path =  "%s/HCvB_val" %(self.main_dir)
                elif directorio == "test_dir":
                    dir_path = "%s/HCvB_test" %(self.main_dir)
                elif directorio == "results_dir":
                    dir_path = "%s/HCvB_Results" %(self.main_dir)

            return dir_path          
        
        def out_subdir(self,instance,trained_model,database):
            
            if trained_model == "default":
                train_db = "BraTS2020"
                submodel = "ILD"
                
            else:
                if "-" in trained_model or "_" in trained_model:
                    train_db = trained_model.split("_")[0]
                    submodel = trained_model.split("_")[1]
                else:
                    train_db = trained_model
                    submodel = "Main"
                    
                if train_db == "BraTS2020" or modelo == "brats" or modelo == "brats2020":
                    train_db = "BraTS2020"
            
            if instance == "train":                        
                dir_path = "UdenseResnet/%s/%s/%s" %(submodel,train_db,instance)   
            elif instance == "test":                        
                dir_path = "UdenseResnet/%s/%s/%s/%s" %(submodel,train_db,instance,database)
            elif instance == "prep_data":
                dir_path = "/%s/%s/%s/%s" %(submodel,train_db,instance,database)
            return dir_path
            
        
               
        def sequences(self):
            seq = [1,1,0,1] #solo existe t1(flair) y flair(T2)
            return seq
        
        def label_avb(self): #si existe segmentacion disponible
            lbl = False
            return lbl
        
        def image_orientation(self):
            orientation = "ASL"
            return orientation
        
        def input_size(self):
            img_sz = (22, 512, 512)
            return img_sz
        
        def spacing(self):
            img_spc = (5.0, 0.4688, 0.4688)
            return img_spc
                    
        def image_transformation(self, op):
            dlt_bgr = False#True #
            rs_ds = False #
            padding = False#True # 
            
            if op == "dlt_bgr":
                return dlt_bgr
            elif op == "resize":
                return rs_ds
            elif op == "padding":
                return padding
        
        def get_subj_id(self,sujeto,instance="generate_db"):
            if instance =="generate_db":
                subj_id = sujeto[1:]     
            elif instance =="test":
                subj_id = sujeto[10:-3] 
                
            return subj_id
        
        def h5_file_name(self,sujeto,grupo):
                
            h5fn = "%s_HCvB_%s" %(grupo,sujeto)
            return h5fn     
        
        def trained_model(self,model,out_mode="model_path",model_state="best"):
            if model == "default":
                train_db = "BraTS2020"
                submodel = "Main"
                
            else:
                if "-" in model or "_" in model:
                    train_db = model.split("_")[0]
                    submodel = model.split("_")[1]
                else:
                    train_db = model.split("-")[0]
                    submodel = "Main"
                    
                if train_db == "BraTS2020" or modelo == "brats" or modelo == "brats2020":
                    train_db = "BraTS2020"
                
            if out_mode == "model_path":
                out = "%s/Trained_Models/UdenseResnet/%s/%s" %(self.main_dir,submodel,train_db)
            if out_mode == "checkpoint":
                if model_state == "best":
                    modelstate = "best_model"
                else:    
                    modelstate = str(model).zfill(5)
                    
                
                out = "%s/Trained_Models/UdenseResnet/%s/%s/%s.pth" %(self.main_dir,submodel,train_db,modelstate)
            
            return out

        def train_config(self,config):
            if self.framework == "desktop":
                if config == "loss_weights":
                    config_value = [1,1.5,1,1]
                elif config == "num_classes":
                    config_value = 4
                elif config == "num_epoch":
                    config_value = 300
                elif config == "lr_init":
                    config_value = 2e-4
                elif config == "lr_stepsize":
                    config_value = 30
                elif config == "batch_train":
                    config_value = 2
                elif config == "patches_size":
                    config_value = (64,64,64)
                elif config == "solapado":
                    config_value = 30
                elif config == "solapado_val":
                    config_value = 1
                elif config == "solapado_test":
                    config_value = 32
                elif config == "whole_val":
                    config_value = True

            elif self.framework == "cluster":
                if config == "loss_weights":
                    config_value = [1,1.5,1,1]
                elif config == "num_classes":
                    config_value = 4
                elif config == "num_epoch":
                    config_value = 3000
                elif config == "lr_init":
                    config_value = 2e-4
                elif config == "lr_stepsize":
                    config_value = 30
                elif config == "batch_train":
                    config_value = 8
                elif config == "patches_size":
                    config_value = (16,128,128)
                elif config == "solapado":
                    config_value = 100
                elif config == "solapado_val":
                    config_value = 1
                elif config == "solapado_test":
                    config_value = 100
                elif config == "whole_val":
                    config_value = True


            return config_value


    class HCvBv3:
        def __init__(self,framework):
            self.framework = framework
            if framework == "cluster":
                self.main_dir = "/home4/eduardo.cavieres"
            elif framework == "desktop":
                self.main_dir = "D:/EduardoCavieres"
            elif framework == "linux":
                self.main_dir = "/home/edo/Documents"


        def raw_data_location(self,raw_data,sujeto,imagen):
            if imagen == "t1ce":
                rdl = os.path.join("%s/P%s/P%s_T1ce" %(raw_data,sujeto,sujeto) + '.nii')
            elif imagen == "t2":
               rdl = os.path.join("%s/P%s/P%s_T2" %(raw_data,sujeto,sujeto) + '.nii')
            return rdl


        def dir_path(self,directorio):
            if self.framework == "cluster":
                if directorio == "raw_data":
                    dir_path = "%s/Bateria_Datos/HCvBv3_reg" %(self.main_dir)
                elif directorio == "train_dir":
                    dir_path = "%s/Bateria_Datos/preproc/HCvB/HCvBv3_reg/train" %(self.main_dir)
                elif directorio == "val_dir":
                    dir_path =  "%s/Bateria_Datos/preproc/HCvB/HCvBv3_reg/val" %(self.main_dir)
                elif directorio == "test_dir":
                    dir_path = "%s/Bateria_Datos/preproc/HCvB/HCvBv3_reg/test" %(self.main_dir)
                elif directorio == "results_dir":
                    dir_path = "%s/Resultados" %(self.main_dir)

            elif self.framework == "desktop":
                if directorio == "raw_data":
                    dir_path = "%s/Bateria_Datos/HCvBv3bet-hd" %(self.main_dir)
                elif directorio == "train_dir":
                    dir_path = "%s/PrepData/HCvB/HCvBv3/train" %(self.main_dir)
                elif directorio == "val_dir":
                    dir_path =  "%s/PrepData/HCvB/HCvBv3/val" %(self.main_dir)
                elif directorio == "test_dir":
                    dir_path = "%s/PrepData/HCvB/HCvBv3/test" %(self.main_dir)
                elif directorio == "results_dir":
                    dir_path = "%s/Resultados" %(self.main_dir)


            elif self.framework == "laptop":
                if directorio == "raw_data":
                    dir_path = "%s/HCvB_BET" %(self.main_dir)
                elif directorio == "train_dir":
                    dir_path = "%s/HCvB_train" %(self.main_dir)
                elif directorio == "val_dir":
                    dir_path =  "%s/HCvB_val" %(self.main_dir)
                elif directorio == "test_dir":
                    dir_path = "%s/HCvB_test" %(self.main_dir)
                elif directorio == "results_dir":
                    dir_path = "%s/HCvB_Results" %(self.main_dir)

            return dir_path

        def out_subdir(self,instance,trained_model,database):

            if trained_model == "default":
                train_db = "BraTS2020"
                submodel = "ILD"

            else:
                if "-" in trained_model or "_" in trained_model:
                    train_db = trained_model.split("_")[0]
                    submodel = trained_model.split("_")[1]
                else:
                    train_db = trained_model
                    submodel = "Main"

                if train_db == "BraTS2020" or modelo == "brats" or modelo == "brats2020":
                    train_db = "BraTS2020"

            if instance == "train":
                dir_path = "UdenseResnet/%s/%s/%s" %(submodel,train_db,instance)
            elif instance == "test":
                dir_path = "UdenseResnet/%s/%s/%s/%s" %(submodel,train_db,instance,database)
            elif instance == "prep_data":
                dir_path = "/%s/%s/%s/%s" %(submodel,train_db,instance,database)
            return dir_path



        def sequences(self):
            seq = [0,1,1,0] #solo existe t1ce y (T2)
            return seq

        def label_avb(self): #si existe segmentacion disponible
            lbl = False
            return lbl

        def image_orientation(self):
            orientation = "RAI"
            return orientation

        def input_size(self):
            img_sz = (240, 240, 155)
            return img_sz

        def spacing(self):
            img_spc = (1, 1, 1)
            return img_spc

        def image_transformation(self, op):
            dlt_bgr = False#True #
            rs_ds = False #
            padding = False#True #

            if op == "dlt_bgr":
                return dlt_bgr
            elif op == "resize":
                return rs_ds
            elif op == "padding":
                return padding

        def get_subj_id(self,sujeto,instance="generate_db"):
            if instance =="generate_db":
                subj_id = sujeto[1:]
            elif instance =="test":
                subj_id = sujeto[10:-3]

            return subj_id

        def h5_file_name(self,sujeto,grupo):

            h5fn = "%s_HCvB_%s" %(grupo,sujeto)
            return h5fn

        def trained_model(self,model,out_mode="model_path",model_state="best"):
            if model == "default":
                train_db = "BraTS2020"
                submodel = "Main"

            else:
                if "-" in model or "_" in model:
                    train_db = model.split("_")[0]
                    submodel = model.split("_")[1]
                else:
                    train_db = model.split("-")[0]
                    submodel = "Main"

                if train_db == "BraTS2020" or modelo == "brats" or modelo == "brats2020":
                    train_db = "BraTS2020"

            if out_mode == "model_path":
                out = "%s/Trained_Models/UdenseResnet/%s/%s" %(self.main_dir,submodel,train_db)
            if out_mode == "checkpoint":
                if model_state == "best":
                    modelstate = "best_model"
                else:
                    modelstate = str(model).zfill(5)


                out = "%s/Trained_Models/UdenseResnet/%s/%s/%s.pth" %(self.main_dir,submodel,train_db,modelstate)

            return out

        def train_config(self,config):
            if self.framework == "desktop":
                if config == "loss_weights":
                    config_value = [1,1.5,1,1]
                elif config == "num_classes":
                    config_value = 4
                elif config == "num_epoch":
                    config_value = 300
                elif config == "lr_init":
                    config_value = 2e-4
                elif config == "lr_stepsize":
                    config_value = 30
                elif config == "batch_train":
                    config_value = 2
                elif config == "patches_size":
                    config_value = (64,64,64)
                elif config == "solapado":
                    config_value = 30
                elif config == "solapado_val":
                    config_value = 1
                elif config == "solapado_test":
                    config_value = 32
                elif config == "whole_val":
                    config_value = True

            elif self.framework == "cluster":
                if config == "loss_weights":
                    config_value = [1,1.5,1,1]
                elif config == "num_classes":
                    config_value = 4
                elif config == "num_epoch":
                    config_value = 3000
                elif config == "lr_init":
                    config_value = 2e-4
                elif config == "lr_stepsize":
                    config_value = 30
                elif config == "batch_train":
                    config_value = 8
                elif config == "patches_size":
                    config_value = (128,128,128)
                elif config == "solapado":
                    config_value = 100
                elif config == "solapado_val":
                    config_value = 1
                elif config == "solapado_test":
                    config_value = 100
                elif config == "whole_val":
                    config_value = True


            return config_value






    class BraTS2020_2d:
        def __init__(self,framework):
            self.framework = framework
            if framework == "cluster":
                self.main_dir = "/home4/eduardo.cavieres/"
            elif framework == "desktop":
                self.main_dir = "D:/EduardoCavieres/TesisMagister"
            elif framework == "linux":
                self.main_dir = "/home/edo/Documents"
    #-------------- path config-------------------------------------            
        def raw_data_location(self,raw_data,sujeto,imagen):
            if imagen == "t1":
                rdl = os.path.join("%s/BraTS20_Training_%s/BraTS20_Training_%s_t1" %(raw_data,sujeto,sujeto) + '.nii.gz') 
            elif imagen == "t2":   
               rdl = os.path.join("%s/BraTS20_Training_%s/BraTS20_Training_%s_t2" %(raw_data,sujeto,sujeto) + '.nii.gz') 
            elif imagen == "t1ce":   
               rdl = os.path.join("%s/BraTS20_Training_%s/BraTS20_Training_%s_t1ce" %(raw_data,sujeto,sujeto) + '.nii.gz')                
            elif imagen == "flair":   
               rdl = os.path.join("%s/BraTS20_Training_%s/BraTS20_Training_%s_flair" %(raw_data,sujeto,sujeto) + '.nii.gz') 
            elif imagen == "seg":   
               rdl = os.path.join("%s/BraTS20_Training_%s/BraTS20_Training_%s_seg" %(raw_data,sujeto,sujeto) + '.nii.gz') 
            return rdl     
        
        def dir_path(self,directorio):
            if self.framework == "cluster":
                if directorio == "raw_data":
                    dir_path = "%s/MICCAI_BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData" %(self.main_dir) 
                elif directorio == "train_dir":
                    dir_path = "%s/Bateria_Datos/PrepData/BraTS2020_2d/data_train" %(self.main_dir)
                elif directorio == "val_dir":
                    dir_path =  "%s/Bateria_Datos/PrepData/BraTS2020_2d/data_val" %(self.main_dir)
                elif directorio == "test_dir":
                    dir_path = "%s/Bateria_Datos/PrepData/BraTS2020_2d/data_test" %(self.main_dir)
                elif directorio == "results_dir":
                    dir_path = "%s/Resultados/UdenseResnet" %(self.main_dir)
                    
            elif self.framework == "desktop":
                if directorio == "raw_data":
                    dir_path = "%s/Bateria_Datos/MICCAI_BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData" %(self.main_dir) 
                elif directorio == "train_dir":
                    dir_path = "%s/PrepData/BraTS_2020_2d/data_train" %(self.main_dir)
                elif directorio == "val_dir":
                    dir_path =  "%s/PrepData/BraTS_2020_2d/data_val" %(self.main_dir)
                elif directorio == "test_dir":
                    dir_path = "%s/PrepData/BraTS_2020_2d/data_test" %(self.main_dir)
                elif directorio == "results_dir":
                    dir_path = "%s/Resultados/UdenseResnet" %(self.main_dir)
                    
            return dir_path
        
        def out_subdir(self,instance,trained_model,database):
            if trained_model == "default":
                train_db = "BraTS2020_2d"
                submodel = "Main"
                
            else:
                if "-" in trained_model or "_" in trained_model:
                    train_db = trained_model.split("_")[0]
                    submodel = trained_model.split("_")[1]
                else:
                    train_db = trained_model
                    submodel = "Main"
                    
                if train_db == "BraTS2020_2d" or train_db == "brats_2d" or train_db == "brats2020_2d":
                    train_db = "BraTS2020_2d"
                elif train_db == "BraTS2023_2d" or train_db == "brats2023_2d":
                    train_db = "BraTS2023_2d"    
            
            if instance == "train":                        
                dir_path = "/%s/%s/%s" %(submodel,train_db,instance)   
            elif instance == "test":                        
                dir_path = "/%s/%s/%s/%s" %(submodel,train_db,instance,database)
            elif instance == "prep_data":
                dir_path = "/%s/%s/%s/%s" %(submodel,train_db,instance,database)
            return dir_path
     #-------image-info------------------------------------------------------    
        def sequences(self):
            seq = [1,1,1,1]  #secuencias disponibles en base de datos, en este orden T1,T2,T1ce,Flair
            return seq
        
        def label_avb(self):
            lbl = True
            return lbl
        
        def image_orientation(self,inout = "in"):
            if inout == "in":
                orientation = "RAI"
            return orientation
        
        def input_size(self):
            img_sz = (29, 240, 240)
            return img_sz
        
        def spacing(self):
            img_spc = (1, 1, 1)
            return img_spc 
        
     #------preprocessing--------------------------------------   
        def image_transformation(self, op):
            dlt_bgr = False #
            rs_ds = False #
            padding = False # 
            extract_slices = False
            merge_slices = True

            if op == "dlt_bgr":
                return dlt_bgr
            elif op == "resize":
                return rs_ds
            elif op == "padding":
                return padding
            elif op == "extract_slices":
                return extract_slices
            elif op == "merge_slices":
                return merge_slices
     #------------------------------------------------------------------
        def get_subj_id(self,sujeto,instance="generate_db"):
            if instance =="generate_db":
                subj_id = sujeto[17:20]     
            elif instance =="test":
                subj_id = sujeto[12:-3] 
                
            return subj_id
        
        def h5_file_name(self,sujeto,grupo):
                
            h5fn = "%s_BraTS2020-2d_%s" %(grupo,sujeto)
            return h5fn
        
        def trained_model(self,model,out_mode="model_path",model_state="best"):
            if model == "default":
                train_db = "BraTS2020_2d"
                submodel = "Main"
                
            else:
                if "-" in model or "_" in model:
                    modelo = model.split("_")[0]
                    submodel = model.split("_")[1]
                else:
                    modelo = model.split("-")[0]
                    submodel = "Main"
                    
                if modelo == "BraTS2020" or modelo == "brats" or modelo == "brats2020":
                    train_db = "BraTS2020"
                if modelo == "BraTS2023" or modelo == "brats2023":
                    train_db = "BraTS2023"   
                if modelo == "BraTS2020_2d" or modelo == "brats_2d" or modelo == "brats2020_2d":
                    train_db = "BraTS2020_2d"


            if out_mode == "model_path":
                out = "%s/Trained_Models/UdenseResnet/%s/%s" %(self.main_dir,submodel,train_db)
            if out_mode == "checkpoint":
                if model_state == "best":
                    modelstate = "best_model"
                else:    
                    modelstate = str(model).zfill(5)
                    
                
                out = "%s/Trained_Models/UdenseResnet/%s/%s/%s.pth" %(self.main_dir,submodel,train_db,modelstate)
            
            return out  


        def train_config(self,config):
            if self.framework == "desktop":
                if config == "loss_weights":
                    config_value = [1,1.5,1,1] 
                elif config == "num_classes":
                    config_value = 4
                elif config == "num_epoch":
                    config_value = 300
                elif config == "lr_init":
                    config_value = 2e-4
                elif config == "lr_stepsize":
                    config_value = 30
                elif config == "batch_train":
                    config_value = 2
                elif config == "patches_size":
                    config_value = (64,64,64)
                elif config == "solapado":
                    config_value = 30
                elif config == "solapado_val":
                    config_value = 1
                elif config == "solapado_test":
                    config_value = 32    
                elif config == "whole_val": 
                    config_value = True

            elif self.framework == "cluster":
                if config == "loss_weights":
                    config_value = [1,1.5,1,1] 
                elif config == "num_classes":
                    config_value = 4
                elif config == "num_epoch":
                    config_value = 3000
                elif config == "lr_init":
                    config_value = 2e-4
                elif config == "lr_stepsize":
                    config_value = 30
                elif config == "batch_train":
                    config_value = 8
                elif config == "patches_size":
                    config_value = (16,128,128)
                elif config == "solapado":
                    config_value = 100
                elif config == "solapado_val":
                    config_value = 1
                elif config == "solapado_test":
                    config_value = 100      
                elif config == "whole_val": 
                    config_value = True


            return config_value            


#cudnn.enabled   = True
#cudnn.benchmark = True


num_epoch = 300
lr_init = 2e-4
lr_stepsize = 30#100# 5000
batch_train =2#2
whole_val = True
ILD = True 

loss_weights = [1,1.5,1,1]                #None si no se define
num_classes= 4                     #numero de labels
img_in_size = (155, 240,240)
patches_size = (32,32,32)
solapado = 16
checkpoint_name= 'model_epoch'
dlt_bgr = False
rs_ds = False

