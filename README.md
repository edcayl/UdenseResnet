Pasos:

1) Editar archivo de configuración:
        global_config.py

          
3) Crear archivos H5:
       Generate_db.py
          -fm    # seleccionar configuracion espacio de trabajo (ver archivo global_config.py) 
          -db    # seleccionar configuracion de base de datos (ver archivo global_config.py)
          -ts    # tamaño grupo entrenamiento, si es flotante =< 1 se considera como porcentaje
          -vs    # tamaño grupo validacion,  si es flotante =< 1 se considera como porcentaje
          --test_size    #tamaño grupo de test,  si es flotante =< 1 se considera como porcentaje
5) Entrenamiento:
       Train.py
          -fm    # seleccionar configuracion espacio de trabajo (ver archivo global_config.py) 
          -db    # seleccionar configuracion de base de datos (ver archivo global_config.py)
         -model #cada cuanto guardar checkpoints ("all" para todos )
   
7) Test
       Test.py
           -fm    # seleccionar configuracion espacio de trabajo (ver archivo global_config.py) 
          -db    # seleccionar configuracion de base de datos (ver archivo global_config.py)
           --test_model  # elegir checkpoint de red entrenada 
   




