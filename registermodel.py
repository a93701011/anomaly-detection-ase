# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 22:43:11 2019

@author: chiku
"""

from sklearn.externals import joblib

from azureml.core import Workspace
ws = Workspace.from_config()

RegisterModel = 'fdc'
WhatModel = 'OneClassSVM'

from azureml.core.model import Model

model = Model.register(model_path = "./{}.pkl".format(WhatModel),
                       model_name = "{}_{}".format(RegisterModel,WhatModel),
                       tags = {'RECIPE': RegisterModel, 'AlGORUTHM':WhatModel },
                       workspace = ws)
print(model.name, model.tags, model.version, sep = '\t')

from azureml.core.model import Model
#from keras.wrappers.scikit_learn import KerasRegressor
#from keras.models import load_model

model=Model(ws, '{}_{}'.format(RegisterModel,WhatModel))
model.download(target_dir='.', exist_ok=True)

import os 
os.stat('./{}.pkl'.format(WhatModel))
model_path = Model.get_model_path('{}.pkl'.format(WhatModel))
estimator = joblib.load(model_path)


# Create Docker Image
    
print('# Create Docker Image')
from azureml.core.conda_dependencies import CondaDependencies 

cd = CondaDependencies()
cd.save_to_file(".", "myenv.yml")

# This specifies the dependencies to include in the environment
from azureml.core.conda_dependencies import CondaDependencies 

myenv = CondaDependencies.create(conda_packages=['pandas', 'scikit-learn', 'numpy', 'keras', 'joblib'])

with open("myenv.yml","w") as f:
    f.write(myenv.serialize_to_string())
    
from azureml.core.image import Image
from azureml.core.image import ContainerImage
# configure the image
image_config = ContainerImage.image_configuration(execution_script="score.py", 
                                                  runtime="python", 
                                                  conda_file="myenv.yml")
image = Image.create(name = "fdc-oneclasssvm",
                     # this is the model object 
                     models = [model],
                     image_config = image_config, 
                     workspace = ws)
image.wait_for_creation(show_output = True)
# Create Container Instance

from azureml.core.webservice import AciWebservice

aciconfig = AciWebservice.deploy_configuration(cpu_cores=1, 
                                               memory_gb=1, 
                                               tags={"data": "nopbcsr-s",  "method" : WhatModel}, 
                                               description='Predict AVM Profile')

from azureml.core.webservice import Webservice

aci_service_name = 'fdc-oneclass-prediction1'
print(aci_service_name)
aci_service = Webservice.deploy_from_image(deployment_config = aciconfig,
                                           image = image,
                                           name = aci_service_name,
                                           workspace = ws)
aci_service.wait_for_deployment(True)
print(aci_service.state)
# retrive service
from azureml.core.webservice import Webservice
aci_service_name = 'fdc-oneclass-prediction'
aci_service = Webservice(ws, aci_service_name)

print(aci_service.scoring_uri)

print('model created complete...')