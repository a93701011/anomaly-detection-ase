# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 22:41:51 2019

@author: chiku
"""

import azureml.core

print("This notebook was created using version 1.0.2 of the Azure ML SDK")
print("You are currently using version", azureml.core.VERSION, "of the Azure ML SDK")

import os
# mlworkspace
#subscription_id = os.getenv("SUBSCRIPTION_ID", default="09ba1f2e-4799-434c-9f88-6ca60b368ac8")
#resource_group = os.getenv("RESOURCE_GROUP", default="mlworkspace")
#workspace_name = os.getenv("WORKSPACE_NAME", default="mlkaren")
#workspace_region = os.getenv("WORKSPACE_REGION", default="southcentral")


subscription_id = os.getenv("SUBSCRIPTION_ID", default="486bb824-bc00-4554-98f0-ed1d4563959d")
resource_group = os.getenv("RESOURCE_GROUP", default="A3CIM")
workspace_name = os.getenv("WORKSPACE_NAME", default="Virtual_Predict")
workspace_region = os.getenv("WORKSPACE_REGION", default="japaneast")

from azureml.core import Workspace

try:
    ws = Workspace(subscription_id = subscription_id, resource_group = resource_group, workspace_name = workspace_name)
    # write the details of the workspace to a configuration file to the notebook library
    ws.write_config()
    print("Workspace configuration succeeded. Skip the workspace creation steps below")
except:
    print("Workspace not accessible. Change your parameters or create a new workspace below")
    
   