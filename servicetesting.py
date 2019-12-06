# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 08:25:53 2019

@author: chiku
"""

import requests
import json
import pandas as pd

X_test_data=pd.read_csv('./raw_data_1205.csv')
data = X_test_data.loc[1].to_json()
data
# send a random row from the test set to score
# random_index = np.random.randint(0, len(X_test)-1)
# input_data = "{\"data\": [" + str(list(data[0])) + "]}"
input_data = data
scoring_uri = 'http://8a17b781-550b-4106-a36d-03ec01a70031.westus.azurecontainer.io/score'
headers = {'Content-Type':'application/json'}
resp = requests.post(scoring_uri, input_data, headers=headers)

# input data
print("POST to url", scoring_uri)
print("input json data:", input_data)

# output data
print(resp.text)
#output_json = json.loads(resp.text)
#output_json = json.loads(output_json)
#print("output json data: ",output_json)

# output data & rule
#result = output_json["result"]

#print("{}".format(result))