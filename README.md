# Installation

1.python connection to sql server

```bash
pip install pyodbc 

2.ODBC Driver 17 for SQL Server 
https://www.microsoft.com/zh-tw/download/details.aspx?id=56567 


# anomaly_detection_ase

run by order

1.feature.py  

$python feature.py start_date end_date output_file_name

2.model.py

$python model.py input_file_name

3.azurews.py

azure machine learning ws connection configuration

4.registermodel.py

$python RegisterModel WhatModel

