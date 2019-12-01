# Installation

1.python connection to sql server

```bash
pip install pyodbc 
```

2.ODBC Driver 17 for SQL Server 
https://www.microsoft.com/zh-tw/download/details.aspx?id=56567 


# How to use the script

run this script by order

1.feature.py  
create feature for training model.

```bash
$python feature.py start_date end_date output_file_name
```

2.model.py
training nodel.

```bash
$python model.py input_file_name
```

3.azurews.py
azure machine learning ws connection configuration.

4.registermodel.py
register model to azure machine learning serivce A3CIM.

```bash
$python RegisterModel WhatModel
```
