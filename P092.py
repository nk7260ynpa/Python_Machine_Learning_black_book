import pandas as pd
from io import StringIO

csv_data="""A,B,C,D
         1.0,2.0,3.0,4.0
         5.0,6.0,,8.0
         0.0,11.0,12.0,"""

df = pd.read_csv(StringIO(csv_data))

#print(df)
#print(df.isnull().sum())
#print(df.values)
#print(df.dropna())
#print(df.dropna(axis=1))
#print(df.dropna(how="all")) drop rows where all columns are NaN
#print(df.dropna(thresh=4))  not at least drop 4 non-NaN values
#print(df.dropna(subset=["C"]))














