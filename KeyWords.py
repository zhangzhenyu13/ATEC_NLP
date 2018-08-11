import pandas as pd
import numpy as np



def transformStr(df1,df2):
    idset=set(df1["id"])
    data={"id":[],"keywords1":[],"keywords2":[]}
    for id in idset:
        keywords1=np.array(df1.loc[id,"keywords"])
        keywords2=np.array(df2.loc[id,"keywords"])
        keywords1=" ".join(keywords1)
        keywords2=" ".join(keywords2)

        data["id"].append(id)
        data["keywords1"].append(keywords1)
        data["keywords2"].append(keywords2)
    data=pd.DataFrame(data=data)

    return data

if __name__ == '__main__':
    data=transformStr(df1,df2)

    topai(1,data)
