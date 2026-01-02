#manually check and edit highest weighted comments in labeled_comments before tuning a model

import pandas as pd
import os.path
import numpy as np

EDIT_CSV = "data/processed/reddit/labeled_comments.csv"

def edit_loop():

    if not os.path.isfile(EDIT_CSV):
        return -1

    df=pd.read_csv(EDIT_CSV)

    sortList=df.to_numpy()
    sortList=sortList[np.abs(sortList[:, 6]).argsort()] #sort array by 6th value (score)
    sortList=sortList[::-1]

    outputDict=df.to_dict(orient='list')

    counter=0
    totalIterLength=len(sortList)
    while counter < totalIterLength:
        
        print(sortList[counter])
        print()
        print(sortList[counter][6])
        print(sortList[counter][5])
        print('-----')
        
        print('new label = ?')
        print("0: neutral, 1: bad, 2: good, 3, irrelevant, q: quit")
        labelValue=input()

        if labelValue in ['0', '1', '2', '3']:
            valueIndex=outputDict['comment_id'].index(sortList[counter][0])
            outputDict['label'][valueIndex]=labelValue
            counter+=1
        elif labelValue == 'q':
            new_df=pd.DataFrame(outputDict)
            new_df.to_csv(EDIT_CSV, index=False)
            return
        else:
            print('try again ')

if __name__ == "__main__":
    edit_loop()