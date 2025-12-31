#manually label comments in CSV file iteratively in preparation for sentiment analysis

import pandas as pd
import os.path

INPUT_CSV = "data/processed/reddit/cleaned_comments.csv" 
OUTPUT_CSV = "data/processed/reddit/labeled_comments.csv"

def label_loop():
    df = pd.read_csv(INPUT_CSV)

    missedLabels=[]
    startingValue=0

    if os.path.isfile(OUTPUT_CSV):
        af=pd.read_csv(OUTPUT_CSV)
        #file exists, find current point
        for i in range (len(af)):
            #missing or skipped rows
            if str(af.iat[i,0]) != str(df.iat[i+len(missedLabels),0]):
                missedLabels.append(i)
        
        startingValue=len(missedLabels)

    counter=startingValue
    outputDict={}
    
    if os.path.isfile(OUTPUT_CSV):
        outputDict=af.to_dict(orient='list')
    else:
        for column in df.columns:
            outputDict[column]=[]
        outputDict["label"]=[]
    
    yf_indexed=pd.DataFrame(outputDict)
    yf_indexed.to_csv(OUTPUT_CSV, index=False)

    returnDf=df.loc[startingValue:len(df)].iterrows()

    vals=next(returnDf)
    while counter < len(df):

        print(vals[1].values[1] + " trade " + vals[1].values[2] + " " + vals[1].values[3])

        print(vals[1].values[5])
        print('-------')

        print("0: neutral, 1: bad, 2: good, 3, irrelevant, q: quit")
        labelValue=input()
        
        if labelValue in ['0', '1', '2', '3', 'q']:
            
            if labelValue!='q':
                smallCount=0
                for column in df.columns:
                    outputDict[column].append(vals[1].values[smallCount])
                    smallCount+=1
                outputDict['label'].append(labelValue)
            else:
                xf_indexed=pd.DataFrame(outputDict)
                xf_indexed.to_csv(OUTPUT_CSV, index=False)
                return

            for x in outputDict.values():
                print(len(x))

            vals=next(returnDf)
            counter+=1
        else:
            print('relabel past')

            
if __name__ == "__main__":
    label_loop()
