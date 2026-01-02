#generate unlabeled_comments.csv

import pandas as pd

LABELED_CSV="data/processed/reddit/labeled_comments.csv"
TOTAL_CSV="data/processed/reddit/cleaned_comments.csv"
OUTPUT_CSV="data/processed/reddit/unlabeled_comments.csv"

def generate_csv():
    df = pd.read_csv(LABELED_CSV)
    df2 = pd.read_csv(TOTAL_CSV)


    labeledDict=df.to_dict(orient='list')
    totalDict=df2.to_dict(orient='list')

    outputDict={}
    
    totalLength=0
    for x in totalDict:
        outputDict[x]=[]
        totalLength=len(totalDict[x]) #same based on format of .to_dict

    for i in range(totalLength):
        if totalDict['comment_id'][i] not in labeledDict['comment_id']: #inefficient but suffices
            
            for x in totalDict:
                outputDict[x].append(totalDict[x][i])
    
    newPd=pd.DataFrame(outputDict)
    newPd.to_csv(OUTPUT_CSV, index=False)

if __name__ == "__main__":
    generate_csv()