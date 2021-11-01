import pandas as pd
drugREsFinderdf=pd.read_csv(r"C:\Users\Katie\Documents\lab_1 BMIN 521 NLP\drugintweetslab\drugREsFinder.tsv", sep='\t')
print(drugREsFinderdf.shape)

FN_q5=drugREsFinderdf[(drugREsFinderdf['label']>0) & (drugREsFinderdf['prediction']<1)]
print(FN_q5.shape)

random_50_FN_q5=FN_q5.sample(50)
print(random_50_FN_q5.shape)
random_50_FN_q5.to_csv("random_50_FN_q5new.csv")

FP_q5=drugREsFinderdf[(drugREsFinderdf['label']<1)&(drugREsFinderdf['prediction']>0)]
print(FP_q5.shape)
#random_50_FP_q5=FP_q5.sample(30)
#print(random_50_FP_q5.shape)
#random_50_FP_q5.to_csv("random_50_FP_q5new.csv")
FP_q5.to_csv("FP_q5new.csv")