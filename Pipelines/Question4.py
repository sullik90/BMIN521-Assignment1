import pandas as pd
drugREsFinderdf=pd.read_csv(r"C:\Users\Katie\Documents\lab_1 BMIN 521 NLP\drugintweetslab\drugREsFinder.tsv", sep='\t')
print(drugREsFinderdf.shape)

FN_q4=drugREsFinderdf[(drugREsFinderdf['label']>0) & (drugREsFinderdf['prediction']<1)]
print(FN_q4.shape)

random_50_FN_q4=FN_q4.sample(50)
print(random_50_FN_q4.shape)
random_50_FN_q4.to_csv("random_50_FN_q4.csv")

FP_q4=drugREsFinderdf[(drugREsFinderdf['label']<1)&(drugREsFinderdf['prediction']>0)]
print(FP_q4.shape)
random_50_FP_q4=FP_q4.sample(50)
print(random_50_FP_q4.shape)
random_50_FP_q4.to_csv("random_50_FP_q4.csv")
