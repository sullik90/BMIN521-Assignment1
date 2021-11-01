import pandas as pd
drugExactMatchdf=pd.read_csv(r"C:\Users\Katie\Documents\lab_1 BMIN 521 NLP\drugintweetslab\drugExactMatch.tsv", sep='\t')
print(drugExactMatchdf.shape)

FP=drugExactMatchdf[(drugExactMatchdf['label']<1) & (drugExactMatchdf['prediction']>0)]
print(FP.shape)
FP.to_csv("FP_question1.csv")

FN=drugExactMatchdf[(drugExactMatchdf['label']>0) & (drugExactMatchdf['prediction']<1)]
print(FN.shape)
FN.to_csv("FN_question1.csv")

random_100_FN=FN.sample(100)
print(random_100_FN.shape)
random_100_FN.to_csv("random_100_FN_question1.csv")

TP=drugExactMatchdf[(drugExactMatchdf['label']>0) & (drugExactMatchdf['prediction']>0)]
print(TP.shape)

random_100_TP=TP.sample(100)
print(random_100_TP.shape)
random_100_TP.to_csv("random_100_TP_question1.csv")

