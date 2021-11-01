import pandas as pd
drugTokenMatchdf=pd.read_csv(r"C:\Users\Katie\Documents\lab_1 BMIN 521 NLP\drugintweetslab\drugTokenMatch.tsv", sep='\t')

#Identify 28 false positives
FP_Q2=drugTokenMatchdf[(drugTokenMatchdf['label']<1) & (drugTokenMatchdf['prediction']>0)]
print(FP_Q2.shape)
FP_Q2.to_csv("FP_question2.csv")

#Randomly select 100 FN
FN_Q2=drugTokenMatchdf[(drugTokenMatchdf['label']>0) & (drugTokenMatchdf['prediction']<1)]
print(FN_Q2.shape)
random_100_FN_Q2=FN_Q2.sample(100)
print(random_100_FN_Q2.shape)
random_100_FN_Q2.to_csv("random_100_FN_question2.csv")
#Randomly select 100 TP
TP_Q2=drugTokenMatchdf[(drugTokenMatchdf['label']>0) & (drugTokenMatchdf['prediction']>0)]
print(TP_Q2.shape)
random_100_TP_Q2=TP_Q2.sample(100)
print(random_100_TP_Q2.shape)
random_100_TP_Q2.to_csv("random_100_TP_question2.csv")

