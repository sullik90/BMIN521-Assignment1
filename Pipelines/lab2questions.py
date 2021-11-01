
import pandas as pd
BayesFeaturesdf=pd.read_csv(r"C:\Users\Katie\Documents\DrugInTweetsLab2 (1)\DrugInTweetsLab2\NaiveBayesPrediction.csv", sep='\t')
print(BayesFeaturesdf.shape)
BayesFeaturesdf_top = BayesFeaturesdf.head()
print(BayesFeaturesdf_top)
print(list(BayesFeaturesdf.columns.values))
labels_list = BayesFeaturesdf['label'].tolist()
print(labels_list)
predictions_list=BayesFeaturesdf['prediction'].tolist()
print(predictions_list)


FP=BayesFeaturesdf[(BayesFeaturesdf['label']<1) & (BayesFeaturesdf['prediction']>0)]
print(FP.shape)
FP_subset=FP[["text","label","prediction","drugs_predicted"]]
FP.to_csv("FP_lab2.csv")
FP_subset.to_csv("FP_subset_lab2.csv")

FN=BayesFeaturesdf[(BayesFeaturesdf['label']>0) & (BayesFeaturesdf['prediction']<1)]
print(FN.shape)
FN.to_csv("FN_lab2.csv")
FN_subset=FN[["text","label","prediction","drugs_predicted"]]
FN_subset.to_csv("FN_subset_lab2.csv")



