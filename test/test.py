# test1
# import pandas as pd
# df = pd.read_excel('../data/elements.xlsx',sheet_name='Sheet2')
# print(df['formula'])
# print(df['MagpieData minimum AtomicWeight'])
# print({df['formula'][i]: df['MagpieData minimum AtomicWeight'][i] for i in range(len(df))})


from base_function import evaluation_top_val_by_percentile

score = evaluation_top_val_by_percentile([1,2,3,4,5,6,7,8,9,10],[9,12,13,14,15])
print(score)