import pandas as pd
df = pd.read_excel('../data/elements.xlsx',sheet_name='Sheet2')
print(df['formula'])
print(df['MagpieData minimum AtomicWeight'])
print({df['formula'][i]: df['MagpieData minimum AtomicWeight'][i] for i in range(len(df))})