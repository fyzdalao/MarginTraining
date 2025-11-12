import pandas as pd
import numpy as np
from scipy.stats import rankdata
import statsmodels.api as sm
from statsmodels.formula.api import ols

# =============================
# 1. 构造数据
# =============================
data = {
    'model': list('abcdefghi'),
    'none': [22.60, 23.60, 25.00, 20.73, 26.20, 23.87, 16.93, 15.67, 21.87],
    'A':    [26.13, 29.47, 34.40, 30.73, 29.27, 30.33, 22.40, 31.93, 29.20],
    'B':    [26.33, 31.00, 30.00, 26.60, 27.87, 39.60, 29.60, 22.47, 28.20],
    'A+B':  [40.47, 42.60, 51.67, 54.27, 57.13, 63.47, 31.20, 32.73, 43.00]
}

records = []
for i, row in pd.DataFrame(data).iterrows():
    records += [
        {'model': row['model'], 'A': 0, 'B': 0, 'acc': row['none']},
        {'model': row['model'], 'A': 1, 'B': 0, 'acc': row['A']},
        {'model': row['model'], 'A': 0, 'B': 1, 'acc': row['B']},
        {'model': row['model'], 'A': 1, 'B': 1, 'acc': row['A+B']},
    ]
df = pd.DataFrame(records)

# =============================
# 2. 定义辅助函数
# =============================
def get_means(df):
    return {
        'grand': df['acc'].mean(),
        'A': df.groupby('A')['acc'].mean().to_dict(),
        'B': df.groupby('B')['acc'].mean().to_dict(),
        'AB': df.groupby(['A','B'])['acc'].mean().to_dict(),
        'model': df.groupby('model')['acc'].mean().to_dict()
    }

def align(df, effect):
    means = get_means(df)
    aligned = df.copy()
    aligned['y'] = df['acc']
    for i, row in aligned.iterrows():
        A, B, M = row['A'], row['B'], row['model']
        # 对齐
        if effect == 'A':
            aligned.loc[i, 'y'] -= (means['B'][B] - means['grand'])
            aligned.loc[i, 'y'] -= (means['AB'][(A,B)] - means['A'][A] - means['B'][B] + means['grand'])
            aligned.loc[i, 'y'] -= (means['model'][M] - means['grand'])
        elif effect == 'B':
            aligned.loc[i, 'y'] -= (means['A'][A] - means['grand'])
            aligned.loc[i, 'y'] -= (means['AB'][(A,B)] - means['A'][A] - means['B'][B] + means['grand'])
            aligned.loc[i, 'y'] -= (means['model'][M] - means['grand'])
        elif effect == 'AB':
            aligned.loc[i, 'y'] -= (means['A'][A] - means['grand'])
            aligned.loc[i, 'y'] -= (means['B'][B] - means['grand'])
            aligned.loc[i, 'y'] -= (means['model'][M] - means['grand'])
        elif effect == 'model':
            aligned.loc[i, 'y'] -= (means['A'][A] - means['grand'])
            aligned.loc[i, 'y'] -= (means['B'][B] - means['grand'])
            aligned.loc[i, 'y'] -= (means['AB'][(A,B)] - means['A'][A] - means['B'][B] + means['grand'])
    return aligned

def run_art_anova(df, effect):
    aligned = align(df, effect)
    aligned['rank'] = rankdata(aligned['y'])
    model = ols('rank ~ C(A)*C(B) + C(model)', data=aligned).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    if effect == 'A':
        return anova_table.loc['C(A)']
    elif effect == 'B':
        return anova_table.loc['C(B)']
    elif effect == 'AB':
        return anova_table.loc['C(A):C(B)']
    elif effect == 'model':
        return anova_table.loc['C(model)']

# =============================
# 3. 运行四个检验
# =============================
effects = ['A','B','AB','model']
rows = []
for e in effects:
    res = run_art_anova(df, e)
    rows.append({
        '效应': e,
        '自由度': int(res['df']),
        'F 值': res['F'],
        'p 值': res['PR(>F)']
    })

result = pd.DataFrame(rows)
print(result)
