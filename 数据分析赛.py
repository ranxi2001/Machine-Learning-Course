import pandas as pd
# 读取Excel文件并转换为pandas数据框
from matplotlib import pyplot as plt
import seaborn as sns

df = pd.read_excel('数据8 顾客购买数据.xlsx')
# 对相同的用户ID进行分组并合并产品名称
df_grouped = df.groupby('用户ID')['产品名称'].apply(list).reset_index()
# 显示处理后的结果
print(df_grouped)
df_grouped.to_excel("关联规则.xlsx")

import pandas as pd
# 读取Excel表格
df = pd.read_excel('关联规则.xlsx', index_col=0)
# 获取包含列表的行
contains_list = df[df['产品名称'].str.contains('\[.*\]')]
# 筛选出列表中不止一个对象的行
multiple_items = contains_list[contains_list['产品名称'].str.count(',') > 0]
# 输出结果
multiple_items

import ast
import pandas as pd
# 将字符串列表转换为实际列表
dataset = df['产品名称'].apply(ast.literal_eval).tolist()
# 查看处理后的 DataFrame
print(dataset)

# 关联规则分析
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
# 将“产品名称”列转换为适合进行关联规则分析的格式
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
clean_df = pd.DataFrame(te_ary, columns=te.columns_)
# 使用Apriori算法获取频繁项集
freq_items = apriori(clean_df, min_support=0.1, use_colnames=True)
# 获取关联规则
rules = association_rules(freq_items, metric="confidence", min_threshold=0.1)
# 打印得出的规则
rules

rules_df = pd.DataFrame(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
rules_df.columns = ['antecedents', 'consequents', 'support', 'confidence', 'lift']
#保存数据预处理好的表格
rules_df.to_excel("rules.xlsx")
# 创建一个新的数据帧，以antecedent支持为索引和列
support_df = rules_df.pivot(index='antecedents', columns='consequents', values='confidence')

# 将缺少值替换为0
support_df.fillna(0, inplace=True)

# 将名称重命名为产品的名称
support_df.rename(columns=lambda x: str(x), inplace=True)
support_df.rename(index=lambda x: str(x), inplace=True)

# 创建热图
plt.figure(figsize=(15, 12))
# 设置中文字体
plt.rcParams['font.family'] = 'SimHei'
sns.heatmap(support_df, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title('Antecedent Support Heatmap')
plt.show()


