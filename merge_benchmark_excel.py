import os

import pandas as pd

# 文件列表
files = [
    "benchmark_cr_cls2_rolling_2013,2014,2015_2016_0701.xlsx",
    "benchmark_cr_cls2_rolling_2012,2013,2014_2016_0701.xlsx",
    "benchmark_cr_cls2_rolling_2012,2013,2014_2015_0701.xlsx",
    "benchmark_cr_cls2_rolling_2011,2012,2013_2016_0701.xlsx",
    "benchmark_cr_cls2_rolling_2011,2012,2013_2015_0701.xlsx",
    "benchmark_cr_cls2_rolling_2011,2012,2013_2014_0701.xlsx",
    "benchmark_cr_cls2_rolling_2010,2011,2012_2016_0701.xlsx",
    "benchmark_cr_cls2_rolling_2010,2011,2012_2015_0701.xlsx",
    "benchmark_cr_cls2_rolling_2010,2011,2012_2014_0701.xlsx",
    "benchmark_cr_cls2_rolling_2010,2011,2012_2013_0701.xlsx",
    "benchmark_cr2_cls2_rolling_2013,2014,2015_2016_0701.xlsx",
    "benchmark_cr2_cls2_rolling_2012,2013,2014_2016_0701.xlsx",
    "benchmark_cr2_cls2_rolling_2012,2013,2014_2015_0701.xlsx",
    "benchmark_cr2_cls2_rolling_2011,2012,2013_2016_0701.xlsx",
    "benchmark_cr2_cls2_rolling_2011,2012,2013_2015_0701.xlsx",
    "benchmark_cr2_cls2_rolling_2011,2012,2013_2014_0701.xlsx",
    "benchmark_cr2_cls2_rolling_2010,2011,2012_2016_0701.xlsx",
    "benchmark_cr2_cls2_rolling_2010,2011,2012_2015_0701.xlsx",
    "benchmark_cr2_cls2_rolling_2010,2011,2012_2014_0701.xlsx",
    "benchmark_cr2_cls2_rolling_2010,2011,2012_2013_0701.xlsx",
]


# 创建一个空的DataFrame，用于存储合并后的数据
merged_data = pd.DataFrame()

# 遍历文件名列表，逐个处理文件
for file_name in files:
    # 读取当前文件的数据
    data = pd.read_excel(f'./excel/{file_name}')

    # 将文件名作为第一行
    data_with_filename = pd.DataFrame({0: [file_name]}).append(data, ignore_index=True)

    # 添加两个空行
    empty_rows = pd.DataFrame(columns=data.columns, index=range(2))

    # 将当前文件的数据添加到merged_data中
    merged_data = merged_data.append(data_with_filename, ignore_index=True)
    merged_data = merged_data.append(empty_rows, ignore_index=True)

# 将合并后的数据保存到一个新的Excel文件中
merged_data.to_excel("./excel/merged_benchmark_rolling_0701.xlsx", index=False, engine='openpyxl')
