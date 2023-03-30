import pandas as pd
import requests

url = "https://github.com/lusccc/aazz/raw/main/%E6%95%B0%E6%8D%AE%E6%B1%87%E6%80%BB%E8%A1%A8(1).xlsx"

# 下载Excel表格并读取数据
excel_data = pd.read_excel('数据汇总表(1).xlsx', sheet_name="指标1-温室气体排放率")

# 选择“水”矩阵
water_matrix = excel_data.loc[excel_data['Unnamed: 0'] == '水', :]

# 打印“水”矩阵
print(water_matrix)
