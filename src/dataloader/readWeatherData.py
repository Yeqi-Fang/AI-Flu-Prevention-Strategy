import os
import pandas as pd
import numpy as np
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
sys.path.append(project_root)

from src.dataloader.readCityBaseline import read_cities_from_excel


def read_weather_data(file_name,cities):
    if os.path.exists(file_name):
        df = pd.read_excel(file_name)
    elif os.path.exists(os.path.join('..', file_name)):
        df = pd.read_excel(os.path.join('..', file_name))
    weather_data = {}
    for city in cities:
        city_name = city.name
        day_list = []
        for _, row in df.iterrows():
            T = row['tmean']  # 温度（摄氏度）
            P = row['pressure']  # 压力（百帕）
            RH = row['humidity']  # 相对湿度（百分比）
            e_s = 6.112 * np.exp((17.67 * T) / (T + 243.5))
            e_a = (RH / 100) * e_s
            AH = (e_a * 2.1674) / P
            day_list.append({
                'ah': AH,
                'tmean':T
            })
        weather_data[city_name] = day_list
    return weather_data

if __name__ == "__main__":

    file_path = 'cities_baseline_data.xlsx'
    excel_file_path = "weatherSichuan.xlsx"
    cities = read_cities_from_excel(file_path)
    # 示例：指定Excel文件路径



    try:
        df = read_weather_data(excel_file_path,cities)
        print("文件验证通过，数据已成功加载为DataFrame。")
        print(df)  # 打印前几行数据以确认
    except Exception as e:
        print(f"发生错误: {e}")