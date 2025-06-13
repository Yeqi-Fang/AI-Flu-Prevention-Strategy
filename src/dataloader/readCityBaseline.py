import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
sys.path.append(project_root)

# 使用绝对导入
from src.env.city_env import City

def read_cities_from_excel(file_path):
    # 读取 Excel 文件
    df = pd.read_excel(file_path)

    # 检查必要的列是否存在
    required_columns = ['city_id', 'name', 'city_type', 'population', 'squared_km2',
                        'economy_base', 'bed_base', 'longitude', 'latitude']
    if not all(column in df.columns for column in required_columns):
        raise ValueError(f"Excel 文件中缺少必要的列: {required_columns}")

    # 初始化城市对象
    cities = []
    for _, row in df.iterrows():
        city = City(
            city_id=row['city_id'],
            name=row['name'],
            city_type=row['city_type'],
            population=row['population'],
            squared_km2=row['squared_km2'],
            economy_base=row['economy_base'],
            bed_base=row['bed_base'],
            longitude=row['longitude'],
            latitude=row['latitude']
        )
        cities.append(city)

    return cities


def calculate_migration_matrix(cities, alpha=0.5, beta=0.5, r=1000):
    # 提取城市的人口和坐标
    populations = np.array([city.population for city in cities])
    coordinates = np.array([[city.longitude, city.latitude] for city in cities])

    # 计算城市之间的欧几里得距离矩阵
    distance_matrix = squareform(pdist(coordinates, 'euclidean'))

    # 计算迁移矩阵
    migration_matrix = np.zeros((len(cities), len(cities)))
    for i in range(len(cities)):
        for j in range(len(cities)):
            if i != j:
                p_i = populations[i]
                p_j = populations[j]
                d_ij = distance_matrix[i, j]
                migration_matrix[i, j] =0.00001 * (p_i ** alpha) * (p_j ** beta) * np.exp(-d_ij / r)

    return migration_matrix

if __name__ == "__main__":
    # 示例：读取 Excel 文件并生成迁移矩阵
    file_path = 'cities_baseline_data.xlsx'  # 替换为你的 Excel 文件路径
    cities = read_cities_from_excel(file_path)
    migration_matrix = calculate_migration_matrix(cities)

    # 打印迁移矩阵
    print("迁移矩阵：")
    print(migration_matrix)