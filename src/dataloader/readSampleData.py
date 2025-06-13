import pandas as pd
import os

def read_and_validate_sample_excel(file_path: str) -> pd.DataFrame:
    """
    读取Excel文件并验证是否包含所需的列（sample_id, city_id, seq）。
    如果文件存在且包含所需列，则返回DataFrame，否则抛出异常。
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    # 读取Excel文件
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        raise ValueError(f"无法读取Excel文件: {file_path}. 错误信息: {e}")

    # 检查是否包含所需的列
    required_columns = ['sample_id', 'city_id', 'seq']
    if not all(column in df.columns for column in required_columns):
        missing_columns = [col for col in required_columns if col not in df.columns]
        raise ValueError(f"Excel文件缺少必要的列: {missing_columns}")

    return df

if __name__ == "__main__":
    # 示例：指定Excel文件路径
    excel_file_path = "sample_data.xlsx"

    try:
        df = read_and_validate_sample_excel(excel_file_path)
        print("文件验证通过，数据已成功加载为DataFrame。")
        print(df.head())  # 打印前几行数据以确认
    except Exception as e:
        print(f"发生错误: {e}")