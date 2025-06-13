import pandas as pd
import os


def read_and_validate_excel(file_path: str) -> pd.DataFrame:
    """
    读取Excel文件并验证是否包含所需的列（year, week, genotype, ILIp）。
    如果文件存在且包含所需列，则返回DataFrame，否则抛出异常。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        raise ValueError(f"无法读取Excel文件: {file_path}. 错误信息: {e}")

    required_columns = ['year', 'week', 'city_name', 'ILIp']
    if not all(column in df.columns for column in required_columns):
        missing_columns = [col for col in required_columns if col not in df.columns]
        raise ValueError(f"Excel文件缺少必要的列: {missing_columns}")

    return df


def calculate_weekly_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算每个genotype在每个week的ILIp均值。
    """
    # 计算每个genotype在每个week的ILIp均值
    weekly_averages = df.groupby(['city_name', 'week'])['ILIp'].mean().reset_index()

    # 按照40-52周和1-39周排列
    weekly_averages['week'] = weekly_averages['week'].astype(int)
    weekly_averages = weekly_averages.sort_values(by=['city_name', 'week'])

    return weekly_averages


def rearrange_weeks(df: pd.DataFrame) -> pd.DataFrame:
    """
    将week列按照40-52周和1-39周排列，并添加从1开始的索引列。
    """
    # 分离40-52周和1-39周
    df_40_52 = df[(df['week'] >= 40) & (df['week'] <= 52)]
    df_1_39 = df[(df['week'] >= 1) & (df['week'] <= 39)]

    # 合并两个DataFrame
    df_rearranged = pd.concat([df_40_52, df_1_39]).reset_index(drop=True)

    # 添加从1开始的索引列
    df_rearranged.insert(0, 'index', range(1, 1 + len(df_rearranged)))

    return df_rearranged


def rearrange_weeks_by_genotype(df: pd.DataFrame) -> pd.DataFrame:
    """
    对每个genotype独立地将week列按照40-52周和1-39周排列。
    """
    # 对每个genotype独立处理
    genotype_groups = df.groupby('city_name')
    rearranged_dfs = []

    for genotype, group in genotype_groups:
        # 分离40-52周和1-39周
        df_40_52 = group[(group['week'] >= 40) & (group['week'] <= 52)]
        df_1_39 = group[(group['week'] >= 1) & (group['week'] <= 39)]

        # 合并两个DataFrame
        df_rearranged = pd.concat([df_40_52, df_1_39]).reset_index(drop=True)

        # 添加从1开始的索引列
        df_rearranged.insert(0, 'index', range(1, 1 + len(df_rearranged)))

        # 保存处理后的DataFrame
        rearranged_dfs.append(df_rearranged)

    # 合并所有处理后的DataFrame
    final_df = pd.concat(rearranged_dfs).reset_index(drop=True)

    return final_df

def read_and_validate_cityILI_excel(file_path: str) -> pd.DataFrame:
    try:
        df = read_and_validate_excel(file_path)

        # 计算每个genotype在每个week的ILIp均值
        weekly_averages = calculate_weekly_averages(df)

        # 对每个genotype独立地将week列按照40-52周和1-39周排列
        weekly_averages_rearranged = rearrange_weeks_by_genotype(weekly_averages)

        return weekly_averages_rearranged

    except Exception as e:
        print(f"发生错误: {e}")


if __name__ == "__main__":
    excel_file_path = "cityILIp.xlsx"

    try:
        df = read_and_validate_cityILI_excel(excel_file_path)
        print("文件验证通过，数据已成功加载为DataFrame。")
        print(df.head())  # 打印前几行数据以确认
    except Exception as e:
        print(f"发生错误: {e}")