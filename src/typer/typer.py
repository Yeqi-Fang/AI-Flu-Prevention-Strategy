from Bio import SeqIO
import re
import pandas as pd
from typing import Dict
import sys
import os

# 获取当前文件的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 添加项目根目录到 sys.path
project_root = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
sys.path.append(project_root)

# 使用绝对导入
from src.dataloader.readSampleData import read_and_validate_sample_excel

# 缓存解析结果
reference_db_cache = {}


def parse_reference_library(fasta_path: str) -> Dict[str, Dict]:
    """
    读取参考库FASTA文件，提取ID、序列、分型（如H1N1）、病毒类型（A/B/...）
    """
    if fasta_path in reference_db_cache:
        return reference_db_cache[fasta_path]

    reference_db = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        header = record.description
        sequence = str(record.seq).upper()
        # 提取 H/N 信息
        hn_match = re.search(r'\(([^()]*H\d+N\d+)\)', header)
        subtype = hn_match.group(1) if hn_match else "Unknown"
        # 提取病毒类型（Influenza A/B/...）
        virus_type_match = re.search(r'Influenza\s([ABCD])', header)
        virus_type = virus_type_match.group(1) if virus_type_match else "Unknown"

        reference_db[record.id] = {
            "header": header,
            "subtype": subtype,
            "virus_type": virus_type,
            "sequence": sequence
        }

    reference_db_cache[fasta_path] = reference_db
    return reference_db


def compute_coverage(query: str, reference: str) -> float:
    """
    计算序列匹配覆盖度（碱基逐一匹配比例）
    """
    match_len = min(len(query), len(reference))
    matches = sum(1 for a, b in zip(query[:match_len], reference[:match_len]) if a == b)
    return round(matches / match_len, 4) if match_len > 0 else 0.0


def identify_sequence_type(query_sequence: str, reference_fasta: str) -> Dict:
    """
    主函数：输入待测序列 + 参考库FASTA路径 → 返回最相似的参考条目信息
    """
    query = query_sequence.upper().replace('\n', '').replace(' ', '')

    ref_db = parse_reference_library(reference_fasta)
    best_match = None
    best_coverage = 0.0

    for ref_id, ref in ref_db.items():
        coverage = compute_coverage(query, ref["sequence"])
        if coverage > best_coverage:
            best_coverage = coverage
            best_match = {
                "ref_id": ref_id,
                "subtype": ref["subtype"],
                "virus_type": ref["virus_type"],
                "coverage": coverage,
                "header": ref["header"]
            }

    return best_match if best_match else {"error": "No match found."}


def identify_sequences(df: pd.DataFrame, seq_column: str, reference_fasta: str) -> pd.DataFrame:
    """
    批量处理DataFrame中的序列，返回分型识别结果
    """
    results = []
    for index, row in df.iterrows():
        query_sequence = row[seq_column]
        result = identify_sequence_type(query_sequence, reference_fasta)
        if "error" not in result:
            results.append({
                "sample_id": row["sample_id"],
                "city_id": row["city_id"],
                "subtype": result["subtype"],
                "coverage": result["coverage"]
            })

    # 将结果转换为DataFrame
    results_df = pd.DataFrame(results)
    return results_df


def get_subtype_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    对每个城市的所有分型进行频率统计，输出每个城市各个分型的比例（concentration）
    """
    grouped = df.groupby(["city_id", "subtype"]).size().reset_index(name="count")
    total_counts = grouped.groupby("city_id")["count"].transform("sum")
    grouped["concentration"] = grouped["count"] / total_counts
    return grouped.sort_values(["city_id", "concentration"], ascending=[True, False])



if __name__ == "__main__":
    # 示例：从DataFrame中读取序列
    excel_file_path = "../dataloader/sample_data.xlsx"
    fasta_path = "sequences.fasta"

    try:
        df = read_and_validate_sample_excel(excel_file_path)

        results_df = identify_sequences(df, "seq", fasta_path)
        print(results_df)

        subtype_dist_df = get_subtype_distribution(results_df)

        print("=== 每个城市各个分型的占比 ===")
        print(subtype_dist_df)
    except Exception as e:
        print(f"发生错误: {e}")