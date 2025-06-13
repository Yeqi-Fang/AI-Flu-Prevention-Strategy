# 全流程main程序 - 流感传播建模与优化系统
# 主要模块：分型分析、参数估计、强化学习训练与仿真

# ===================== 导入各模块功能函数 =====================
# 分型分析模块 (typer)
from typer.typer import read_and_validate_sample_excel  # 读取并验证样本数据
from typer.typer import identify_sequences  # 识别病毒序列分型
from typer.typer import get_subtype_distribution  # 计算分型分布

# 数据加载模块 (dataloader)
from dataloader.readWeatherData import read_weather_data  # 读取气象数据
from dataloader.readGenotypeILI import read_and_validate_genotypeILI_excel  # 读取基因型ILI数据
from dataloader.readCityBaseline import read_cities_from_excel  # 读取城市基线数据
from dataloader.readCityILIp import read_and_validate_cityILI_excel  # 读取城市ILI数据

# 参数估计模块 (params)
from params.baseParamsEstimation import run_base_bayesian_optimization  # 基础参数贝叶斯优化
from params.weatherParamsEstimation import run_weather_bayesian_optimization  # 气象参数优化
from params.cityParamsEstimation import run_city_bayesian_optimization  # 城市参数优化

# 强化学习模块 (rl)
from rl.trainDQN import train_models  # 训练DQN模型
from rl.testDQN import run_simulations  # 运行仿真测试

# ===================== 模块1：病毒分型分析 =====================
"""
功能：分析病毒样本数据，识别序列分型并统计各城市分型分布
输入：
  excel_file_path: 样本数据Excel路径（输入）
  fasta_path: 各个病毒分型参考文件（不动）
输出：
  results_df: 包含分型识别结果的数据帧
  subtype_dist_df: 各城市分型分布统计
"""
excel_file_path = "dataloader/sample_data.xlsx"
fasta_path = "typer/sequences.fasta"

try:
    # 读取并验证样本数据
    df = read_and_validate_sample_excel(excel_file_path)

    # 识别病毒序列分型 (输入: 样本数据框, 序列列名, 输出路径)
    results_df = identify_sequences(df, "seq", fasta_path)
    print(results_df)

    # 计算各城市分型分布 (输入: 分型结果数据框)
    subtype_dist_df = get_subtype_distribution(results_df)

    print("=== 各城市病毒分型占比分布 ===")
    print(subtype_dist_df)
except Exception as e:
    print(f"分型分析模块错误: {e}")

# ===================== 模块2：参数估计 =====================
"""
执行顺序要求：必须先估计基础参数，再气象参数，最后城市参数
输入数据文件：
  ili_file_path: 全国ILI数据
  city_file_path: 城市基线数据
  weather_file_path: 气象数据
  city_ili_file_path: 分城市ILI数据
"""
ili_file_path = "dataloader/allILI.xlsx"
city_file_path = "dataloader/cities_baseline_data.xlsx"
weather_file_path = "dataloader/weatherSichuan.xlsx"
city_ili_file_path = "dataloader/cityILIp.xlsx"

# 加载基础数据
citiesconfig = read_cities_from_excel(city_file_path)  # 读取城市配置
weather_data = read_weather_data(weather_file_path, citiesconfig)  # 读取气象数据
ili_df = read_and_validate_genotypeILI_excel(ili_file_path)  # 读取全国ILI数据

# 2.1 基础参数估计 (beta, gamma, kappa, epsilon)
"""
功能：估计传播模型基本参数
输入：
  citiesconfig: 城市配置对象列表
  weather_data: 气象数据
  ili_df: 全国ILI数据
输出：
  base_results: 包含参数数据帧的字典
"""
base_results = run_base_bayesian_optimization(citiesconfig, weather_data, ili_df)
params_df = base_results['parameter_df']

# 将估计参数设置到城市对象中
for city in citiesconfig:
    city.set_base_params(
        beta0=params_df['beta0'],
        gamma0=params_df['gamma0'],
        kappa0=params_df['kappa0'],
        epsilon0=params_df['epsilon']
    )

# 2.2 气象参数估计 (温度/湿度相关参数)
"""
功能：估计气象因素对传播的影响参数
输入：同上
输出：weather_results字典
"""
weather_results = run_weather_bayesian_optimization(citiesconfig, weather_data, ili_df)
params_df = weather_results['parameter_df']

for city in citiesconfig:
    city.set_weather_params(
        temp_a0=params_df['tmp_a'],
        temp_score0=params_df['tmp_score'],
        temp_c0=params_df['tmp_c'],
        ah_score0=params_df['ah_score'],
        ah_a0=params_df['ah_a'],
        ah_c0=params_df['ah_c']
    )

# 2.3 城市参数估计 (人口流动相关参数)
"""
功能：估计城市间迁移参数
输入：
  citiesconfig: 带基础参数的城市配置
  weather_data: 气象数据
  city_ili_file_path: 分城市ILI数据
"""
ili_df = read_and_validate_cityILI_excel(city_ili_file_path)  # 重新加载城市级ILI数据
city_results = run_city_bayesian_optimization(citiesconfig, weather_data, ili_df)
params_df = city_results['parameter_df']

for city in citiesconfig:
    city.set_city_params(
        rho_a0=params_df['rho_a'],
        rho_score0=params_df['rho_score'],
        rho_c0=params_df['rho_c'],
        lambda0=params_df['lambda0']
    )

# ===================== 模块3：强化学习 =====================
# 3.1 DQN模型训练
"""
功能：训练深度Q网络生成防控策略
输入：
  citiesconfig: 带参数的城市配置
  weather_data: 气象数据
  actionfile: 预定义防控策略文件（不动）
  epochs: 训练轮次
  batch_size: 批处理大小（可调整）
  use_gpu: 是否使用GPU训练
输出：保存训练好的模型
"""
train_models(
    cities=citiesconfig,
    weather_data=weather_data,
    actionfile='rl/actions.xlsx',
    epochs=200,
    models_path='rl/saved_models',
    batch_size=128,  # 可调整批处理大小
    use_gpu=True     # 启用GPU训练
)

# 3.2 防控策略仿真测试
"""
功能：评估防控策略效果
输入：
  citiesconfig: 城市配置
  weather_data: 气象数据
  num_simulations: 仿真次数
  actionfile: 策略文件（不动）
  model_path: 训练模型路径（不动）
输出：
  result: 包含结果数据框的字典（主要用result_df也就是模拟结果）
"""
result = run_simulations(
    cities=citiesconfig,
    weather_data=weather_data,
    num_simulations=100,
    actionfile='rl/actions.xlsx',
    model_path='rl/saved_models'
)

# 输出仿真结果
print("\n仿真结果:")
resultdf = result['result_df']
print(resultdf)