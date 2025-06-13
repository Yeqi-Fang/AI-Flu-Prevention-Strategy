import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
sys.path.append(project_root)

# 使用绝对导入
from src.env.city_env import City, get_strategy_params
from src.dataloader.readWeatherData import read_weather_data
from src.dataloader.readCityBaseline import read_cities_from_excel
from src.dataloader.readCityILIp import read_and_validate_cityILI_excel

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 指定默认字体

scale = 10000  # 缩放因子，用于将预测结果转换为万人


class BayesianCity(City):
    """扩展City类，新增需要优化的参数"""

    def __init__(self, rho_score, rho_c, rho_a, lambda0, **kwargs):
        super().__init__(**kwargs)
        self.rho_score = rho_score
        self.rho_a = rho_a
        self.rho_c = rho_c
        self.lambda0 = lambda0
        self.infection_history = []  # 用于保存模拟结果


def run_city_bayesian_optimization(citiesconfig, weather_data, ili_df, weeks=30, scale_factor=1.946997710649481):
    """
    执行城市参数贝叶斯优化过程，返回最佳参数和优化结果

    参数:
    citiesconfig -- 城市配置列表
    weather_data -- 天气数据字典
    ili_df -- ILI数据DataFrame
    weeks -- 模拟周数 (默认30)
    scale_factor -- 缩放因子 (默认1.947)

    返回:
    dict -- 包含优化结果、图表和数据的字典
    """

    # 定义内部函数
    def run_simulation(rho_score, rho_c, rho_a, lambda0):
        """运行模拟并返回各城市预测结果"""
        cities = []
        for cfg in citiesconfig:
            city = BayesianCity(
                rho_score=rho_score,
                rho_c=rho_c,
                rho_a=rho_a,
                lambda0=lambda0,
                name=cfg.name,
                city_type=cfg.type,
                population=cfg.population,
                squared_km2=cfg.squared_km2,
                economy_base=cfg.economy_base,
                bed_base=cfg.bed_base,
                longitude=cfg.longitude,
                latitude=cfg.latitude,
                city_id=cfg.city_id
            )
            city.infection_history = []
            cities.append(city)

        for city in cities:
            city_weather_list = weather_data[city.name]
            pop = city.population
            for week in range(weeks):
                if week == 0:
                    city.set_initial_conditions()
                params = get_strategy_params(1, cities)  # 固定使用策略1（无干预）
                daily_weather = city_weather_list[week]
                for day in range(7):
                    city.update(daily_weather, params[city.name], 3)
                city.infection_history.append(city.I / pop / scale_factor)  # 转换为万人

        return {city.name: np.array(city.infection_history) for city in cities}

    def evaluate_predictions(predicted, real_data):
        """评估预测结果，计算综合损失"""
        city_names = {city.name for city in citiesconfig}
        city_names = list(city_names)
        all_mse = []
        all_peak_diff = []
        all_peak_height_diff = []

        # 按城市分组真实数据
        real_data = {city: group['ILIp'].values for city, group in real_data.groupby('city_name')}

        for city_name in city_names:
            pred_series = np.array(predicted[city_name])
            real_series = np.array(real_data[city_name])

            # 1) 计算MSE (从第5周开始)
            mse = mean_squared_error(real_series[5:30], pred_series[5:30])
            all_mse.append(mse)

            # 2) 计算达峰时间差异
            real_peak = np.argmax(real_series[5:])
            pred_peak = np.argmax(pred_series[5:])
            peak_diff = abs(pred_peak - real_peak)
            all_peak_diff.append(peak_diff)

            # 3) 计算峰值高度差异
            real_peek = np.max(real_series[5:])
            pred_peek = np.max(pred_series[5:])
            peak_height_diff = abs(pred_peek - real_peek)
            all_peak_height_diff.append(peak_height_diff)

        # 计算平均误差
        avg_mse = np.mean(all_mse)
        avg_peak_diff = np.mean(all_peak_diff)
        avg_peak_height_diff = np.mean(all_peak_height_diff)

        # 组合误差 (可根据需要调整权重)
        total_loss = 10000 * avg_mse + avg_peak_diff + 10000 * avg_peak_height_diff
        return total_loss

    def black_box_function(rho_score, rho_c, rho_a, lambda0):
        """贝叶斯优化的目标函数"""
        predicted = run_simulation(rho_score, rho_c, rho_a, lambda0)
        total_loss = evaluate_predictions(predicted, ili_df)
        return -total_loss  # 取负值用于最大化

    # 定义参数搜索空间
    pbounds = {
        'rho_score': (0.01, 5),
        'rho_c': (0.3, 2.0),
        'rho_a': (0.6, 1.2),
        'lambda0': (0.005, 1),
    }

    # 创建并运行优化器
    optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        random_state=42,
        allow_duplicate_points=True
    )

    optimizer.maximize(init_points=10, n_iter=100)
    best_params = optimizer.max['params']

    def calculate_parameter_confidence(optimizer, n_bootstrap=100, alpha=0.95):
        """使用自助法计算参数估计的置信区间"""
        res = optimizer.res
        params = []
        targets = []
        for r in res:
            params.append([r['params'][k] for k in sorted(r['params'].keys())])
            targets.append(r['target'])

        params = np.array(params)
        targets = np.array(targets)
        param_names = sorted(optimizer.max['params'].keys())

        bootstrap_params = []
        for _ in range(n_bootstrap):
            idx = np.random.choice(len(params), size=len(params), replace=True)
            best_idx = np.argmax(targets[idx])
            bootstrap_params.append(params[idx][best_idx])

        bootstrap_params = np.array(bootstrap_params)
        param_ci = {}
        for i, name in enumerate(param_names):
            values = bootstrap_params[:, i]
            lower = np.percentile(values, (1 - alpha) / 2 * 100)
            upper = np.percentile(values, (1 + alpha) / 2 * 100)
            param_ci[name] = {'mean': np.mean(values), 'lower': lower, 'upper': upper}

        return param_ci

    # 计算参数置信区间
    param_ci = calculate_parameter_confidence(optimizer)

    # 准备结果数据
    params_df = pd.DataFrame([best_params])
    param_ci_df = pd.DataFrame(param_ci).T.reset_index()
    param_ci_df.columns = ['Parameter', 'Mean', 'CI_Lower', 'CI_Upper']

    # 使用最佳参数运行模拟，获取预测结果
    best_prediction = run_simulation(**best_params)
    results_data = []
    for city_name in best_prediction:
        for week, value in enumerate(best_prediction[city_name]):
            results_data.append({
                'City': city_name,
                'Week': week,
                'Predicted': value,
                'Observed': ili_df[ili_df['city_name'] == city_name]['ILIp'].values[week] / scale
            })
    results_df = pd.DataFrame(results_data)

    # 创建图表
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(range(len(optimizer.res)), [-res['target'] for res in optimizer.res], 'o-')
    ax1.set_title('贝叶斯优化过程（损失函数变化）')
    ax1.set_xlabel('迭代次数')
    ax1.set_ylabel('损失函数')
    ax1.grid(True)

    # 创建预测结果对比图
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    for city_name in best_prediction:
        city_data = results_df[results_df['City'] == city_name]
        ax2.plot(city_data['Week'], city_data['Observed'], '--', label=f'{city_name} 真实值')
        ax2.plot(city_data['Week'], city_data['Predicted'], '-', label=f'{city_name} 预测值')
    ax2.set_title('各城市预测结果对比')
    ax2.set_xlabel('周数')
    ax2.set_ylabel('感染率')
    ax2.legend()
    ax2.grid(True)

    # 更新城市参数
    for city in citiesconfig:
        city.set_city_params(
            rho_a0=best_params['rho_a'],
            rho_score0=best_params['rho_score'],
            rho_c0=best_params['rho_c'],
            lambda0=best_params['lambda0']
        )

    # 返回结果
    return {
        'optimizer': optimizer,
        'best_params': best_params,
        'parameter_ci': param_ci,
        'simulation_results': results_df,
        'optimization_plot': fig1,
        'prediction_plot': fig2,
        'parameter_df': params_df,
        'parameter_ci_df': param_ci_df
    }


# 使用示例
if __name__ == "__main__":
    # 加载数据
    ili_file_path = "../dataloader/cityILIp.xlsx"
    city_file_path = "../dataloader/cities_baseline_data.xlsx"
    weather_file_path = "../dataloader/weatherSichuan.xlsx"

    citiesconfig = read_cities_from_excel(city_file_path)
    weather_data = read_weather_data(weather_file_path, citiesconfig)
    ili_df = read_and_validate_cityILI_excel(ili_file_path)

    # 执行优化
    results = run_city_bayesian_optimization(citiesconfig, weather_data, ili_df)

    # 显示图表
    results['optimization_plot'].show()
    results['prediction_plot'].show()

    # 保存结果到Excel
    output_file = 'city_simulation_results.xlsx'
    with pd.ExcelWriter(output_file) as writer:
        results['simulation_results'].to_excel(writer, sheet_name='Simulation_Results', index=False)
        results['parameter_df'].to_excel(writer, sheet_name='Best_Parameters', index=False)
        results['parameter_ci_df'].to_excel(writer, sheet_name='Parameter_CI', index=False)

    print("城市参数优化完成！最佳参数:")
    print(results['best_params'])