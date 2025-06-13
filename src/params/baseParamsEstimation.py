import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from scipy import stats
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
sys.path.append(project_root)

# 使用绝对导入
from src.env.city_env import City,get_strategy_params
from src.dataloader.readWeatherData import read_weather_data
from src.dataloader.readCityBaseline import read_cities_from_excel
from src.dataloader.readGenotypeILI import read_and_validate_genotypeILI_excel

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 指定默认字体

scale = 10000  # 缩放因子，用于将预测结果转换为万人

class BayesianCity(City):
    def __init__(self, beta0, gamma0, epsilon0, kappa0,
                  ** kwargs):
        super().__init__(
         ** kwargs
        )
        # 动态更新传播动力学参数
        self.beta0 = beta0
        self.gamma0 = gamma0
        self.epsilon_0 = epsilon0
        self.kappa_0 = kappa0


def run_base_bayesian_optimization(citiesconfig, weather_data, ili_df, weeks=30):
    """
    执行贝叶斯优化过程，返回最佳参数和优化结果

    参数:
    citiesconfig -- 城市配置列表
    weather_data -- 天气数据字典
    ili_df -- ILI数据DataFrame
    weeks -- 模拟周数 (默认30)

    返回:
    dict -- 包含优化结果、图表和数据的字典
    """

    # 定义内部函数
    def run_simulation(beta0, gamma0, epsilon, kappa0, scale_factor):
        """运行模拟并返回预测结果"""
        cities = []
        for cfg in citiesconfig:
            city = BayesianCity(
                beta0=beta0,
                gamma0=gamma0,
                epsilon0=epsilon,
                kappa0=kappa0,
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
            cities.append(city)

        predicted = []
        for week in range(weeks):
            params = get_strategy_params(1, cities)  # 固定使用策略1（无干预）
            if week == 0:
                for city in cities:
                    city.set_initial_conditions()
            for day in range(7):
                # 每个城市执行每日更新
                for city in cities:
                    city_weather_list = weather_data[city.name]
                    daily_weather = city_weather_list[week]
                    city.update(daily_weather, params[city.name], 1)

            current_I = (cities[0].I + cities[1].I + cities[2].I) / (
                    cities[0].total_population() + cities[1].total_population() + cities[
                2].total_population()) / scale_factor
            predicted.append(current_I)

        return predicted

    def calculate_confidence_intervals(params, n_runs=30, alpha=0.95):
        """计算预测结果的置信区间"""
        all_predictions = []
        for _ in range(n_runs):
            pred = run_simulation(**params)
            all_predictions.append(pred)

        all_predictions = np.array(all_predictions)
        mean_pred = np.mean(all_predictions, axis=0)
        std_pred = np.std(all_predictions, axis=0)

        n = all_predictions.shape[0]
        se = std_pred / np.sqrt(n)
        t_value = stats.t.ppf((1 + alpha) / 2, n - 1)
        ci_lower = mean_pred - t_value * se
        ci_upper = mean_pred + t_value * se

        return mean_pred, ci_lower, ci_upper

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

    def objective_function(beta0, gamma0, epsilon, kappa0, scale_factor):
        """贝叶斯优化的目标函数"""
        predicted = run_simulation(beta0, gamma0, epsilon, kappa0, scale_factor)
        true_values = ili_df[ili_df['genotype'] == 'H1N1']['ILIp'].values[0:30] / scale

        def calculate_peak_error(true, pred):
            true_peak_idx = np.argmax(true)
            pred_peak_idx = np.argmax(pred)
            height_error = abs(true[true_peak_idx] - pred[pred_peak_idx])
            position_error = abs(true_peak_idx - pred_peak_idx)
            return height_error, position_error

        def calculate_combined_error(true, pred):
            mse = mean_squared_error(true, pred)
            height_error, position_error = calculate_peak_error(true, pred)
            combined_error = 10000 * mse + position_error
            return combined_error

        start = 5
        error = calculate_combined_error(true_values[start:], predicted[start:])
        return -error  # 负误差用于最大化

    # 定义参数搜索空间
    pbounds = {
        'beta0': (0.1, 0.3),
        'gamma0': (0.1, 0.3),
        'epsilon': (0.1, 0.3),
        'scale_factor': (1.7, 2.8),
        'kappa0': (0.2, 0.8)
    }

    # 创建并运行优化器
    optimizer = BayesianOptimization(
        f=objective_function,
        pbounds=pbounds,
        random_state=42,
        allow_duplicate_points=True
    )

    optimizer.maximize(init_points=10, n_iter=100)
    best_params = optimizer.max['params']

    # 计算置信区间
    mean_pred, ci_lower, ci_upper = calculate_confidence_intervals(best_params)
    param_ci = calculate_parameter_confidence(optimizer)

    # 准备结果数据
    results_df = pd.DataFrame({
        'Week': range(len(mean_pred)),
        'True_Values': ili_df[ili_df['genotype'] == 'H1N1']['ILIp'].values[0:30] / scale,
        'Predicted_Mean': mean_pred,
        'CI_Lower': ci_lower,
        'CI_Upper': ci_upper
    })

    param_ci_df = pd.DataFrame(param_ci).T.reset_index()
    param_ci_df.columns = ['Parameter', 'Mean', 'CI_Lower', 'CI_Upper']
    params_df = pd.DataFrame([best_params])

    # 创建图表
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(range(len(optimizer.res)), [-res['target'] for res in optimizer.res], 'o-')
    ax1.set_title('贝叶斯优化过程（误差变化）')
    ax1.set_xlabel('迭代次数')
    ax1.set_ylabel('误差')
    ax1.grid(True)

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    true_values = ili_df[ili_df['genotype'] == 'H1N1']['ILIp'].values[0:30] / scale
    ax2.plot(true_values, 'b-', label='真实值')
    ax2.plot(mean_pred, 'r-', label='预测均值')
    ax2.fill_between(range(len(mean_pred)), ci_lower, ci_upper, color='pink', alpha=0.3, label='95%置信区间')
    ax2.set_title('模型预测与真实值对比（带置信区间）')
    ax2.set_xlabel('周数')
    ax2.set_ylabel('感染率')
    ax2.legend()
    ax2.grid(True)

    # 更新城市参数
    for city in citiesconfig:
        city.set_base_params(
            beta0=best_params['beta0'],
            gamma0=best_params['gamma0'],
            kappa0=best_params['kappa0'],
            epsilon0=best_params['epsilon']
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


if __name__ == "__main__":
    ili_file_path = "../dataloader/allILI.xlsx"
    city_file_path = "../dataloader/cities_baseline_data.xlsx"
    weather_file_path = "../dataloader/weatherSichuan.xlsx"
    citiesconfig = read_cities_from_excel(city_file_path)
    weather_data = read_weather_data(weather_file_path, citiesconfig)

    ili_df = read_and_validate_genotypeILI_excel(ili_file_path)

    results = run_base_bayesian_optimization(citiesconfig, weather_data, ili_df)

    # 保存结果到Excel
    with pd.ExcelWriter('SEIQR_simulation_results.xlsx') as writer:
        results['simulation_results'].to_excel(writer, sheet_name='Simulation_Results', index=False)
        results['parameter_df'].to_excel(writer, sheet_name='Best_Parameters', index=False)
        results['parameter_ci_df'].to_excel(writer, sheet_name='Parameter_CI', index=False)

    print("优化完成！最佳参数:")
    print(results['best_params'])