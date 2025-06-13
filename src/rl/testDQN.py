import numpy as np
import torch
import pandas as pd
import os
import sys
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
sys.path.append(project_root)

# 使用绝对导入
from src.dataloader.readWeatherData import read_weather_data
from src.dataloader.readCityBaseline import read_cities_from_excel, calculate_migration_matrix
from src.rl.modelDQN import SEIQREnv, DQNAgent


def load_trained_agents(env, cities, model_path):
    """
    为每个城市加载已经训练好的模型权重，
    返回一个按城市顺序排列的 DQNAgent list。
    """
    agents = []
    for i, cfg in enumerate(cities):
        agent = DQNAgent(env)
        city_type = cfg.type
        # 加载已经保存的策略网络参数
        model_file = f"{model_path}/policy_net_{city_type}.pth"
        agent.policy_net.load_state_dict(torch.load(model_file))
        agent.target_net.load_state_dict(agent.policy_net.state_dict())
        # 测试阶段通常不需要探索，直接用贪心策略
        agent.epsilon = 0.0
        agents.append(agent)
    return agents


def run_evaluation(env, agents, max_weeks=52):
    """
    在给定环境和若干城市智能体(agents)上跑一个完整的流感季(默认 52 周)。
    返回一个保存测试记录的 DataFrame，列包含:
      [city, week, mask, home, med, infection, economic, infection_rate, r].
    """
    state = env.reset()

    results = []
    done = False
    week = 0

    while not done and week < max_weeks:
        # 每个城市都用同一个 state(拼接了所有城市信息)
        action_indices = []
        for i, agent in enumerate(agents):
            action_idx = agent.select_action(state, training=False)
            action_indices.append(action_idx)

        next_state, reward, done, info = env.step(action_indices)

        # 将当前周每个城市的结果记录下来
        for i, city in enumerate(env.cities):
            city_name = city.name
            mask_val = env.actions[action_indices[i]]['mask']
            home_val = env.actions[action_indices[i]]['home']
            med_val = env.actions[action_indices[i]]['med']
            infection = city.I
            economic_loss = env.economy_history[city.name][-1]

            results.append({
                'city': city_name,
                'week': week,
                'mask': mask_val,
                'home': home_val,
                'med': med_val,
                'infection': infection,
                'economic': economic_loss,
                'infection_rate': infection / city.population / 10000,
                'r': city.R / city.population / 10000
            })

        state = next_state
        week += 1

    df_result = pd.DataFrame(results)
    return df_result


def run_simulations(cities, weather_data, model_path, num_simulations=1000,
                    output_file="simulation_results.xlsx",actionfile='actions.xlsx'):
    """
    运行多次模拟并保存结果到Excel文件

    参数:
        baseline (str): 城市基线数据文件路径
        weather (str): 天气数据文件路径
        model_path (str): 训练模型保存路径
        num_simulations (int): 模拟次数
        output_file (str): 输出Excel文件名

    返回:
        dict: 包含操作结果
            success (bool): 是否成功
            output_path (str): 输出文件路径
            error (str): 错误信息
    """
    try:
        # 1. 读取数据
        migration_matrix = calculate_migration_matrix(cities)

        # 2. 固定随机种子，生成随机数种子
        np.random.seed(123)
        torch.manual_seed(123)
        seeds = np.random.randint(0, 2 ** 31 - 1, size=num_simulations)

        # 3. 收集所有模拟结果
        all_results = []

        for idx, seed in enumerate(seeds):
            if (idx + 1) % 100 == 0:
                print(f"Processing simulation {idx + 1}/{num_simulations}")

            # 设置当前模拟的随机种子
            np.random.seed(seed)
            torch.manual_seed(seed)

            # 创建环境
            env = SEIQREnv(
                cities=cities,
                migration_matrix=migration_matrix,
                action_file=actionfile,
                time_window=5,
                max_weeks=52,
                weather_data=weather_data
            )

            # 加载训练好的智能体
            agents = load_trained_agents(env, cities, model_path)

            # 运行评估
            df_run = run_evaluation(env, agents, max_weeks=52)

            # 添加模拟标识信息
            df_run['simulation_index'] = idx
            df_run['seed'] = seed

            all_results.append(df_run)

        # 4. 合并结果并保存
        df_all = pd.concat(all_results, ignore_index=True)
        df_all.to_excel(output_file, index=False)

        return {
            "success": True,
            "output_path": os.path.abspath(output_file),
            "message": f"Successfully completed {num_simulations} simulations",
            "result_df":df_all
        }

    except Exception as e:
        return {
            "success": False,
            "output_path": None,
            "error": f"Simulation failed: {str(e)}"
        }


if __name__ == "__main__":
    baseline = '../dataloader/cities_baseline_data.xlsx'
    weather = '../dataloader/weatherSichuan.xlsx'
    actionfile = 'actions.xlsx'
    cities = read_cities_from_excel(baseline)
    weather_data = read_weather_data(weather, cities)
    # 示例调用
    result = run_simulations(
        cities=cities,
        weather_data=weather_data,
        num_simulations=100,
        model_path='saved_models'
    )

    print("\nSimulation Result:")
    resultdf = result['result_df']
    print(resultdf)