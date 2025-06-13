import numpy as np
import torch
import os
import random
from tqdm import tqdm
import sys
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
sys.path.append(project_root)

# 使用绝对导入
from src.env.city_env import City, do_migration, calculate_economic_loss
from src.dataloader.readWeatherData import read_weather_data
from src.dataloader.readCityBaseline import read_cities_from_excel, calculate_migration_matrix
from src.rl.modelDQN import SEIQREnv, DQNAgent


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def save_models(agents, city_types, path):
    os.makedirs(path, exist_ok=True)
    for agent, c_type in zip(agents, city_types):
        torch.save(agent.policy_net.state_dict(), f'{path}/policy_net_{c_type}.pth')
    return os.path.abspath(path)


def train_models(cities, weather_data, actionfile, epochs, models_path='saved_models', 
                batch_size=64, use_gpu=True):
    """
    训练DQN模型的函数
    参数:
        cities: 城市配置列表
        weather_data: 天气数据
        actionfile: 动作文件路径
        epochs (int): 训练轮数（随机种子数量）
        models_path (str): 模型保存路径
        batch_size (int): 批处理大小，默认64
        use_gpu (bool): 是否使用GPU，默认True

    返回:
        dict: 包含训练状态和模型路径的字典
            success (bool): 训练是否成功
            model_path (str): 模型保存路径（成功时）
            error (str): 错误信息（失败时）
    """
    try:
        # 检查GPU可用性
        device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        print(f"Training on device: {device}")
        if use_gpu and not torch.cuda.is_available():
            print("Warning: GPU requested but not available, using CPU instead")

        # 读取数据
        migration_matrix = calculate_migration_matrix(cities)

        # 初始化环境和智能体
        env = SEIQREnv(
            cities=cities,
            migration_matrix=migration_matrix,
            action_file=actionfile,
            time_window=5,
            max_weeks=52,
            weather_data=weather_data
        )
        
        # 创建agents时传入batch_size和use_gpu参数
        agents = [DQNAgent(env, batch_size=batch_size, use_gpu=use_gpu) for _ in range(len(cities))]

        # 生成随机种子
        random.seed(42)
        random_seeds = [random.randint(0, 10000) for _ in range(epochs)]

        # 训练循环
        episodes = 5
        for seed in tqdm(random_seeds, desc="Training progress", unit="seed"):
            set_seed(seed)
            state = env.reset()
            done = False
            while not done:
                action_indices = [agent.select_action(state) for agent in agents]
                next_state, reward, done, info = env.step(action_indices)
                for i, agent in enumerate(agents):
                    agent.remember(state, action_indices[i], info['city_rewards'][i], next_state, done)
                    agent.learn()
                state = next_state

        # 保存模型
        model_path = save_models(agents, [c.type for c in cities], models_path)

        return {
            "success": True,
            "model_path": model_path,
            "message": f"Models saved to {model_path} after {epochs} epochs (batch_size={batch_size}, device={device})"
        }

    except Exception as e:
        return {
            "success": False,
            "model_path": None,
            "error": f"Training failed: {str(e)}"
        }


if __name__ == "__main__":
    # 示例调用
    baseline = '../dataloader/cities_baseline_data.xlsx'
    weather = '../dataloader/weatherSichuan.xlsx'
    actionfile = 'actions.xlsx'
    cities = read_cities_from_excel(baseline)
    weather_data = read_weather_data(weather,cities)
    epochs = 200

    result = train_models(cities, weather_data,actionfile, epochs)
    print("\nTraining Result:")
    print(json.dumps(result, indent=2))