import numpy as np
import pandas as pd

class City:
    def __init__(self, city_id,name, city_type, population, squared_km2,
                 economy_base, bed_base,longitude=0,latitude=0):
        # 环境与传播参数
        self.beta0_range = (0.198440563, 0.227805819)
        self.gamma0_range = (0.196290662, 0.255311277)
        self.kappa0_range = (0.542206504, 0.711468987)
        self.epsilon0_range = (0.1, 0.175161062)

        self.ah_a = 0.8947065898679676
        self.ah_c = 0.02890372039629151
        self.ah_score = 61.70051869599825
        self.tmp_c = 18.57608998569208
        self.tmp_score = 0.3827618737395368
        self.tmp_a = -0.21151705827632955
        self.rho_score = 0.06454950351559771
        self.rho_c = 1.4711351635054517
        self.rho_a = 0.9717321214728396
        self.eta_score = 0.2

        self.epsilon_0 = np.random.uniform(low=self.epsilon0_range[0], high=self.epsilon0_range[1])

        # 传播动力学参数
        self.beta0 = np.random.uniform(low=self.beta0_range[0], high=self.beta0_range[1])
        self.gamma0 = np.random.uniform(low=self.gamma0_range[0], high=self.gamma0_range[1])
        self.lambda0 = 0.0656252708302435
        self.kappa0 = np.random.uniform(low=self.kappa0_range[0], high=self.kappa0_range[1])
        self.d = 0.1


        self.city_id = city_id
        self.name = name
        self.type = city_type
        self.population = population
        self.squared_km2 = squared_km2
        self.economy_base = economy_base
        self.bed_base = bed_base
        self.longitude = longitude
        self.latitude = latitude
        self.h1n1ratio = 1

        # SEIQR模型初始状态
        self.S = None
        self.E = None
        self.I = None
        self.Q = 0
        self.R = 0

        self.rho = population / squared_km2 * 10  # 人口密度，乘10仅作缩放

    def set_h1n1ratio(self, ratio):
        self.h1n1ratio = ratio

    def set_initial_conditions(self, init_exposed_rate=0.002, init_infected_rate=0.001):
        """
        设置初始暴露率和感染率
        """
        self.init_exposed_rate = init_exposed_rate
        self.init_infected_rate = init_infected_rate
        self.S = self.population * (1 - init_exposed_rate - init_infected_rate) * 1e4
        self.E = self.population * init_exposed_rate * 1e4
        self.I = self.population * init_infected_rate * 1e4

    def set_base_params(self,beta0,gamma0,kappa0,epsilon0):
        self.beta0_range = (beta0*0.8,beta0*1.2)
        self.gamma0_range = (gamma0*0.8,gamma0*1.2)
        self.kappa0_range = (kappa0*0.8,kappa0*1.2)
        self.epsilon0_range = (epsilon0*0.8,epsilon0*1.2)
        self.epsilon_0 = np.random.uniform(low=self.epsilon0_range[0], high=self.epsilon0_range[1])
        self.beta0 = np.random.uniform(low=self.beta0_range[0], high=self.beta0_range[1])
        self.gamma0 = np.random.uniform(low=self.gamma0_range[0], high=self.gamma0_range[1])
        self.kappa0 = np.random.uniform(low=self.kappa0_range[0], high=self.kappa0_range[1])

    def set_weather_params(self,ah_a0,ah_c0,ah_score0,temp_a0,temp_c0,temp_score0):
        self.ah_a = ah_a0
        self.ah_c = ah_c0
        self.ah_score = ah_score0
        self.tmp_c = temp_c0
        self.tmp_score = temp_score0
        self.tmp_a = temp_a0

    def set_city_params(self, rho_a0, rho_c0, rho_score0,lambda0):
        self.rho_a = rho_a0
        self.rho_c = rho_c0
        self.rho_score = rho_score0
        self.lambda0 = lambda0



    def total_population(self):
        return self.S + self.E + self.I + self.Q + self.R

    def update(self, weather, params,estimation=0):
        """
        根据每日天气和策略参数更新SEIQR状态
        weather: {'ah': float, 'tmean': float}
        params: {'mask': float, 'home': float, 'med': float}
        """
        ah = weather['ah']
        tmp = weather['tmean']
        m = params['mask']
        med = params['med']
        home = params['home']

        # 计算环境因子
        H_factor = (self.ah_score * (ah - self.ah_c) ** 2 + self.ah_a) * ((self.tmp_c / tmp) ** self.tmp_score + self.tmp_a)
        density_factor = self.rho_a + self.rho_score * (self.rho / self.rho_c)
        mask_factor = 1 - self.eta_score * m * (4 - self.type)

        # 计算传播率
        beta = self.beta0 * H_factor * density_factor * mask_factor

        if estimation == 1:
            beta = self.beta0

        if estimation == 2:
            beta = self.beta0 * H_factor

        if estimation == 3:
            beta = self.beta0 * H_factor * density_factor

        # 计算医疗资源影响
        B_effective = self.bed_base * (1 + med * 15) * self.type  # 医疗资源影响恢复率
        I_val = max(self.I, 1e-9)
        resource_ratio = min(0.3, self.lambda0 * B_effective / I_val)
        gamma = self.gamma0 * (1 + resource_ratio)

        if (estimation == 1) or (estimation == 2):
            gamma = self.gamma0

        # 隔离强度
        delta = home * (4 - self.type) * 0.2  # home参数直接影响隔离率

        # 微分方程
        N = self.total_population()
        dS = -beta * self.S * (self.I + self.kappa0 * self.E) / (N + 1e-9)
        dE = beta * self.S * (self.I + self.kappa0 * self.E) / (N + 1e-9) - self.epsilon_0 * self.E
        dI = self.epsilon_0 * self.E - (gamma + delta) * self.I
        dQ = delta * self.I - self.d * self.Q
        dR = gamma * self.I + self.d * self.Q

        # 更新状态（非负约束）
        self.S = max(self.S + dS, 0)
        self.E = max(self.E + dE, 0)
        self.I = max(self.I + dI, 0)
        self.Q = max(self.Q + dQ, 0)
        self.R = max(self.R + dR, 0)


def get_strategy_params(strategy, cities):
    """给定策略编号，生成不同的干预参数 (mask, home, med) """
    params = {}

    if strategy == 1:  # 无干预
        for city in cities:
            params[city.name] = {'mask': 0, 'home': 0, 'med': 0}

    elif strategy == 2:  # 动态响应（示例，阈值可调）
        strategy_value = [0.0153, 0.0127, 0.0110]
        for city in cities:
            inf_ratio = city.I / (city.total_population())
            baseline = strategy_value[city.type - 1]*1.75

            if inf_ratio/baseline > 4 :
                params[city.name] = {'mask': 0.3, 'home': 0.2, 'med': 0.3}
            elif inf_ratio/baseline > 3 :
                params[city.name] = {'mask': 0.2, 'home': 0.1, 'med': 0.2}
            elif inf_ratio/baseline > 2 :
                params[city.name] = {'mask': 0.1, 'home': 0.0, 'med': 0.1}
            else:
                params[city.name] = {'mask': 0, 'home': 0, 'med': 0}

    return params


def do_migration(cities, migration_matrix):
    """人口迁移逻辑（每周执行一次）"""
    num_cities = len(cities)

    for i in range(num_cities):
        for j in range(i + 1, num_cities):
            # 计算基础迁移率
            exchange_rate = min(migration_matrix[i, j], migration_matrix[j, i])
            base_i = cities[i].total_population() * exchange_rate
            base_j = cities[j].total_population() * exchange_rate
            exchange_base = min(base_i, base_j)

            # 双向迁移
            for src, dest in [(i, j), (j, i)]:
                total = cities[src].total_population()
                if total <= 0:
                    continue

                # 计算迁移比例
                ratios = [
                    cities[src].S / total,
                    cities[src].E / total,
                    cities[src].I / total,
                    cities[src].R / total
                ]

                mig_total = exchange_base
                S_mig = mig_total * ratios[0]
                E_mig = mig_total * ratios[1]
                I_mig = mig_total * ratios[2]
                R_mig = mig_total * ratios[3]

                # 更新人口
                cities[src].S -= S_mig
                cities[src].E -= E_mig
                cities[src].I -= I_mig
                cities[src].R -= R_mig

                cities[dest].S += S_mig
                cities[dest].E += E_mig
                cities[dest].I += I_mig
                cities[dest].R += R_mig


def calculate_economic_loss(city, mask, home, med):
    """经济影响计算"""
    coefficients = {
        1: (15,0.20,0.5),  # 平原城市：居家影响大
        2: (10,0.22,0.5),  # 丘陵城市：均衡影响
        3: (6,0.24,0.6)  # 高原城市：医疗投入影响大
    }
    c1, c2, c3 = coefficients[city.type]
    # 每日损失，这里简单按 city.economy_base 比例缩放
    return city.economy_base * (c1 * home + c2 * mask + c3 * med) / 7  # 按天计算