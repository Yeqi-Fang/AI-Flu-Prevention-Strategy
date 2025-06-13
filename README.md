# 流感智能防控系统 Web 应用

## 项目结构

```
your_project/
│
├── src/                    # 原有的源代码目录
│   ├── typer/
│   ├── dataloader/
│   ├── params/
│   ├── rl/
│   └── env/
│
├── app.py                  # Flask主应用文件
│
├── templates/              # HTML模板目录
│   ├── base.html          # 基础模板
│   ├── index.html         # 主页
│   ├── typing.html        # 分型分析页面
│   ├── params.html        # 参数拟合页面
│   ├── rl_train.html      # RL训练页面
│   └── decision.html      # 智慧决策页面
│
├── static/                 # 静态资源目录
│   ├── css/
│   │   └── style.css      # 样式文件
│   ├── js/
│   │   └── main.js        # JavaScript文件
│   └── results/           # 生成的结果文件（自动创建）
│
├── uploads/               # 上传文件目录（自动创建）
│
└── saved_models/          # 保存的模型文件（RL训练后生成）
```

## 安装依赖

在原有的 `requirements.txt` 基础上，添加 Flask 相关依赖：

```bash
flask>=2.3.0
werkzeug>=2.3.0
```

或者单独安装：

```bash
pip install flask werkzeug
```

## 使用说明

### 1. 启动应用

```bash
python app.py
```

应用将在 http://localhost:5000 启动

### 2. 功能流程

#### 2.1 分型分析
1. 点击左侧"分型结果"
2. 上传 `sample_data.xlsx` 文件
3. 点击"分析"查看结果

#### 2.2 参数拟合
1. 点击"传播动力学参数拟合"
2. 上传三个文件：
   - weatherSichuan.xlsx
   - cities_baseline_data.xlsx
   - all_ILI.xlsx
3. 点击"分析"开始基础参数估计
4. 依次完成三个阶段：
   - 基础参数
   - 气候驱动
   - 城市驱动（需额外上传 city_ILIp.xlsx）

#### 2.3 RL训练
1. 点击"RL训练"
2. 上传训练集数据（重命名为带 _train 后缀）
3. 点击"开始训练"

#### 2.4 智慧决策
1. 点击"智慧决策"
2. 上传测试集数据（重命名为带 _test 后缀）
3. 点击"获取智慧决策"
4. 查看策略图表并下载结果

## 文件格式要求

### sample_data.xlsx
| sample_id | city_id | seq |
|-----------|---------|-----|
| S001      | C01     | ATCG... |

### cities_baseline_data.xlsx
| city_id | name | city_type | population | squared_km2 | economy_base | bed_base | longitude | latitude |
|---------|------|-----------|------------|-------------|--------------|----------|-----------|----------|
| C01     | 成都  | 1         | 2093.8     | 14335       | 19000        | 12.5     | 104.06    | 30.67    |

### weatherSichuan.xlsx
| date       | tmean | humidity | pressure |
|------------|-------|----------|----------|
| 2024-01-01 | 15.2  | 75       | 1013     |

### all_ILI.xlsx / city_ILIp.xlsx
| year | week | genotype/city_name | ILIp |
|------|------|--------------------|------|
| 2024 | 1    | H1N1/成都          | 0.05 |

## 注意事项

1. 确保所有数据文件格式正确
2. 文件大小限制为 16MB
3. 训练过程可能需要较长时间
4. 建议使用 Chrome 或 Firefox 浏览器
5. 如果遇到中文显示问题，确保系统安装了中文字体

## 常见问题

### Q: 页面显示 "分析失败"
A: 检查上传的文件格式是否正确，确保包含所有必需的列

### Q: RL训练时间过长
A: 可以在 `app.py` 中调整训练轮数（epochs）参数

### Q: 图表中文显示为方块
A: 需要安装中文字体，或修改 matplotlib 字体设置

### Q: 上传文件失败
A: 检查文件大小是否超过 16MB，或文件扩展名是否为 .xlsx/.xls

## 开发说明

- Flask 应用使用 session 存储用户状态
- 文件上传保存在 `uploads/` 目录
- 生成的结果保存在 `static/results/` 目录
- 可以通过修改 `app.config` 调整配置

## 部署建议

生产环境部署时：
1. 设置 `app.secret_key` 为安全的随机值
2. 关闭 debug 模式
3. 使用 WSGI 服务器（如 Gunicorn）
4. 配置反向代理（如 Nginx）
5. 设置合适的文件上传限制