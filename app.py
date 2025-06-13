import os
import sys
from flask import Flask, render_template, request, jsonify, send_file, session
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import json
import time
from io import BytesIO
import base64
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import uuid

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# 导入项目模块
from src.typer.typer import identify_sequences, get_subtype_distribution
from src.dataloader.readWeatherData import read_weather_data
from src.dataloader.readGenotypeILI import read_and_validate_genotypeILI_excel
from src.dataloader.readCityBaseline import read_cities_from_excel
from src.dataloader.readCityILIp import read_and_validate_cityILI_excel
from src.params.baseParamsEstimation import run_base_bayesian_optimization
from src.params.weatherParamsEstimation import run_weather_bayesian_optimization
from src.params.cityParamsEstimation import run_city_bayesian_optimization
from src.rl.trainDQN import train_models
from src.rl.testDQN import run_simulations

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# 确保上传文件夹存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/results', exist_ok=True)

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {'xlsx', 'xls'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def fig_to_base64(fig):
    """将matplotlib图形转换为base64字符串"""
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{image_base64}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/typing')
def typing():
    return render_template('typing.html')

@app.route('/params')
def params():
    return render_template('params.html')

@app.route('/rl_train')
def rl_train():
    return render_template('rl_train.html')

@app.route('/decision')
def decision():
    return render_template('decision.html')

@app.route('/upload_typing', methods=['POST'])
def upload_typing():
    """处理分型分析文件上传"""
    if 'file' not in request.files:
        return jsonify({'error': '没有文件上传'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # 读取并验证文件
            df = pd.read_excel(filepath)
            # 保存文件路径到session
            session['typing_file'] = filepath
            return jsonify({'success': True, 'message': '文件上传成功'})
        except Exception as e:
            return jsonify({'error': f'文件读取失败: {str(e)}'}), 400
    
    return jsonify({'error': '文件格式不支持'}), 400

@app.route('/analyze_typing', methods=['POST'])
def analyze_typing():
    """执行分型分析"""
    if 'typing_file' not in session:
        return jsonify({'error': '请先上传文件'}), 400
    
    try:
        filepath = session['typing_file']
        df = pd.read_excel(filepath)
        
        # 执行分型分析
        fasta_path = os.path.join('src', 'typer', 'sequences.fasta')
        results_df = identify_sequences(df, "seq", fasta_path)
        subtype_dist_df = get_subtype_distribution(results_df)
        
        # 转换为HTML表格
        results_html = results_df.to_html(classes='table table-striped', index=False)
        dist_html = subtype_dist_df.to_html(classes='table table-striped', index=False)
        
        return jsonify({
            'success': True,
            'results_table': results_html,
            'distribution_table': dist_html
        })
    except Exception as e:
        return jsonify({'error': f'分析失败: {str(e)}'}), 400

@app.route('/upload_params', methods=['POST'])
def upload_params():
    """处理参数估计文件上传"""
    required_files = ['weather_file', 'cities_file', 'ili_file']
    uploaded_files = {}
    
    for file_key in required_files:
        if file_key not in request.files:
            return jsonify({'error': f'缺少文件: {file_key}'}), 400
        
        file = request.files[file_key]
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': f'文件 {file_key} 无效'}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_key}_{filename}")
        file.save(filepath)
        uploaded_files[file_key] = filepath
    
    # 保存文件路径到session
    session['params_files'] = uploaded_files
    session['params_stage'] = 'base'  # 初始阶段
    
    return jsonify({'success': True, 'message': '文件上传成功'})

@app.route('/analyze_params/<stage>', methods=['POST'])
def analyze_params(stage):
    """执行参数估计的不同阶段"""
    if 'params_files' not in session:
        return jsonify({'error': '请先上传文件'}), 400
    
    try:
        files = session['params_files']
        
        # 加载数据
        citiesconfig = read_cities_from_excel(files['cities_file'])
        weather_data = read_weather_data(files['weather_file'], citiesconfig)
        
        if stage == 'base':
            # 基础参数估计
            ili_df = read_and_validate_genotypeILI_excel(files['ili_file'])
            results = run_base_bayesian_optimization(citiesconfig, weather_data, ili_df)
            
            # 保存结果到session
            session['base_params'] = results['best_params']
            session['citiesconfig'] = [vars(city) for city in citiesconfig]  # 序列化城市配置
            
            # 转换图形为base64
            opt_plot = fig_to_base64(results['optimization_plot'])
            pred_plot = fig_to_base64(results['prediction_plot'])
            
            return jsonify({
                'success': True,
                'optimization_plot': opt_plot,
                'prediction_plot': pred_plot,
                'parameters': results['best_params']
            })
            
        elif stage == 'weather':
            # 天气参数估计
            ili_df = read_and_validate_genotypeILI_excel(files['ili_file'])
            
            # 应用基础参数
            if 'base_params' in session:
                base_params = session['base_params']
                for city in citiesconfig:
                    city.set_base_params(
                        beta0=base_params['beta0'],
                        gamma0=base_params['gamma0'],
                        kappa0=base_params['kappa0'],
                        epsilon0=base_params['epsilon']
                    )
            
            results = run_weather_bayesian_optimization(citiesconfig, weather_data, ili_df)
            
            # 保存结果
            session['weather_params'] = results['best_params']
            
            # 转换图形
            opt_plot = fig_to_base64(results['optimization_plot'])
            pred_plot = fig_to_base64(results['prediction_plot'])
            
            return jsonify({
                'success': True,
                'optimization_plot': opt_plot,
                'prediction_plot': pred_plot,
                'parameters': results['best_params']
            })
            
        elif stage == 'city':
            # 需要额外的city_ili文件
            if 'city_ili_file' not in request.files:
                return jsonify({'error': '请上传城市ILI数据文件'}), 400
            
            file = request.files['city_ili_file']
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # 应用之前的参数
            if 'base_params' in session:
                base_params = session['base_params']
                for city in citiesconfig:
                    city.set_base_params(
                        beta0=base_params['beta0'],
                        gamma0=base_params['gamma0'],
                        kappa0=base_params['kappa0'],
                        epsilon0=base_params['epsilon']
                    )
            
            if 'weather_params' in session:
                weather_params = session['weather_params']
                for city in citiesconfig:
                    city.set_weather_params(
                        temp_a0=weather_params['tmp_a'],
                        temp_score0=weather_params['tmp_score'],
                        temp_c0=weather_params['tmp_c'],
                        ah_score0=weather_params['ah_score'],
                        ah_a0=weather_params['ah_a'],
                        ah_c0=weather_params['ah_c']
                    )
            
            ili_df = read_and_validate_cityILI_excel(filepath)
            results = run_city_bayesian_optimization(citiesconfig, weather_data, ili_df)
            
            # 转换图形
            opt_plot = fig_to_base64(results['optimization_plot'])
            pred_plot = fig_to_base64(results['prediction_plot'])
            
            # 合并所有参数
            all_params = {}
            if 'base_params' in session:
                all_params.update(session['base_params'])
            if 'weather_params' in session:
                all_params.update(session['weather_params'])
            all_params.update(results['best_params'])
            
            return jsonify({
                'success': True,
                'optimization_plot': opt_plot,
                'prediction_plot': pred_plot,
                'parameters': results['best_params'],
                'all_parameters': all_params
            })
            
    except Exception as e:
        return jsonify({'error': f'分析失败: {str(e)}'}), 400

@app.route('/upload_rl_train', methods=['POST'])
def upload_rl_train():
    """处理RL训练文件上传"""
    required_files = ['weather_file', 'cities_file']
    uploaded_files = {}
    
    for file_key in required_files:
        if file_key not in request.files:
            return jsonify({'error': f'缺少文件: {file_key}'}), 400
        
        file = request.files[file_key]
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': f'文件 {file_key} 无效'}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"train_{filename}")
        file.save(filepath)
        uploaded_files[file_key] = filepath
    
    session['rl_train_files'] = uploaded_files
    return jsonify({'success': True, 'message': '文件上传成功'})

@app.route('/train_rl', methods=['POST'])
def train_rl():
    """执行RL训练"""
    if 'rl_train_files' not in session:
        return jsonify({'error': '请先上传文件'}), 400
    
    try:
        files = session['rl_train_files']
        
        # 加载数据
        citiesconfig = read_cities_from_excel(files['cities_file'])
        weather_data = read_weather_data(files['weather_file'], citiesconfig)
        
        # 应用参数（如果有）
        if 'base_params' in session and 'weather_params' in session:
            base_params = session['base_params']
            weather_params = session['weather_params']
            
            for city in citiesconfig:
                city.set_base_params(
                    beta0=base_params['beta0'],
                    gamma0=base_params['gamma0'],
                    kappa0=base_params['kappa0'],
                    epsilon0=base_params['epsilon']
                )
                city.set_weather_params(
                    temp_a0=weather_params['tmp_a'],
                    temp_score0=weather_params['tmp_score'],
                    temp_c0=weather_params['tmp_c'],
                    ah_score0=weather_params['ah_score'],
                    ah_a0=weather_params['ah_a'],
                    ah_c0=weather_params['ah_c']
                )
        
        # 训练模型
        actionfile = os.path.join('src', 'rl', 'actions.xlsx')
        result = train_models(
            cities=citiesconfig,
            weather_data=weather_data,
            actionfile=actionfile,
            epochs=50,  # 减少训练轮数以加快速度
            models_path='saved_models',
            batch_size=64,
            use_gpu=False  # 设置为False以避免GPU问题
        )
        
        if result['success']:
            session['model_trained'] = True
            return jsonify({
                'success': True,
                'message': result['message']
            })
        else:
            return jsonify({'error': result['error']}), 400
            
    except Exception as e:
        return jsonify({'error': f'训练失败: {str(e)}'}), 400

@app.route('/upload_decision', methods=['POST'])
def upload_decision():
    """处理智慧决策文件上传"""
    required_files = ['weather_file', 'cities_file']
    uploaded_files = {}
    
    for file_key in required_files:
        if file_key not in request.files:
            return jsonify({'error': f'缺少文件: {file_key}'}), 400
        
        file = request.files[file_key]
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': f'文件 {file_key} 无效'}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"test_{filename}")
        file.save(filepath)
        uploaded_files[file_key] = filepath
    
    session['decision_files'] = uploaded_files
    return jsonify({'success': True, 'message': '文件上传成功'})

@app.route('/get_decision', methods=['POST'])
def get_decision():
    """执行智慧决策"""
    if 'decision_files' not in session:
        return jsonify({'error': '请先上传文件'}), 400
    
    if 'model_trained' not in session or not session['model_trained']:
        return jsonify({'error': '请先完成模型训练'}), 400
    
    try:
        files = session['decision_files']
        
        # 加载数据
        citiesconfig = read_cities_from_excel(files['cities_file'])
        weather_data = read_weather_data(files['weather_file'], citiesconfig)
        
        # 应用参数
        if 'base_params' in session and 'weather_params' in session:
            base_params = session['base_params']
            weather_params = session['weather_params']
            
            for city in citiesconfig:
                city.set_base_params(
                    beta0=base_params['beta0'],
                    gamma0=base_params['gamma0'],
                    kappa0=base_params['kappa0'],
                    epsilon0=base_params['epsilon']
                )
                city.set_weather_params(
                    temp_a0=weather_params['tmp_a'],
                    temp_score0=weather_params['tmp_score'],
                    temp_c0=weather_params['tmp_c'],
                    ah_score0=weather_params['ah_score'],
                    ah_a0=weather_params['ah_a'],
                    ah_c0=weather_params['ah_c']
                )
        
        # 运行仿真
        actionfile = os.path.join('src', 'rl', 'actions.xlsx')
        result = run_simulations(
            cities=citiesconfig,
            weather_data=weather_data,
            model_path='saved_models',
            num_simulations=100,  # 减少仿真次数
            output_file='simulation_results.xlsx',
            actionfile=actionfile
        )
        
        if result['success']:
            # 读取结果数据
            df_all = result['result_df']
            
            # 生成三个策略图表
            fig, axes = plt.subplots(3, 1, figsize=(10, 12))
            
            # 按城市和周聚合数据
            weekly_avg = df_all.groupby(['city', 'week']).agg({
                'mask': 'mean',
                'home': 'mean',
                'med': 'mean'
            }).reset_index()
            
            # Mask策略图
            for city in weekly_avg['city'].unique():
                city_data = weekly_avg[weekly_avg['city'] == city]
                axes[0].plot(city_data['week'], city_data['mask'], label=city, linewidth=2)
            axes[0].set_title('口罩佩戴策略', fontsize=16)
            axes[0].set_xlabel('周数', fontsize=12)
            axes[0].set_ylabel('策略强度', fontsize=12)
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Home策略图
            for city in weekly_avg['city'].unique():
                city_data = weekly_avg[weekly_avg['city'] == city]
                axes[1].plot(city_data['week'], city_data['home'], label=city, linewidth=2)
            axes[1].set_title('居家隔离策略', fontsize=16)
            axes[1].set_xlabel('周数', fontsize=12)
            axes[1].set_ylabel('策略强度', fontsize=12)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            # Med策略图
            for city in weekly_avg['city'].unique():
                city_data = weekly_avg[weekly_avg['city'] == city]
                axes[2].plot(city_data['week'], city_data['med'], label=city, linewidth=2)
            axes[2].set_title('医疗资源投入策略', fontsize=16)
            axes[2].set_xlabel('周数', fontsize=12)
            axes[2].set_ylabel('策略强度', fontsize=12)
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图表
            plot_path = os.path.join('static', 'results', f'decision_plots_{uuid.uuid4().hex}.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # 保存Excel文件
            excel_path = os.path.join('static', 'results', f'all_1000_runs_results_{uuid.uuid4().hex}.xlsx')
            df_all.to_excel(excel_path, index=False)
            
            return jsonify({
                'success': True,
                'plot_url': '/' + plot_path,
                'excel_url': '/' + excel_path
            })
        else:
            return jsonify({'error': result['error']}), 400
            
    except Exception as e:
        return jsonify({'error': f'决策生成失败: {str(e)}'}), 400

@app.route('/progress/<task>')
def progress(task):
    """获取任务进度（模拟）"""
    # 这里可以实现真实的进度追踪
    # 现在返回模拟进度
    progress = request.args.get('current', 0, type=int)
    if progress < 100:
        progress += 10
    
    return jsonify({'progress': progress})

if __name__ == '__main__':
    # 设置matplotlib字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    app.run(debug=True, port=5000)