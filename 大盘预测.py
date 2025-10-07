# -*- coding: utf-8 -*-
"""
上证大盘涨跌预测系统 (基于Prophet-XGBoost混合模型)
环境要求: Python 3.8+, baostock, pandas, numpy, xgboost, sklearn, statsmodels, prophet
"""

import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免Tkinter问题
import matplotlib.pyplot as plt

import baostock as bs
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
import warnings
import logging
import traceback
import os
import sys
import re
from collections import Counter

# 配置日志
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sh_index_prediction.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# 尝试导入sklearn库
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectKBest, f_classif
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn库不可用")

# 尝试导入XGBoost库
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost库不可用")

# 尝试导入statsmodels用于特征工程
try:
    from statsmodels.tsa.stattools import adfuller, acf, pacf
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logger.warning("Statsmodels库不可用")

# 尝试导入Prophet用于时间序列预测
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logger.warning("Prophet库不可用")

def plot_feature_importance(feature_importance, feature_names, model_name):
    """绘制特征重要性图"""
    try:
        # 创建特征重要性DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        })
        
        # 按重要性排序
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        # 绘制水平条形图
        plt.figure(figsize=(10, 8))
        plt.barh(importance_df['feature'][:15], importance_df['importance'][:15])
        plt.xlabel('Feature Importance')
        plt.title(f'{model_name} - Top 15 Feature Importance')
        plt.gca().invert_yaxis()  # 最重要的特征在顶部
        
        # 保存图表
        os.makedirs("figures", exist_ok=True)
        plt.savefig(f"figures/{model_name}_feature_importance.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"特征重要性图表已保存到 figures/{model_name}_feature_importance.png")
        
        return importance_df
    except Exception as e:
        logger.error(f"绘制特征重要性图时出错: {e}")
        # 返回一个基本的DataFrame
        return pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        })

class SHIndexPredictor:
    def __init__(self):
        self.data = None
        self.returns = None
        self.rf_model = None
        self.scaler = None
        self.prophet_model = None
        self.prophet_xgb_model = None
        self.prophet_xgb_scaler = None
        self.feature_selector = None
        
    def fetch_data(self, start_date, end_date=None, frequency="d"):
        """从baostock获取上证指数数据"""
        if end_date is None:
            end_date = datetime.today().strftime("%Y-%m-%d")
            
        # 登录系统
        lg = bs.login()
        if lg.error_code != '0':
            raise ConnectionError(f"登录失败: {lg.error_msg}")

        try:
            # 上证指数代码
            stock_code = "sh.000001"
            
            # 获取K线数据
            rs = bs.query_history_k_data_plus(
                stock_code,
                "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg",
                start_date=start_date,
                end_date=end_date,
                frequency=frequency,
                adjustflag="3"  # 后复权
            )
            
            if rs.error_code != '0':
                raise ConnectionError(f"获取数据失败: {rs.error_msg}")
            
            # 转换为DataFrame
            data_list = []
            while (rs.error_code == '0') and rs.next():
                data_list.append(rs.get_row_data())
            
            if not data_list:
                raise ValueError("未获取到任何数据")
            
            df = pd.DataFrame(data_list, columns=rs.fields)
            
            # 数据类型转换
            numeric_cols = ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'turn', 'pctChg']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 处理日期
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            
            # 处理缺失值
            df.dropna(inplace=True)
            
            # 过滤低流动性交易日
            if 'volume' in df.columns:
                df = df[df['volume'] > 0]
            
            if len(df) < 30:
                raise ValueError("数据量不足，至少需要30个交易日的数据")
            
            # 添加技术指标
            self._add_technical_indicators(df)
            
            self.data = df
            logger.info(f"成功获取上证指数从{start_date}到{end_date}的数据，共{len(df)}条记录")
            return df
            
        except Exception as e:
            logger.error(f"获取数据时出错: {e}")
            raise
        finally:
            # 确保注销
            try:
                bs.logout()
            except:
                pass
    
    def _add_technical_indicators(self, df):
        """添加技术指标"""
        # 移动平均线
        df['MA5'] = df['close'].rolling(window=5).mean()
        df['MA10'] = df['close'].rolling(window=10).mean()
        df['MA20'] = df['close'].rolling(window=20).mean()
        df['MA60'] = df['close'].rolling(window=60).mean()
        df['MA120'] = df['close'].rolling(window=120).mean()
        
        # 移动平均线差异
        df['MA5_MA20_diff'] = df['MA5'] - df['MA20']
        df['MA10_MA60_diff'] = df['MA10'] - df['MA60']
        
        # 指数移动平均线
        df['EMA12'] = df['close'].ewm(span=12).mean()
        df['EMA26'] = df['close'].ewm(span=26).mean()
        
        # MACD
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 布林带
        df['BB_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']  # 布林带宽度
        
        # 日内波动特征
        df['intraday_high_low'] = (df['high'] - df['low']) / df['close']  # 日内价差
        df['overnight_gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)  # 隔夜跳空
        
        # 成交量特征
        df['volume_ma5'] = df['volume'].rolling(window=5).mean()
        df['volume_ma20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma20']  # 成交量比率
        
        # 添加收益率指标
        df['return_1d'] = df['close'].pct_change(1)  # 1日收益率
        df['return_5d'] = df['close'].pct_change(5)  # 5日收益率
        
        # 创建涨跌标签 (1: 上涨, 0: 下跌)
        df['target_1d'] = (df['return_1d'] > 0).astype(int)  # 次日涨跌
        df['target_5d'] = (df['return_5d'] > 0).astype(int)  # 未来5日涨跌
        
        # 填充NaN值
        df.fillna(method='bfill', inplace=True)
    
    def preprocess_data(self):
        """数据预处理"""
        if self.data is None:
            raise ValueError("请先获取数据")
            
        # 计算对数收益率
        self.data['log_return'] = np.log(self.data['close'] / self.data['close'].shift(1))
        self.returns = self.data['log_return'].dropna() * 100  # 转换为百分比
        
        # 缩尾处理异常值 (1%分位数)
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
        for col in ohlcv_cols:
            if col in self.data.columns:
                winsorized = stats.mstats.winsorize(self.data[col], limits=[0.01, 0.01])
                self.data[col] = winsorized.data if hasattr(winsorized, 'data') else winsorized
            
        return self.data, self.returns
    
    def setup_prophet_xgboost_model(self, target='target_5d'):
        """初始化Prophet-XGBoost混合模型"""
        if not PROPHET_AVAILABLE or not XGBOOST_AVAILABLE or not SKLEARN_AVAILABLE:
            logger.warning("Prophet、XGBoost或Scikit-learn不可用，Prophet-XGBoost模型将不可使用")
            return None, None
            
        try:
            # 准备Prophet数据
            prophet_df = self.data.reset_index()[['date', 'close']].copy()
            prophet_df.columns = ['ds', 'y']
            
            # 训练Prophet模型
            self.prophet_model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0
            )
            
            self.prophet_model.fit(prophet_df)
            
            # 创建未来5天的预测
            future = self.prophet_model.make_future_dataframe(periods=5)
            forecast = self.prophet_model.predict(future)
            
            # 提取预测结果
            prophet_predictions = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(len(self.data) + 5)
            prophet_predictions.set_index('ds', inplace=True)
            
            # 计算Prophet预测的收益率
            prophet_predictions['prophet_return_1d'] = prophet_predictions['yhat'].pct_change(1)
            prophet_predictions['prophet_return_5d'] = prophet_predictions['yhat'].pct_change(5)
            
            # 将Prophet预测结果合并到原始数据
            self.data['prophet_return_1d'] = prophet_predictions['prophet_return_1d'].reindex(self.data.index)
            self.data['prophet_return_5d'] = prophet_predictions['prophet_return_5d'].reindex(self.data.index)
            
            # 准备XGBoost特征
            feature_cols = [
                'MA5', 'MA10', 'MA20', 'MA60', 'MA120',
                'MA5_MA20_diff', 'MA10_MA60_diff',
                'EMA12', 'EMA26', 'MACD', 'MACD_signal', 'MACD_hist',
                'RSI', 'BB_middle', 'BB_upper', 'BB_lower', 'BB_width',
                'intraday_high_low', 'overnight_gap',
                'volume', 'volume_ma5', 'volume_ma20', 'volume_ratio',
                'return_1d', 'return_5d', 
                'prophet_return_1d', 'prophet_return_5d'
            ]
            
            # 确保所有特征都存在
            available_features = [col for col in feature_cols if col in self.data.columns]
            
            if len(available_features) < 15:
                logger.warning("可用特征不足，无法训练Prophet-XGBoost模型")
                return None, None
                
            # 准备数据
            X = self.data[available_features].copy()
            y = self.data[target].copy()  # 预测目标
            
            # 删除包含NaN的行
            valid_idx = X.notna().all(axis=1) & y.notna()
            X = X[valid_idx]
            y = y[valid_idx]
            
            if len(X) < 100:
                logger.warning(f"数据量不足({len(X)} < 100)，无法训练Prophet-XGBoost模型")
                return None, None
            
            # 检查样本不平衡情况
            class_counts = Counter(y)
            logger.info(f"类别分布: {class_counts}")
            
            # 检查是否有足够的样本进行训练
            if len(np.unique(y)) < 2:
                logger.warning("类别不足，无法训练分类模型")
                return None, None
            
            # 计算样本权重
            scale_pos_weight = class_counts[0] / class_counts[1] if class_counts[1] > 0 else 1
            
            # 标准化特征
            self.prophet_xgb_scaler = StandardScaler()
            X_scaled = self.prophet_xgb_scaler.fit_transform(X)
            
            # 特征选择
            self.feature_selector = SelectKBest(score_func=f_classif, k=min(20, len(available_features)))
            X_selected = self.feature_selector.fit_transform(X_scaled, y)
            selected_features = [available_features[i] for i in self.feature_selector.get_support(indices=True)]
            logger.info(f"选择的特征: {selected_features}")
            
            # 使用时间序列交叉验证
            tscv = TimeSeriesSplit(n_splits=5)
            
            # 参数网格
            param_grid = {
                'max_depth': [3, 6],
                'learning_rate': [0.01, 0.1],
                'n_estimators': [100, 200],
                'subsample': [0.8, 1.0]
            }
            
            # 基础模型
            base_model = XGBClassifier(
                objective='binary:logistic',
                eval_metric='auc',
                scale_pos_weight=scale_pos_weight,
                reg_alpha=1,
                random_state=42,
                n_jobs=-1
            )
            
            # 网格搜索
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                scoring='roc_auc',
                cv=tscv,
                verbose=0,
                n_jobs=-1
            )
            
            # 训练模型
            grid_search.fit(X_selected, y)
            
            # 获取最佳模型
            self.prophet_xgb_model = grid_search.best_estimator_
            
            # 评估模型
            best_score = grid_search.best_score_
            best_params = grid_search.best_params_
            
            logger.info(f"Prophet-XGBoost模型训练完成，最佳参数: {best_params}")
            logger.info(f"交叉验证最佳AUC: {best_score:.4f}")
            
            # 特征重要性
            feature_importance = self.prophet_xgb_model.feature_importances_
            importance_df = plot_feature_importance(feature_importance, selected_features, "Prophet-XGBoost")
            
            return self.prophet_xgb_model, importance_df
            
        except Exception as e:
            logger.error(f"训练Prophet-XGBoost模型失败: {e}")
            return None, None
    
    def setup_random_forest_model(self, target='target_5d'):
        """初始化随机森林分类模型"""
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn不可用，随机森林模型将不可用")
            return None
            
        try:
            # 准备特征和目标变量
            feature_cols = [
                'MA5', 'MA10', 'MA20', 'MA60', 'MA120',
                'MA5_MA20_diff', 'MA10_MA60_diff',
                'EMA12', 'EMA26', 'MACD', 'MACD_signal', 'MACD_hist',
                'RSI', 'BB_middle', 'BB_upper', 'BB_lower', 'BB_width',
                'intraday_high_low', 'overnight_gap',
                'volume', 'volume_ma5', 'volume_ma20', 'volume_ratio',
                'return_1d', 'return_5d'
            ]
            
            # 确保所有特征都存在
            available_features = [col for col in feature_cols if col in self.data.columns]
            
            if len(available_features) < 10:
                logger.warning("可用特征不足，无法训练随机森林模型")
                return None
                
            # 准备数据
            X = self.data[available_features].copy()
            y = self.data[target].copy()  # 预测目标
            
            # 删除包含NaN的行
            valid_idx = X.notna().all(axis=1) & y.notna()
            X = X[valid_idx]
            y = y[valid_idx]
            
            if len(X) < 100:
                logger.warning(f"数据量不足({len(X)} < 100)，无法训练随机森林模型")
                return None
                
            # 标准化特征
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # 划分训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # 训练随机森林模型
            self.rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            self.rf_model.fit(X_train, y_train)
            
            # 评估模型
            y_pred = self.rf_model.predict(X_test)
            y_pred_proba = self.rf_model.predict_proba(X_test)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            logger.info(f"随机森林模型训练完成，测试集准确率: {accuracy:.4f}, AUC: {auc:.4f}")
            logger.info(f"分类报告:\n{classification_report(y_test, y_pred)}")
            
            return self.rf_model
        except Exception as e:
            logger.error(f"训练随机森林模型失败: {e}")
            return None
    
    def prophet_xgboost_prediction(self, days=5):
        """使用Prophet-XGBoost预测未来涨跌方向"""
        if not PROPHET_AVAILABLE or not XGBOOST_AVAILABLE or self.prophet_xgb_model is None:
            logger.warning("Prophet-XGBoost模型不可用，无法进行涨跌方向预测")
            return None, None, None, None
            
        try:
            # 准备Prophet预测
            future = self.prophet_model.make_future_dataframe(periods=days)
            forecast = self.prophet_model.predict(future)
            
            # 获取最新的Prophet预测收益率
            latest_prophet_return_1d = forecast['yhat'].pct_change(1).iloc[-1]
            latest_prophet_return_5d = forecast['yhat'].pct_change(days).iloc[-1]
            
            # 准备特征
            feature_cols = [
                'MA5', 'MA10', 'MA20', 'MA60', 'MA120',
                'MA5_MA20_diff', 'MA10_MA60_diff',
                'EMA12', 'EMA26', 'MACD', 'MACD_signal', 'MACD_hist',
                'RSI', 'BB_middle', 'BB_upper', 'BB_lower', 'BB_width',
                'intraday_high_low', 'overnight_gap',
                'volume', 'volume_ma5', 'volume_ma20', 'volume_ratio',
                'return_1d', 'return_5d', 
                'prophet_return_1d', 'prophet_return_5d'
            ]
            
            # 确保所有特征都存在
            available_features = [col for col in feature_cols if col in self.data.columns]
            
            # 获取最新数据点
            latest_data = self.data[available_features].iloc[-1:].copy()
            
            # 更新Prophet预测值
            latest_data['prophet_return_1d'] = latest_prophet_return_1d
            latest_data['prophet_return_5d'] = latest_prophet_return_5d
            
            # 标准化特征
            latest_data_scaled = self.prophet_xgb_scaler.transform(latest_data)
            
            # 特征选择
            latest_data_selected = self.feature_selector.transform(latest_data_scaled)
            
            # 预测未来涨跌概率
            predicted_prob = self.prophet_xgb_model.predict_proba(latest_data_selected)[0]
            predicted_class = self.prophet_xgb_model.predict(latest_data_selected)[0]
            
            # 获取特征重要性
            feature_importance = self.prophet_xgb_model.feature_importances_
            selected_features = [available_features[i] for i in self.feature_selector.get_support(indices=True)]
            
            logger.info(f"Prophet-XGBoost预测未来{days}天涨跌方向: {'上涨' if predicted_class == 1 else '下跌'}, "
                       f"上涨概率: {predicted_prob[1]:.4f}")
            
            return predicted_class, predicted_prob[1], feature_importance, selected_features
        except Exception as e:
            logger.error(f"Prophet-XGBoost预测失败: {e}")
            return None, None, None, None
    
    def random_forest_prediction(self):
        """使用随机森林预测未来涨跌方向"""
        if not SKLEARN_AVAILABLE or self.rf_model is None:
            logger.warning("随机森林模型不可用，无法进行涨跌方向预测")
            return None, None
            
        try:
            # 准备特征
            feature_cols = [
                'MA5', 'MA10', 'MA20', 'MA60', 'MA120',
                'MA5_MA20_diff', 'MA10_MA60_diff',
                'EMA12', 'EMA26', 'MACD', 'MACD_signal', 'MACD_hist',
                'RSI', 'BB_middle', 'BB_upper', 'BB_lower', 'BB_width',
                'intraday_high_low', 'overnight_gap',
                'volume', 'volume_ma5', 'volume_ma20', 'volume_ratio',
                'return_1d', 'return_5d'
            ]
            
            # 确保所有特征都存在
            available_features = [col for col in feature_cols if col in self.data.columns]
            
            # 获取最新数据点
            latest_data = self.data[available_features].iloc[-1:].copy()
            
            # 标准化特征
            latest_data_scaled = self.scaler.transform(latest_data)
            
            # 预测未来涨跌概率
            predicted_prob = self.rf_model.predict_proba(latest_data_scaled)[0]
            predicted_class = self.rf_model.predict(latest_data_scaled)[0]
            
            logger.info(f"随机森林预测未来5天涨跌方向: {'上涨' if predicted_class == 1 else '下跌'}, "
                       f"上涨概率: {predicted_prob[1]:.4f}")
            
            return predicted_class, predicted_prob[1]
        except Exception as e:
            logger.error(f"随机森林预测失败: {e}")
            return None, None

def save_results_to_file(results_text, file_path):
    """将结果保存到文件"""
    try:
        # 确保目录存在
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            logger.info(f"创建目录: {directory}")
        
        with open(file_path, 'a', encoding='utf-8') as file:
            file.write(results_text + "\n")
        
        logger.info(f"结果已保存到文件: {file_path}")
        return True
    except Exception as e:
        logger.error(f"保存结果到文件时出错: {e}")
        # 尝试保存到当前目录
        try:
            fallback_path = os.path.join(os.getcwd(), "sh_index_prediction_results.txt")
            with open(fallback_path, 'a', encoding='utf-8') as file:
                file.write(results_text + "\n")
            logger.info(f"结果已保存到备用文件: {fallback_path}")
            return True
        except Exception as e2:
            logger.error(f"保存到备用文件也失败: {e2}")
            return False

# 主函数
def main():
    # 初始化预测器
    predictor = SHIndexPredictor()
    
    # 输出文件路径
    output_file_path = r"C:\Users\Administrator\AppData\Local\Programs\Python\Python312\Kronos\sh_index_prediction_results.txt"
    
    # 清空输出文件
    try:
        directory = os.path.dirname(output_file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            logger.info(f"创建目录: {directory}")
        
        with open(output_file_path, 'w', encoding='utf-8') as file:
            file.write("=== 上证指数涨跌预测结果 ===\n")
            file.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        logger.info(f"已清空输出文件: {output_file_path}")
    except Exception as e:
        logger.error(f"清空输出文件时出错: {e}")
        # 尝试在当前目录创建文件
        try:
            output_file_path = os.path.join(os.getcwd(), "sh_index_prediction_results.txt")
            with open(output_file_path, 'w', encoding='utf-8') as file:
                file.write("=== 上证指数涨跌预测结果 ===\n")
                file.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            logger.info(f"已创建备用输出文件: {output_file_path}")
        except Exception as e2:
            logger.error(f"创建备用输出文件也失败: {e2}")
            return
    
    start_date = "2000-01-01"  # 从2000年开始获取数据
    end_date = datetime.today().strftime("%Y-%m-%d")  # 到今天为止
    
    try:
        logger.info("开始处理上证指数数据")
        
        # 获取数据
        logger.info("开始获取数据...")
        data = predictor.fetch_data(start_date, end_date=end_date, frequency="d")
        
        logger.info("数据预处理...")
        data, returns = predictor.preprocess_data()
        
        # 检查数据是否足够
        if len(returns) < 100:
            error_msg = "数据量不足，无法训练模型"
            logger.error(error_msg)
            save_results_to_file(error_msg, output_file_path)
            return
        
        # 设置Prophet-XGBoost模型 (预测未来5日涨跌)
        logger.info("初始化Prophet-XGBoost模型...")
        prophet_xgb_model, importance_df = predictor.setup_prophet_xgboost_model(target='target_5d')
        
        # 设置随机森林模型 (预测未来5日涨跌)
        logger.info("初始化随机森林模型...")
        rf_model = predictor.setup_random_forest_model(target='target_5d')
        
        # 进行Prophet-XGBoost涨跌方向预测
        logger.info("进行Prophet-XGBoost涨跌方向预测...")
        prophet_xgb_direction, prophet_xgb_prob, prophet_xgb_importance, prophet_xgb_features = predictor.prophet_xgboost_prediction(days=5)
        
        # 进行随机森林涨跌方向预测
        logger.info("进行随机森林涨跌方向预测...")
        rf_direction, rf_prob = predictor.random_forest_prediction()
        
        # 准备输出文本
        prophet_xgb_direction_str = '上涨' if prophet_xgb_direction == 1 else '下跌' if prophet_xgb_direction == 0 else 'N/A'
        prophet_xgb_prob_str = f"{prophet_xgb_prob:.4f}" if prophet_xgb_prob is not None else 'N/A'
        
        rf_direction_str = '上涨' if rf_direction == 1 else '下跌' if rf_direction == 0 else 'N/A'
        rf_prob_str = f"{rf_prob:.4f}" if rf_prob is not None else 'N/A'
        
        # 获取当前指数信息
        current_close = data['close'].iloc[-1] if len(data) > 0 else 'N/A'
        current_date = data.index[-1].strftime('%Y-%m-%d') if len(data) > 0 else 'N/A'
        
        results_text = f"""
=== 上证指数预测结果 ===
数据截止日期: {current_date}
当前收盘价: {current_close}

Prophet-XGBoost预测未来5日涨跌方向: {prophet_xgb_direction_str}
Prophet-XGBoost预测上涨概率: {prophet_xgb_prob_str}

随机森林预测未来5日涨跌方向: {rf_direction_str}
随机森林预测上涨概率: {rf_prob_str}

模型一致预测: {'上涨' if prophet_xgb_direction == rf_direction else '分歧'}
"""
        
        # 输出到控制台
        print(results_text)
        
        # 保存到文件
        if not save_results_to_file(results_text, output_file_path):
            logger.error("保存结果失败")
            
    except Exception as e:
        error_msg = f"程序执行出错: {e}"
        logger.error(error_msg)
        save_results_to_file(error_msg, output_file_path)
        traceback.print_exc()
    finally:
        logger.info("程序执行完毕")

if __name__ == "__main__":
    main()