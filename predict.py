import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. 数据加载与预处理
df = pd.read_csv('stock_data.csv')  # 替换为你的数据文件路径
df['trade_time'] = pd.to_datetime(df['trade_time'])  # 确保日期列是日期格式
df.set_index('trade_time', inplace=True)

# 选择需要的特征列，这里假设你想预测"开盘价"
data = df[['open']].values  # 获取开盘价列

# 数据标准化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# 创建训练数据和测试数据，假设使用60天的历史数据来预测下一天的开盘价
look_back = 500  # 选择历史天数
X, y = [], []
for i in range(look_back, len(scaled_data)):
    X.append(scaled_data[i-look_back:i, 0])  # 取前60天的数据
    y.append(scaled_data[i, 0])  # 预测第61天的开盘价

X = np.array(X)
y = np.array(y)

# 将数据重塑为LSTM输入格式
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# 2. 切分训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 3. 构建LSTM模型
model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(units=1))  # 预测下一天的开盘价

# 编译模型
model.compile(optimizer='adam', loss='mean_squared_error')

# 4. 训练模型
model.fit(X_train, y_train, epochs=10, batch_size=32)

# 5. 预测与反标准化
predicted_stock_price = model.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

# 实际值反标准化
y_test = np.reshape(y_test, (-1, 1))
y_test = scaler.inverse_transform(y_test)

# 6. 计算评估指标
# 计算均方误差 (MSE)
mse = mean_squared_error(y_test, predicted_stock_price)
# 计算均方根误差 (RMSE)
rmse = np.sqrt(mse)
# 计算平均绝对误差 (MAE)
mae = mean_absolute_error(y_test, predicted_stock_price)
# 计算R² (决定系数)
r2 = r2_score(y_test, predicted_stock_price)

# 打印指标
print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'R² (Coefficient of Determination): {r2}')

# 7. 绘制评估指标图表
metrics = ['MSE', 'RMSE', 'MAE', 'R²']
values = [mse, rmse, mae, r2]

plt.figure(figsize=(8, 5))
plt.bar(metrics, values, color=['blue', 'green', 'red', 'orange'])

# 给柱状图加上数值标签
for i, v in enumerate(values):
    plt.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontsize=12)

plt.title('Evaluation Metrics for Stock Price Prediction')
plt.xlabel('Metrics')
plt.ylabel('Value')
plt.ylim(0, max(values) + 0.2)  # 确保y轴有足够的空间显示数值
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# 7. 可视化预测结果
# 保留测试集对应的日期
test_dates = df.index[-len(y_test):]  # 获取测试集的日期

plt.figure(figsize=(12, 6))
plt.plot(test_dates, y_test, color='red', label='Actual Stock Price')  # 用日期作为X轴
plt.plot(test_dates, predicted_stock_price, color='blue', label='Predicted Stock Price')
plt.title('Stock Price Prediction using LSTM')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.xticks(rotation=45)  # 旋转X轴日期标签，避免重叠
plt.grid()
plt.show()

