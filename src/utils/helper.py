import numpy as np


def clean_row(row):
    """
    Cleans a time series row by removing leading and trailing zeros, replacing in-between zeros with NaN, 
    and filling NaNs using forward and backward fill.
    """
    row = row[(row != 0).cumsum() > 0][::-1][(row != 0).cumsum() > 0][::-1]
    row = row.replace(0, np.nan).ffill().bfill()

    return row.astype(int).values

# def clean_row_float(row):
#     """
#     清理时间序列行：正确移除首尾的0，将中间的0替换为NaN，然后使用前向和后向填充
#     """
#     # 找到第一个非零值的索引
#     first_non_zero = row.ne(0).idxmax() if not (row == 0).all() else None
    
#     # 找到最后一个非零值的索引
#     last_non_zero = row.ne(0)[::-1].idxmax() if not (row == 0).all() else None
    
#     # 如果全是0，返回空数组
#     if first_non_zero is None or last_non_zero is None:
#         return np.array([])
    
#     # 截取首尾非零部分
#     row_trimmed = row.loc[first_non_zero:last_non_zero]
    
#     # 将中间的0替换为NaN，然后进行填充
#     row_filled = row_trimmed.replace(0, np.nan).ffill().bfill()
    
#     # 保留一位小数
#     return np.round(row_filled.values, 1)


# 首先定义clean_row_float函数
def clean_row_float(row, max_length=None):
    """
    清理时间序列行：
    1. 正确移除首尾的0
    2. 将中间的0替换为NaN，然后使用前向填充
    3. 补全长度使所有序列一致，开头用0补全
    
    参数:
    row - 输入的时间序列行
    max_length - 最大长度，如果为None则使用当前行的有效长度
    
    返回:
    处理后的numpy数组
    """
    # 找到第一个非零值的索引
    first_non_zero = row.ne(0).idxmax() if not (row == 0).all() else None
    
    # 找到最后一个非零值的索引
    last_non_zero = row.ne(0)[::-1].idxmax() if not (row == 0).all() else None
    
    # 如果全是0，返回指定长度的0数组
    if first_non_zero is None or last_non_zero is None:
        return np.zeros(max_length) if max_length is not None else np.array([])
    
    # 截取首尾非零部分
    row_trimmed = row.loc[first_non_zero:last_non_zero]
    
    # 将中间的0替换为NaN，然后进行前向填充
    row_filled = row_trimmed.replace(0, np.nan).ffill()
    
    # 将剩余的NaN（主要是开头的）用0填充
    row_filled = row_filled.fillna(0)
    
    # 保留一位小数
    result = np.round(row_filled.values, 1)
    
    # 如果指定了max_length，则进行长度调整
    if max_length is not None:
        if len(result) < max_length:
            # 在开头补0
            padding = np.zeros(max_length - len(result))
            result = np.concatenate([padding, result])
        elif len(result) > max_length:
            # 如果超出长度，则截取后面的部分
            result = result[-max_length:]
    
    return result

# 计算每个数据集的最大有效长度
def calculate_max_lengths(data_dict):
    max_lengths = {}
    for key, df in data_dict.items():
        max_length = 0
        for _, row in df.iterrows():
            data_row = row.iloc[1:]  # 跳过标签列
            first_non_zero = data_row.ne(0).idxmax() if not (data_row == 0).all() else 0
            last_non_zero = data_row.ne(0)[::-1].idxmax() if not (data_row == 0).all() else 0
            
            if not (data_row == 0).all():
                length = last_non_zero - first_non_zero + 1
                max_length = max(max_length, length)
        
        max_lengths[key] = max_length
    
    return max_lengths