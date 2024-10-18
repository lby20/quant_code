"""
@author: caiyucheng
@contact: caiyucheng23@163.com
@date: 2024-07-02
"""
import numpy as np
import bisect
import pandas as pd
from tqdm import tqdm
from collections import deque
from scipy.stats import skew, kurtosis
from functools import reduce
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')


SIZE = None

def div_cs(a, b): return 100 * np.where(a + b == 0, np.nan, (a - b) / (a + b))
def div_cs_abs(a, b): return 100 * np.where(a + b == 0, np.nan, (a - b) / (a + b))
def div(a, b): return np.where(b == 0, np.nan, a / b)
def sub(a, b): return a - b
def Abs(a): return np.fabs(a)
def if_big(a,b,c,d): return np.where(a > b, c, d)
def if_small(a,b,c,d): return np.where(a < b, c, d)
def if_equal(a,b,c,d): return np.where(a == b, c, d)
def growth(a,b): return 100 * np.where(b == 0, np.nan, a / b - 1)
def log(a): return np.log(a)
def pow(a, b): return a ** b


def Max(*args): 
    res = reduce(lambda a,b: np.where(a >= b, a, b), args)
    res = np.array(res)
    if isinstance(res, (int, float, bool)):
        res = np.full(shape=SIZE, fill_value=res)
    return res


def Min(*args): 
    res = reduce(lambda a,b: np.where(a <= b, a, b), args)
    if isinstance(res, (int, float, bool)):
        res = np.full(shape=SIZE, fill_value=res)
    return res


def avg(*args): 
    res = reduce(lambda a,b: a + b, args) / len(args)
    if isinstance(res, (int, float, bool)):
        res = np.full(shape=SIZE, fill_value=res)
    return res


def add(*args): 
    res = reduce(lambda a,b: a + b, args)
    if isinstance(res, (int, float, bool)):
        res = np.full(shape=SIZE, fill_value=res)
    return res


def med(*args): 
    res = np.nanmedian(args, axis=0) if (len(args) > 1) else np.full_like(args[0],fill_value=np.nanmedian(args[0]))
    return res


def mul(*args): 
    res = reduce(lambda a,b: a * b, args)
    if isinstance(res, (int, float, bool)):
        res = np.full(shape=SIZE, fill_value=res)
    return res


# 残差
def de_beta(*args):
    x = sm.add_constant(np.array(args[:-1]).T)
    y = args[-1]
    mask = np.isnan(x).sum(axis=1).astype(bool) | np.isnan(args[-1]) | \
        np.isinf(x).sum(axis=1).astype(bool) | np.isinf(args[-1])
    x = x[~mask]
    y = args[-1][~mask]
    res = sm.OLS(y,x).fit().resid
    res_data = np.zeros_like(args[-1])
    res_data[~mask] = res
    return res_data

# 拟合优度
def r2(*args):
    x = sm.add_constant(np.array(args[:-1]).T)
    y = args[-1]
    mask = np.isnan(x).sum(axis=1).astype(bool) | np.isnan(args[-1]) | \
        np.isinf(x).sum(axis=1).astype(bool) | np.isinf(args[-1])
    x = x[~mask]
    y = args[-1][~mask]
    res_data = np.full_like(args[-1], fill_value=sm.OLS(y,x).fit().rsquared)
    return res_data

# 多元回归中,最后一个元素的斜率系数
def beta(*args):
    x = sm.add_constant(np.array(args[:-1]).T)
    y = args[-1]
    mask = np.isnan(x).sum(axis=1).astype(bool) | np.isnan(args[-1]) | \
        np.isinf(x).sum(axis=1).astype(bool) | np.isinf(args[-1])
    x = x[~mask]
    y = args[-1][~mask]
    res = sm.OLS(y,x).fit().params[-1]
    res_data = np.zeros_like(args[-1])
    res_data[~mask] = res
    res_data[mask] = np.nan
    return res_data

# 多元回归中,最后一个元素的t检验值
def t_value(*args):
    x = sm.add_constant(np.array(args[:-1]).T)
    y = args[-1]
    mask = np.isnan(x).sum(axis=1).astype(bool) | np.isnan(args[-1]) | \
        np.isinf(x).sum(axis=1).astype(bool) | np.isinf(args[-1])
    x = x[~mask]
    y = args[-1][~mask]
    res_data = np.full_like(args[-1], fill_value=sm.OLS(y,x).fit().tvalues[-1])
    return res_data

# 多元回归中,截距项
def intercept(*args):
    x = sm.add_constant(np.array(args[:-1]).T)
    y = args[-1]
    mask = np.isnan(x).sum(axis=1).astype(bool) | np.isnan(args[-1]) | \
        np.isinf(x).sum(axis=1).astype(bool) | np.isinf(args[-1])
    x = x[~mask]
    y = args[-1][~mask]
    res_data = np.full_like(args[-1], fill_value=sm.OLS(y,x).fit().params[0])
    return res_data

# x - delay(x, n)
class ts_delta:
    def __init__(self, window, size, dtype):
        self.window = window
        self.size = size
        self.data_queue = np.zeros(shape=(window, size), dtype=dtype)
        self.count = 0 
        self.new_idx = 0  # 当前更新到的某个窗口
        self._info = np.zeros(shape=size, dtype=dtype)

    def __call__(self, data):
        mask = np.isnan(data) | np.isinf(data)
        # 更时间窗口超过window
        if self.count >= self.window:
            self._info = data - self.data_queue[self.new_idx]
        # 更新时间窗口不足window
        else:
            self._info = data - self.data_queue[0]
            self.count += 1
        self._info = np.where(np.isnan(self._info), np.nan, self._info)
        self.data_queue[self.new_idx][mask] = np.nan
        self.data_queue[self.new_idx][~mask] = data[~mask]
        self.new_idx += 1 
        if self.new_idx > self.window - 1:
            self.new_idx = 0
        return self.info

    def update_base(self, data):
        mask = np.isnan(data) | np.isinf(data)
        self._info = data - self.data_queue[self.new_idx]
        self.data_queue[self.new_idx][mask] = np.nan
        self.data_queue[self.new_idx][~mask] = data[~mask]
        self.new_idx += 1
        if self.new_idx > self.window - 1:
            self.new_idx = 0
        return self
    
    def update_info(self):
        # delta没法从循环队列中更新 只能在update_base里相减
        return self.info
    
    @property
    def info(self):
        return self._info
    
    def adjust(self, new_size, old_index, new_index):
        self.size = new_size
        all_attr = list(self.__dict__.keys())
        for name in all_attr:
            if isinstance(getattr(self, name), np.ndarray):
                nd = len(getattr(self, name).shape)
                if nd == 2:
                    new_attr = np.zeros(shape=(self.window, new_size), dtype=getattr(self,name).dtype)
                    new_attr[:,new_index] = getattr(self,name)[:,old_index]
                    setattr(self, name, new_attr)
                elif nd == 1:
                    new_attr = np.zeros(shape=new_size, dtype=getattr(self,name).dtype)
                    new_attr[new_index] = getattr(self,name)[old_index]
                    setattr(self,name,new_attr)
                else:
                    raise Exception("A higher than 2-dimensional ndarray has appeared.")

# delay(x, n)
class ts_delay:
    def __init__(self, window, size, dtype):
        self.window = window
        self.size = size
        self.data_queue = np.zeros(shape=(window, size), dtype=dtype)
        self.count = 0 
        self.new_idx = 0  # 当前更新到的某个窗口
        self._info = np.zeros(shape=size, dtype=dtype)

    def __call__(self, data):
        mask = np.isnan(data) | np.isinf(data)
        # 更时间窗口超过window
        if self.count >= self.window:
            self._info = self.data_queue[self.new_idx]
        # 更新时间窗口不足window
        else:
            self._info = self.data_queue[0]
            self.count += 1
        self._info = np.where(np.isnan(self._info), np.nan, self._info)
        self.data_queue[self.new_idx][mask] = np.nan
        self.data_queue[self.new_idx][~mask] = data[~mask]
        self.new_idx += 1 
        if self.new_idx > self.window - 1:
            self.new_idx = 0
        return self.info

    def update_base(self, data):
        mask = np.isnan(data) | np.isinf(data)
        # 这里必须深拷贝
        self._info = self.data_queue[self.new_idx] + 0  
        self.data_queue[self.new_idx][mask] = np.nan
        self.data_queue[self.new_idx][~mask] = data[~mask]
        self.new_idx += 1
        if self.new_idx > self.window - 1:
            self.new_idx = 0
        return self
    
    def update_info(self):
        # delta没法从循环队列中更新 只能在update_base里相减
        return self.info
    
    @property
    def info(self):
        return self._info
    
    def adjust(self, new_size, old_index, new_index):
        self.size = new_size
        all_attr = list(self.__dict__.keys())
        for name in all_attr:
            if isinstance(getattr(self, name), np.ndarray):
                nd = len(getattr(self, name).shape)
                if nd == 2:
                    new_attr = np.zeros(shape=(self.window, new_size), dtype=getattr(self,name).dtype)
                    new_attr[:,new_index] = getattr(self,name)[:,old_index]
                    setattr(self, name, new_attr)
                elif nd == 1:
                    new_attr = np.zeros(shape=new_size, dtype=getattr(self,name).dtype)
                    new_attr[new_index] = getattr(self,name)[old_index]
                    setattr(self,name,new_attr)
                else:
                    raise Exception("A higher than 2-dimensional ndarray has appeared.")

# 100 * (x - delay(x,n)) / delay(x,n)
class ts_growth:
    def __init__(self, window, size, dtype):
        self.window = window
        self.size = size
        self.data_queue = np.zeros(shape=(window, size), dtype=dtype)
        self.count = 0 
        self.new_idx = 0  # 当前更新到的某个窗口
        self._info = np.zeros(shape=size, dtype=dtype)

    def __call__(self, data):
        mask = np.isnan(data) | np.isinf(data)
        # 更时间窗口超过window
        if self.count >= self.window:
            self._info =  100 * (data - self.data_queue[self.new_idx])/self.data_queue[self.new_idx]
        # 更新时间窗口不足window
        else:
            self._info = 100 * (data - self.data_queue[0]) / self.data_queue[0]
            self.count += 1
        self._info[np.isinf(self._info)] = np.nan
        self.data_queue[self.new_idx][mask] = np.nan
        self.data_queue[self.new_idx][~mask] = data[~mask]
        self.new_idx += 1 
        if self.new_idx > self.window - 1:
            self.new_idx = 0
        return self.info

    def update_base(self, data):
        mask = np.isnan(data) | np.isinf(data)
        self._info = 100 * (data - self.data_queue[self.new_idx]) / self.data_queue[self.new_idx]
        self._info[np.isinf(self._info)] = np.nan
        self.data_queue[self.new_idx][mask] = np.nan
        self.data_queue[self.new_idx][~mask] = data[~mask]
        self.new_idx += 1
        if self.new_idx > self.window - 1:
            self.new_idx = 0
        return self
    
    def update_info(self):
        # delta没法从循环队列中更新 只能在update_base里相减
        return self.info
    
    @property
    def info(self):
        return self._info
    
    def adjust(self, new_size, old_index, new_index):
        self.size = new_size
        all_attr = list(self.__dict__.keys())
        for name in all_attr:
            if isinstance(getattr(self, name), np.ndarray):
                nd = len(getattr(self, name).shape)
                if nd == 2:
                    new_attr = np.zeros(shape=(self.window, new_size), dtype=getattr(self,name).dtype)
                    new_attr[:,new_index] = getattr(self,name)[:,old_index]
                    setattr(self, name, new_attr)
                elif nd == 1:
                    new_attr = np.zeros(shape=new_size, dtype=getattr(self,name).dtype)
                    new_attr[new_index] = getattr(self,name)[old_index]
                    setattr(self,name,new_attr)
                else:
                    raise Exception("A higher than 2-dimensional ndarray has appeared.")

# 100 * abs((x - delay(x,n)) / delay(x,n))
class ts_growth_abs:
    def __init__(self, window, size, dtype):
        self.window = window
        self.size = size
        self.data_queue = np.zeros(shape=(window, size), dtype=dtype)
        self.count = 0 
        self.new_idx = 0  # 当前更新到的某个窗口
        self._info = np.zeros(shape=size, dtype=dtype)

    def __call__(self, data):
        mask = np.isnan(data) | np.isinf(data)
        # 更时间窗口超过window
        if self.count >= self.window:
            self._info =  100 * (data - self.data_queue[self.new_idx])/self.data_queue[self.new_idx]
        # 更新时间窗口不足window
        else:
            self._info = 100 * (data - self.data_queue[0]) / self.data_queue[0]
            self.count += 1
        self._info[np.isinf(self._info)] = np.nan
        self._info = np.fabs(self._info)
        self.data_queue[self.new_idx][mask] = np.nan
        self.data_queue[self.new_idx][~mask] = data[~mask]
        self.new_idx += 1 
        if self.new_idx > self.window - 1:
            self.new_idx = 0
        return self.info

    def update_base(self, data):
        mask = np.isnan(data) | np.isinf(data)
        self._info = 100 * (data - self.data_queue[self.new_idx]) / self.data_queue[self.new_idx]
        self._info[np.isinf(self._info)] = np.nan
        self.data_queue[self.new_idx][mask] = np.nan
        self.data_queue[self.new_idx][~mask] = data[~mask]
        self.new_idx += 1
        if self.new_idx > self.window - 1:
            self.new_idx = 0
        return self
    
    def update_info(self):
        # delta没法从循环队列中更新 只能在update_base里相减
        return self.info
    
    @property
    def info(self):
        return self._info
    
    def adjust(self, new_size, old_index, new_index):
        self.size = new_size
        all_attr = list(self.__dict__.keys())
        for name in all_attr:
            if isinstance(getattr(self, name), np.ndarray):
                nd = len(getattr(self, name).shape)
                if nd == 2:
                    new_attr = np.zeros(shape=(self.window, new_size), dtype=getattr(self,name).dtype)
                    new_attr[:,new_index] = getattr(self,name)[:,old_index]
                    setattr(self, name, new_attr)
                elif nd == 1:
                    new_attr = np.zeros(shape=new_size, dtype=getattr(self,name).dtype)
                    new_attr[new_index] = getattr(self,name)[old_index]
                    setattr(self,name,new_attr)
                else:
                    raise Exception("A higher than 2-dimensional ndarray has appeared.")

# 滑动均值 0.03秒/1000个因子/5000个股票/一个截面
class ts_avg:
    def __init__(self, window, size, dtype):
        """
        window: 回溯窗口长度
        size: 合约数量
        dtype: 中间状态类型,近似为返回结果类似
        """
        self.window = window
        self.size = size
        self.data_queue = np.zeros(shape=(window, size), dtype=dtype)
        self.mask_queue = np.zeros(shape=(window, size), dtype=np.int32)
        self.sum = np.zeros(shape=size, dtype=dtype)
        self.cnt = np.zeros(shape=size, dtype=dtype)
        self.count = 0 
        self.new_idx = 0  # 当前更新到的某个窗口
        self._info = None

    def __call__(self, data):
        # 更时间窗口超过window
        if self.count >= self.window:
            new_mask = np.isnan(data) | np.isinf(data)
            new_data = data.copy()
            new_data[new_mask] = 0
            self.sum += (new_data - self.data_queue[self.new_idx])
            self.cnt += (np.where(new_mask,0,1) - self.mask_queue[self.new_idx])
            self.data_queue[self.new_idx] = new_data
            self.mask_queue[self.new_idx] = np.where(new_mask, 0, 1)
            self._info = np.where(self.cnt > 0, self.sum / self.cnt, np.nan)
        # 更新时间窗口不足window
        else:
            mask = np.isnan(data) | np.isinf(data)
            self.data_queue[self.new_idx] = data
            self.data_queue[self.new_idx][mask] = 0
            mask = np.where(mask, 0, 1)
            self.mask_queue[self.new_idx] = mask
            self.cnt += mask
            self.sum += self.data_queue[self.new_idx]
            self._info = np.where(self.cnt > 0, self.sum / self.cnt, np.nan)
            self.count += 1
        self.new_idx += 1 
        if self.new_idx > self.window - 1:
            self.new_idx = 0
        return self.info

    def update_base(self, data):
        mask = np.isnan(data)
        self.data_queue[self.new_idx][mask] = - np.inf
        self.data_queue[self.new_idx][~mask] = data[~mask]
        self.new_idx += 1
        if self.new_idx > self.window - 1:
            self.new_idx = 0
    
    def update_info(self):
        self._info = np.mean(self.data_queue, axis=0)
        return self.info
    
    @property
    def info(self):
        return self._info
    
    def adjust(self, new_size, old_index, new_index):
        self.size = new_size
        all_attr = list(self.__dict__.keys())
        for name in all_attr:
            if isinstance(getattr(self, name), np.ndarray):
                nd = len(getattr(self, name).shape)
                if nd == 2:
                    new_attr = np.zeros(shape=(self.window, new_size), dtype=getattr(self,name).dtype)
                    new_attr[:,new_index] = getattr(self,name)[:,old_index]
                    setattr(self, name, new_attr)
                elif nd == 1:
                    new_attr = np.zeros(shape=new_size, dtype=getattr(self,name).dtype)
                    new_attr[new_index] = getattr(self,name)[old_index]
                    setattr(self,name,new_attr)
                else:
                    raise Exception("A higher than 2-dimensional ndarray has appeared.")

# 滑动求和
class ts_sum:
    def __init__(self, window, size, dtype):
        """
        window: 回溯窗口长度
        size: 合约数量
        dtype: 中间状态类型,近似为返回结果类似
        """
        self.window = window
        self.size = size
        self.data_queue = np.zeros(shape=(window, size), dtype=dtype)
        self.mask_queue = np.zeros(shape=(window, size), dtype=np.int32)
        self.sum = np.zeros(shape=size, dtype=dtype)
        self.count = 0 
        self.new_idx = 0  # 当前更新到的某个窗口
        self._info = np.zeros(shape=size, dtype=dtype)

    def __call__(self, data):
        mask = np.isnan(data) | np.isinf(data)
        # 更时间窗口超过window
        if self.count >= self.window:
            self._info[~mask] += data[~mask]
            self._info -= self.data_queue[self.new_idx]
        # 更新时间窗口不足window
        else:
            self._info[~mask] += data[~mask]
            self.count += 1
        self.data_queue[self.new_idx][mask] = 0
        self.data_queue[self.new_idx][~mask] = data[~mask]
        self.new_idx += 1 
        if self.new_idx > self.window - 1:
            self.new_idx = 0
        return self.info

    def update_base(self, data):
        mask = np.isnan(data)
        self.data_queue[self.new_idx][mask] = 0
        self.data_queue[self.new_idx][~mask] = data[~mask]
        self.new_idx += 1
        if self.new_idx > self.window - 1:
            self.new_idx = 0
        return self
    
    def update_info(self):
        self._info = np.sum(self.data_queue, axis=0)
        return self.info
    
    @property
    def info(self):
        return self._info
    
    def adjust(self, new_size, old_index, new_index):
        self.size = new_size
        all_attr = list(self.__dict__.keys())
        for name in all_attr:
            if isinstance(getattr(self, name), np.ndarray):
                nd = len(getattr(self, name).shape)
                if nd == 2:
                    new_attr = np.zeros(shape=(self.window, new_size), dtype=getattr(self,name).dtype)
                    new_attr[:,new_index] = getattr(self,name)[:,old_index]
                    setattr(self, name, new_attr)
                elif nd == 1:
                    new_attr = np.zeros(shape=new_size, dtype=getattr(self,name).dtype)
                    new_attr[new_index] = getattr(self,name)[old_index]
                    setattr(self,name,new_attr)
                else:
                    raise Exception("A higher than 2-dimensional ndarray has appeared.")


class ts_rsi:
    def __init__(self, window, size, dtype):
        self.window = window
        self.size = size
        self.pos_queue = np.zeros(shape=(window, size), dtype=dtype)
        self.pos_sum = np.zeros(shape=size, dtype=dtype)
        self.pos_cnt = np.zeros(shape=size, dtype=np.int32)
        self.pos_mask = np.zeros(shape=(window, size), dtype=np.int32)
        self.neg_queue = np.zeros(shape=(window, size), dtype=dtype)
        self.neg_sum = np.zeros(shape=size, dtype=dtype)
        self.neg_cnt = np.zeros(shape=size, dtype=np.int32)
        self.neg_mask = np.zeros(shape=(window, size), dtype=np.int32)
        self.count = 0
        self.new_idx = 0
        self._info = np.zeros(shape=size, dtype=dtype)

    def __call__(self, data):
        mask = np.isnan(data) | np.isinf(data)
        new_data = data.copy()
        new_data[mask] = np.nan
        pos_bool = np.where(new_data > 0, 1, 0)
        neg_bool = np.where(new_data < 0, -1, 0)
        if self.count >= self.window:
            self.pos_sum += (new_data * pos_bool - self.pos_queue[self.new_idx])
            self.pos_cnt += (pos_bool - self.pos_mask[self.new_idx])
            self.pos_mask[self.new_idx] = pos_bool
            self.neg_sum += (neg_bool * new_data - self.neg_queue[self.new_idx])
            self.neg_cnt += (neg_bool - self.neg_mask[self.new_idx])
            self.neg_mask[self.new_idx] = neg_bool
        else:
            self.pos_sum += new_data * pos_bool
            self.pos_cnt += pos_bool
            self.pos_mask[self.new_idx] = pos_bool
            self.neg_sum += new_data * neg_bool
            self.neg_cnt += neg_bool
            self.neg_mask[self.new_idx] = neg_bool
            self.count += 1
        self.pos_queue[self.new_idx] = new_data * pos_bool
        self.neg_queue[self.new_idx] = new_data * neg_bool
        self._info = 100 - 100 / (1 + ((self.pos_sum / self.pos_cnt) / (- self.neg_sum / self.neg_cnt)))
        self.new_idx += 1
        if self.new_idx > self.window - 1:
            self.new_idx = 0
        return self.info
    
    def update_base(self, data):
        mask = np.isnan(data) | np.isinf(data)
        new_data = data.copy()
        new_data[mask] = np.nan
        pos_bool = np.where(new_data > 0, 1, 0)
        neg_bool = np.where(new_data < 0, -1, 0)
        self.pos_queue[self.new_idx] = new_data * pos_bool
        self.pos_mask[self.new_idx] = pos_bool
        self.neg_queue[self.new_idx] = new_data * neg_bool
        self.neg_mask[self.new_idx] = neg_bool
        self.new_idx += 1
        if self.new_idx > self.window - 1:
            self.new_idx = 0
        return self

    def update_info(self):
        self._info = 100 - 100 / (1 + ((self.pos_queue.sum(axis=0) / self.pos_mask.sum(axis=0)) / (- self.neg_queue.sum(axis=0) / self.neg_mask.sum(axis=0))))
        return self.info

    @property
    def info(self):
        return self._info
    
    def adjust(self, new_size, old_index, new_index):
        self.size = new_size
        all_attr = list(self.__dict__.keys())
        for name in all_attr:
            if isinstance(getattr(self, name), np.ndarray):
                nd = len(getattr(self, name).shape)
                if nd == 2:
                    new_attr = np.zeros(shape=(self.window, new_size), dtype=getattr(self,name).dtype)
                    new_attr[:,new_index] = getattr(self,name)[:,old_index]
                    setattr(self, name, new_attr)
                elif nd == 1:
                    new_attr = np.zeros(shape=new_size, dtype=getattr(self,name).dtype)
                    new_attr[new_index] = getattr(self,name)[old_index]
                    setattr(self,name,new_attr)
                else:
                    raise Exception("A higher than 2-dimensional ndarray has appeared.")


class ts_rs:
    def __init__(self, window, size, dtype):
        self.window = window
        self.size = size
        self.cnt = np.zeros(shape=size, dtype=np.int32)
        self.mask_queue = np.zeros(shape=(window, size), dtype=np.int32)
        self.pos_queue = np.zeros(shape=(window, size), dtype=dtype)
        self.pos_sum = np.zeros(shape=size, dtype=dtype)
        self.abs_queue = np.zeros(shape=(window, size), dtype=dtype)
        self.abs_sum = np.zeros(shape=size, dtype=dtype)
        self.count = 0
        self.new_idx = 0
        self._info = np.zeros(shape=size, dtype=dtype)

    def __call__(self, data):
        mask = np.isnan(data) | np.isinf(data)
        new_data = data.copy()
        new_data[mask] = np.nan
        pos_bool = np.where(new_data > 0, 1, 0)
        if self.count >= self.window:
            self.pos_sum += (new_data * pos_bool - self.pos_queue[self.new_idx])
            self.abs_sum += (np.abs(new_data) - self.abs_queue[self.new_idx])
            self.cnt += ((~ mask) - self.mask_queue[self.new_idx])
            self.mask_queue[self.new_idx] = ~ mask
            cnt = np.where(self.cnt > 0, self.cnt, np.nan)
        else:
            self.pos_sum += new_data * pos_bool
            self.abs_sum += np.abs(new_data)
            self.cnt += (~ mask)
            self.mask_queue[self.new_idx] = ~ mask
            cnt = np.where(self.cnt > 0, self.cnt, np.nan)
            self.count += 1
        self.pos_queue[self.new_idx] = new_data * pos_bool
        self.abs_queue[self.new_idx] = np.abs(new_data)
        self._info = self.pos_sum / np.clip(self.abs_sum, 1e-8, None)
        self._info[self.abs_sum  < 1e-8] = np.nan
        self._info[cnt < self.window] = np.nan # 样本数不够的时候返回空值
        self.new_idx += 1
        if self.new_idx > self.window - 1:
            self.new_idx = 0
        return self.info
    
    @property
    def info(self):
        return self._info
    
    def adjust(self, new_size, old_index, new_index):
        self.size = new_size
        all_attr = list(self.__dict__.keys())
        for name in all_attr:
            if isinstance(getattr(self, name), np.ndarray):
                nd = len(getattr(self, name).shape)
                if nd == 2:
                    new_attr = np.zeros(shape=(self.window, new_size), dtype=getattr(self,name).dtype)
                    new_attr[:,new_index] = getattr(self,name)[:,old_index]
                    setattr(self, name, new_attr)
                elif nd == 1:
                    new_attr = np.zeros(shape=new_size, dtype=getattr(self,name).dtype)
                    new_attr[new_index] = getattr(self,name)[old_index]
                    setattr(self,name,new_attr)
                else:
                    raise Exception("A higher than 2-dimensional ndarray has appeared.")


class ts_ema:
    def __init__(self, window, size, dtype):
        self.window = window
        self.size = size
        self.data_queue = np.zeros(shape=(window,size),dtype=dtype)
        self.decay_coef = (1 - 2 / (window + 1))
        self.tail_coef = self.decay_coef ** window
        self.denominator = (1 - self.decay_coef ** window) / (1 - self.decay_coef)
        self.rigid_queue = []
        self.rigid_weight = np.array([self.decay_coef ** i for i in range(window)][::-1]).reshape(-1, 1)
        self.count = 0
        self.new_idx = 0
        self.sma = np.zeros(shape=size, dtype=dtype)
        self._info = np.full(shape=size, fill_value=np.nan, dtype=dtype)

    def __call__(self, data):
        mask = np.isnan(data) | np.isinf(data)
        new_data = data.copy()
        new_data[mask] = 0
        if self.count >= self.window:
            self.sma = (self.sma * self.decay_coef + new_data - self.data_queue[self.new_idx] * self.tail_coef)
            self.data_queue[self.new_idx] = new_data
            self._info = self.sma / self.denominator
        else:
            self.count += 1
            self.data_queue[self.new_idx] = new_data
            self.sma = new_data + self.sma * self.decay_coef
            self._info = self.sma / ((1 - self.decay_coef ** self.count)/(1 - self.decay_coef))
        self.new_idx += 1
        if self.new_idx > self.window - 1:
            self.new_idx = 0
        return self.info
    
    def update_base(self, data):
        new_data = data.copy()
        new_data[np.isnan(new_data) | np.isinf(new_data)] = 0
        self.rigid_queue.append(new_data)
        if self.count >= self.window:
            self.rigid_queue.pop(0)
        else:
            self.count += 1
        return self
    
    def update_info(self):
        if self.count >= self.window:
            self._info = (np.array(self.rigid_queue) * self.rigid_weight).sum(axis=0) / self.denominator
        return self.info

    @property
    def info(self):
        return self._info
    
    def adjust(self, new_size, old_index, new_index):
        self.size = new_size
        
        data_queue = np.zeros(shape=(self.window, new_size), dtype=self.data_queue.dtype)
        data_queue[:,new_index] = self.data_queue[:,old_index]
        self.data_queue = data_queue

        sma = np.zeros(shape=new_size, dtype=self.sma.dtype)
        sma[new_index] = self.sma[old_index]
        self.sma = sma

# ts_sum(x) / ts_sum(y)
class ts_sum_div:
    def __init__(self, window, size, dtype):
        """
        window: 回溯窗口长度
        size: 合约数量
        dtype: 中间状态类型,近似为返回结果类似
        """
        self.window = window
        self.size = size
        self.x_queue = np.zeros(shape=(window, size), dtype=dtype)
        self.y_queue = np.zeros(shape=(window, size), dtype=dtype)
        self.mask_queue = np.zeros(shape=(window, size), dtype=np.int32)
        self.x_sum = np.zeros(shape=size, dtype=dtype)
        self.y_sum = np.zeros(shape=size, dtype=dtype)
        self.count = 0 
        self.new_idx = 0  # 当前更新到的某个窗口
        self._info = np.zeros(shape=size, dtype=dtype)

    def __call__(self, x, y):
        mask = np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y)
        self.x_sum[~mask] += x[~mask]
        self.x_sum -= self.x_queue[self.new_idx]
        self.y_sum[~mask] += y[~mask]
        self.y_sum -= self.y_queue[self.new_idx]
        self.x_queue[self.new_idx][mask] = 0
        self.x_queue[self.new_idx][~mask] = x[~mask]
        self.y_queue[self.new_idx][mask] = 0
        self.y_queue[self.new_idx][~mask] = y[~mask]
        self._info[:] = np.where(self.y_sum == 0, np.nan, self.x_sum / self.y_sum)
        self.new_idx += 1
        if self.new_idx > self.window - 1:
            self.new_idx = 0
        return self.info
    
    def update_base(self, x, y):
        mask = np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y)
        self.x_queue[self.new_idx][~mask] = x[~mask]
        self.x_queue[self.new_idx][mask] = 0
        self.y_queue[self.new_idx][~mask] = y[~mask]
        self.y_queue[self.new_idx][mask] = 0
        self.new_idx += 1
        if self.new_idx > self.window - 1:
            self.new_idx = 0
        return self
    
    def update_info(self):
        self._info = self.x_queue.sum(axis=0) / self.y_queue.sum(axis=0)
        self._info = np.where(np.isinf(self._info), np.nan,self._info)
        return self.info

    @property
    def info(self):
        return self._info
    
    def adjust(self, new_size, old_index, new_index):
        self.size = new_size
        all_attr = list(self.__dict__.keys())
        for name in all_attr:
            if isinstance(getattr(self, name), np.ndarray):
                nd = len(getattr(self, name).shape)
                if nd == 2:
                    new_attr = np.zeros(shape=(self.window, new_size), dtype=getattr(self,name).dtype)
                    new_attr[:,new_index] = getattr(self,name)[:,old_index]
                    setattr(self, name, new_attr)
                elif nd == 1:
                    new_attr = np.zeros(shape=new_size, dtype=getattr(self,name).dtype)
                    new_attr[new_index] = getattr(self,name)[old_index]
                    setattr(self,name,new_attr)
                else:
                    raise Exception("A higher than 2-dimensional ndarray has appeared.")

# 100 * (ts_sum(x) - ts_sum(y)) / (ts_sum(x) + ts_sum(y))
class ts_sum_div_cs:
    def __init__(self, window, size, dtype):
        """
        window: 回溯窗口长度
        size: 合约数量
        dtype: 中间状态类型,近似为返回结果类似
        """
        self.window = window
        self.size = size
        self.x_queue = np.zeros(shape=(window, size), dtype=dtype)
        self.y_queue = np.zeros(shape=(window, size), dtype=dtype)
        self.mask_queue = np.zeros(shape=(window, size), dtype=np.int32)
        self.x_sum = np.zeros(shape=size, dtype=dtype)
        self.y_sum = np.zeros(shape=size, dtype=dtype)
        self.count = 0 
        self.new_idx = 0  # 当前更新到的某个窗口
        self._info = np.zeros(shape=size, dtype=dtype)

    def __call__(self, x, y):
        mask = np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y)
        if self.count >= self.window:
            self.x_sum[~mask] += x[~mask]
            self.x_sum -= self.x_queue[self.new_idx]
            self.y_sum[~mask] += y[~mask]
            self.y_sum -= self.y_queue[self.new_idx]
            
        else:
            self.x_sum[~mask] += x[~mask]
            self.y_sum[~mask] += y[~mask]
            self.count += 1
        self.x_queue[self.new_idx][mask] = 0
        self.x_queue[self.new_idx][~mask] = x[~mask]
        self.y_queue[self.new_idx][mask] = 0
        self.y_queue[self.new_idx][~mask] = y[~mask]
        self._info = (self.x_sum - self.y_sum) / (self.x_sum + self.y_sum)
        self._info = 100 * np.where(np.isinf(self._info),np.nan,self._info)
        self.new_idx += 1
        if self.new_idx > self.window - 1:
            self.new_idx = 0
        return self.info
    
    def update_base(self, x, y):
        mask = np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y)
        self.x_queue[self.new_idx][~mask] = x[~mask]
        self.x_queue[self.new_idx][mask] = 0
        self.y_queue[self.new_idx][~mask] = y[~mask]
        self.y_queue[self.new_idx][mask] = 0
        self.new_idx += 1
        if self.new_idx > self.window - 1:
            self.new_idx = 0
        return self
    
    def update_info(self):
        self._info = (self.x_queue.sum(axis=0) - self.y_queue.sum(axis=0)) / (self.x_queue.sum(axis=0) + self.y_queue.sum(axis=0))
        self._info = 100 * np.where(np.isinf(self._info), np.nan,self._info)
        return self.info

    @property
    def info(self):
        return self._info
    
    def adjust(self, new_size, old_index, new_index):
        self.size = new_size
        all_attr = list(self.__dict__.keys())
        for name in all_attr:
            if isinstance(getattr(self, name), np.ndarray):
                nd = len(getattr(self, name).shape)
                if nd == 2:
                    new_attr = np.zeros(shape=(self.window, new_size), dtype=getattr(self,name).dtype)
                    new_attr[:,new_index] = getattr(self,name)[:,old_index]
                    setattr(self, name, new_attr)
                elif nd == 1:
                    new_attr = np.zeros(shape=new_size, dtype=getattr(self,name).dtype)
                    new_attr[new_index] = getattr(self,name)[old_index]
                    setattr(self,name,new_attr)
                else:
                    raise Exception("A higher than 2-dimensional ndarray has appeared.")

# 100 * abs((ts_sum(x) - ts_sum(y)) / (ts_sum(x) + ts_sum(y)))
class ts_sum_div_cs_abs:
    def __init__(self, window, size, dtype):
        """
        window: 回溯窗口长度
        size: 合约数量
        dtype: 中间状态类型,近似为返回结果类似
        """
        self.window = window
        self.size = size
        self.x_queue = np.zeros(shape=(window, size), dtype=dtype)
        self.y_queue = np.zeros(shape=(window, size), dtype=dtype)
        self.mask_queue = np.zeros(shape=(window, size), dtype=np.int32)
        self.x_sum = np.zeros(shape=size, dtype=dtype)
        self.y_sum = np.zeros(shape=size, dtype=dtype)
        self.count = 0 
        self.new_idx = 0  # 当前更新到的某个窗口
        self._info = np.zeros(shape=size, dtype=dtype)

    def __call__(self, x, y):
        mask = np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y)
        if self.count >= self.window:
            self.x_sum[~mask] += x[~mask]
            self.x_sum -= self.x_queue[self.new_idx]
            self.y_sum[~mask] += y[~mask]
            self.y_sum -= self.y_queue[self.new_idx]
            
        else:
            self.x_sum[~mask] += x[~mask]
            self.y_sum[~mask] += y[~mask]
            self.count += 1
        self.x_queue[self.new_idx][mask] = 0
        self.x_queue[self.new_idx][~mask] = x[~mask]
        self.y_queue[self.new_idx][mask] = 0
        self.y_queue[self.new_idx][~mask] = y[~mask]
        self._info = (self.x_sum - self.y_sum) / (self.x_sum + self.y_sum)
        self._info = np.fabs(100 * np.where(np.isinf(self._info),np.nan,self._info))
        self.new_idx += 1
        if self.new_idx > self.window - 1:
            self.new_idx = 0
        return self.info
    
    def update_base(self, x, y):
        mask = np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y)
        self.x_queue[self.new_idx][~mask] = x[~mask]
        self.x_queue[self.new_idx][mask] = 0
        self.y_queue[self.new_idx][~mask] = y[~mask]
        self.y_queue[self.new_idx][mask] = 0
        self.new_idx += 1
        if self.new_idx > self.window - 1:
            self.new_idx = 0
        return self
    
    def update_info(self):
        self._info = (self.x_queue.sum(axis=0)-self.y_queue.sum(axis=0))/(self.x_queue.sum(axis=0)+self.y_queue.sum(axis=0))
        self._info = np.fabs(100 * np.where(np.isinf(self._info), np.nan,self._info))
        return self.info

    @property
    def info(self):
        return self._info
    
    def adjust(self, new_size, old_index, new_index):
        self.size = new_size
        all_attr = list(self.__dict__.keys())
        for name in all_attr:
            if isinstance(getattr(self, name), np.ndarray):
                nd = len(getattr(self, name).shape)
                if nd == 2:
                    new_attr = np.zeros(shape=(self.window, new_size), dtype=getattr(self,name).dtype)
                    new_attr[:,new_index] = getattr(self,name)[:,old_index]
                    setattr(self, name, new_attr)
                elif nd == 1:
                    new_attr = np.zeros(shape=new_size, dtype=getattr(self,name).dtype)
                    new_attr[new_index] = getattr(self,name)[old_index]
                    setattr(self,name,new_attr)
                else:
                    raise Exception("A higher than 2-dimensional ndarray has appeared.")

# ts_std(a) / ts_std(b)
class ts_std_div:
    def __init__(self, window, size, dtype):
        """
        window: 回溯窗口长度
        size: 合约数量
        dtype: 中间状态类型,近似为返回结果类似
        """
        self.window = window
        self.size = size
        # data queue
        self.x_queue = np.zeros(shape=(window, size), dtype=dtype)
        self.y_queue = np.zeros(shape=(window, size), dtype=dtype)
        self.x_mask_queue = np.zeros(shape=(window, size), dtype=np.int32)
        self.y_mask_queue = np.zeros(shape=(window, size), dtype=np.int32)
        # middle stats
        self.cnt_x = np.zeros(shape=size, dtype=np.int32)
        self.cnt_y = np.zeros(shape=size, dtype=np.int32)
        self.sum_x = np.zeros(shape=size, dtype=dtype)
        self.sum_x2 = np.zeros(shape=size, dtype=dtype)
        self.sum_y = np.zeros(shape=size, dtype=dtype)
        self.sum_y2 = np.zeros(shape=size, dtype=dtype)
        self.count = 0 
        self.new_idx = 0  # 当前更新到的某个窗口
        self._info = np.nan * np.zeros(shape=size, dtype=dtype)

    def __call__(self, x, y):
        x_mask = np.isnan(x) | np.isinf(x)
        y_mask = np.isnan(y) | np.isinf(y)
        if self.count >= self.window:
            self.sum_x[~x_mask] += x[~x_mask]
            self.sum_x -= self.x_queue[self.new_idx]
            self.sum_x2[~x_mask] += x[~x_mask] ** 2
            self.sum_x2 -= self.x_queue[self.new_idx] ** 2
            self.cnt_x += (~x_mask - self.x_mask_queue[self.new_idx])
            self.sum_y[~y_mask] += y[~y_mask]
            self.sum_y -= self.y_queue[self.new_idx]
            self.sum_y2[~y_mask] += y[~y_mask] ** 2
            self.sum_y2 -= self.y_queue[self.new_idx] ** 2
            self.cnt_y += (~y_mask - self.y_mask_queue[self.new_idx])
        else:
            self.sum_x[~x_mask] += x[~x_mask]
            self.sum_x2[~x_mask] += x[~x_mask] ** 2
            self.sum_y[~y_mask] += y[~y_mask]
            self.sum_y2[~y_mask] += y[~y_mask] ** 2
            self.cnt_x += ~x_mask
            self.cnt_y += ~y_mask
            self.count += 1
        self._info = np.sqrt(np.fabs(((self.sum_x2/self.cnt_x) - (self.sum_x/self.cnt_x)**2))) / \
                    np.sqrt(np.fabs(((self.sum_y2/self.cnt_y) - (self.sum_y/self.cnt_y)**2))) 
        self._info[np.isinf(self._info)] = np.nan
        self.x_mask_queue[self.new_idx] = ~x_mask
        self.y_mask_queue[self.new_idx] = ~y_mask
        self.x_queue[self.new_idx][~x_mask] = x[~x_mask]
        self.y_queue[self.new_idx][~y_mask] = y[~y_mask]
        self.new_idx += 1
        if self.new_idx > self.window - 1:
            self.new_idx = 0
        return self.info
    
    def update_base(self, x, y):
        self.x_queue[self.new_idx] = x
        self.y_queue[self.new_idx] = y
        self.new_idx += 1
        if self.new_idx > self.window - 1:
            self.new_idx = 0
        return self

    def update_info(self):
        self._info = np.nanstd(self.x_queue,axis=0) / np.nanstd(self.y_queue,axis=0)
        self._info[np.isinf(self._info)] = np.nan
        return self.info

    @property
    def info(self):
        return self._info
    
    def adjust(self, new_size, old_index, new_index):
        self.size = new_size
        all_attr = list(self.__dict__.keys())
        for name in all_attr:
            if isinstance(getattr(self, name), np.ndarray):
                nd = len(getattr(self, name).shape)
                if nd == 2:
                    new_attr = np.zeros(shape=(self.window, new_size), dtype=getattr(self,name).dtype)
                    new_attr[:,new_index] = getattr(self,name)[:,old_index]
                    setattr(self, name, new_attr)
                elif nd == 1:
                    new_attr = np.zeros(shape=new_size, dtype=getattr(self,name).dtype)
                    new_attr[new_index] = getattr(self,name)[old_index]
                    setattr(self,name,new_attr)
                else:
                    raise Exception("A higher than 2-dimensional ndarray has appeared.")

# 相对前值上涨概率
class ts_up_ratio:
    def __init__(self, window, size, dtype):
        self.window = window
        self.size = size
        self.data_queue = np.zeros(shape=(window, size), dtype=dtype)
        self.up_queue = np.zeros(shape=(window, size), dtype=dtype)
        self.count = 0
        self.new_idx = 0
        self._info = np.zeros(shape=size, dtype=dtype)
    
    def __call__(self, data):
        if self.count >= self.window:
            self._info += (np.where(data > self.data_queue[self.new_idx - 1],1,0) - self.up_queue[self.new_idx]) / self.window
            self.data_queue[self.new_idx] = data + 0
            self.up_queue[self.new_idx] = np.where(data > self.data_queue[self.new_idx - 1], 1, 0)
        else:
            self.data_queue[self.new_idx] = data + 0
            self.up_queue[self.new_idx] = np.where(data > self.data_queue[self.new_idx - 1], 1, 0)
            self._info = ((self._info * self.count) + self.up_queue[self.new_idx]) / (self.count + 1)
            self.count += 1
        self.new_idx += 1
        if self.new_idx > self.window - 1:
            self.new_idx = 0
        return self._info
    
    def update_base(self, data):
        self.data_queue[self.new_idx] = data
        self.up_queue[self.new_idx] = np.where(data > self.data_queue[self.new_idx - 1], 1, 0)
        self.new_idx += 1
        if self.new_idx > self.window - 1:
            self.new_idx = 0
        return self
    
    def update_info(self):
        self._info = self.up_queue.sum(axis=0) / self.window
        return self.info

    @property
    def info(self):
        return self._info
    
    def adjust(self, new_size, old_index, new_index):
        self.size = new_size
        all_attr = list(self.__dict__.keys())
        for name in all_attr:
            if isinstance(getattr(self, name), np.ndarray):
                nd = len(getattr(self, name).shape)
                if nd == 2:
                    new_attr = np.zeros(shape=(self.window, new_size), dtype=getattr(self,name).dtype)
                    new_attr[:,new_index] = getattr(self,name)[:,old_index]
                    setattr(self, name, new_attr)
                elif nd == 1:
                    new_attr = np.zeros(shape=new_size, dtype=getattr(self,name).dtype)
                    new_attr[new_index] = getattr(self,name)[old_index]
                    setattr(self,name,new_attr)
                else:
                    raise Exception("A higher than 2-dimensional ndarray has appeared.")

# ts_avg(short) / ts_avg(long)
class ts_updown:
    def __init__(self, short_window, long_window, size, dtype):
        """
        window: 回溯窗口长度
        size: 合约数量
        dtype: 中间状态类型,近似为返回结果类似
        """
        if long_window < short_window:
            raise Exception("long_window must be longer than short_window")
        self.short_window = short_window
        self.long_window = long_window
        self.window_diff = long_window - short_window
        self.size = size
        self.data_queue = np.zeros(shape=(long_window, size), dtype=dtype)
        self.mask_queue = np.zeros(shape=(long_window, size), dtype=np.int32)
        self.cnt_l = long_window * np.ones(shape=size, dtype=np.int32)
        self.cnt_s = short_window * np.ones(shape=size, dtype=np.int32)
        self.sum_l = np.zeros(shape=size, dtype=dtype)
        self.sum_s = np.zeros(shape=size, dtype=dtype)
        self.count = 0 
        self.new_idx = 0  # 当前更新到的某个窗口
        self._info = np.ones(shape=size, dtype=dtype)

    def __call__(self, data):
        mask = np.isnan(data) | np.isinf(data)
        # 更时间窗口超过window
        if self.count >= self.long_window:
            self.sum_l[~mask] += data[~mask]
            self.sum_s[~mask] += data[~mask]
            self.sum_l -= self.data_queue[self.new_idx]
            self.sum_s -= self.data_queue[self.new_idx - self.short_window]
            self.cnt_l += (np.where(mask,0,1) - self.mask_queue[self.new_idx])
            self.cnt_s += (np.where(mask,0,1) - self.mask_queue[self.new_idx - self.short_window])
            self._info = (self.sum_s / self.cnt_s) / (self.sum_l / self.cnt_l)
            self._info[np.isinf(self._info)] = np.nan
            self.mask_queue[self.new_idx] = np.where(mask,0,1)
        # 更新时间窗口不足window
        else:
            self.sum_l[~mask] += data[~mask] 
            if self.count >= self.long_window - self.short_window: 
                self.sum_s[~mask] += data[~mask] 
            self.count += 1 
        self.data_queue[self.new_idx][mask] = 0 
        self.data_queue[self.new_idx][~mask] = data[~mask] 
        self.mask_queue[self.new_idx] = np.where(mask,0,1)
        self.new_idx += 1 
        if self.new_idx > self.long_window - 1:
            self.new_idx = 0 
        return self.info

    def update_base(self, data):
        mask = np.isnan(data)
        self.data_queue[self.new_idx][mask] = 0
        self.data_queue[self.new_idx][~mask] = data[~mask]
        self.new_idx += 1
        if self.new_idx > self.long_window - 1:
            self.new_idx = 0
        return self
    
    def update_info(self):
        if self.new_idx - self.short_window >= 0:
            s_avg = np.mean(self.data_queue[self.new_idx - self.short_window:self.new_idx],axis=0)
        else:
            s_avg = np.mean(np.append(self.data_queue[:self.new_idx], self.data_queue[self.new_idx-self.short_window:,:],axis=0),axis=0)
        l_avg = np.mean(self.data_queue,axis=0)
        self._info = s_avg / l_avg
        return self.info
    
    @property
    def info(self):
        return self._info
    
    def adjust(self, new_size, old_index, new_index):
        self.size = new_size
        
        data_queue = np.zeros(shape=(self.long_window, new_size), dtype=self.data_queue.dtype)
        data_queue[:,new_index] = self.data_queue[:,old_index]
        self.data_queue = data_queue

        mask_queue = np.zeros(shape=(self.long_window,new_size), dtype=self.mask_queue.dtype)
        mask_queue[:,new_index] = self.mask_queue[:,old_index]
        self.mask_queue = mask_queue

        cnt_l = self.long_window * np.ones(shape=new_size, dtype=np.int32)
        cnt_l[new_index] = self.cnt_l[old_index]
        self.cnt_l = cnt_l

        cnt_s = self.short_window * np.ones(shape=new_size, dtype=np.int32)
        cnt_s[new_index] = self.cnt_s[old_index]
        self.cnt_s = cnt_s

        sum_l = np.zeros(shape=new_size, dtype=self.sum_l.dtype)
        sum_l[new_index] = self.sum_l[old_index]
        self.sum_l = sum_l

        sum_s = np.zeros(shape=new_size, dtype=self.sum_s.dtype)
        sum_s[new_index] = self.sum_s[old_index]
        self.sum_s = sum_s        
         
# 方差
class ts_var:
    def __init__(self, window, size, dtype):
        """
        window: 回溯窗口长度
        size: 合约数量
        dtype: 中间状态类型,近似为返回结果类似
        """
        self.window = window
        self.size = size
        # data queue
        self.data_queue = np.zeros(shape=(window, size), dtype=dtype)
        self.mask_queue = np.zeros(shape=(window, size), dtype=np.int32)
        # middle stats
        self.cnt = np.zeros(shape=size, dtype=np.int32)
        self.sum = np.zeros(shape=size, dtype=dtype)
        self.sum2 = np.zeros(shape=size, dtype=dtype)
        self.count = 0 
        self.new_idx = 0  # 当前更新到的某个窗口
        self._info = None

    def __call__(self, data):
        mask = np.isnan(data) | np.isinf(data)
        temp_data = np.where(mask, 0, data)
        if self.count >= self.window:
            self.sum +=  (temp_data - self.data_queue[self.new_idx])
            self.sum2 += (temp_data ** 2 - self.data_queue[self.new_idx] ** 2)
            self.cnt += (~mask - self.mask_queue[self.new_idx])
        else:
            self.sum[~mask] += data[~mask]
            self.sum2[~mask] += data[~mask] ** 2
            self.cnt += ~mask
            self.count += 1

        self.data_queue[self.new_idx] = temp_data
        self.mask_queue[self.new_idx] = ~mask
        self._info = np.abs((self.sum2/self.cnt-(self.sum/self.cnt)**2))
        self._info[np.isinf(self._info)] = np.nan
        self.new_idx += 1
        if self.new_idx > self.window - 1:
            self.new_idx = 0
        return self._info
    
    def update_base(self, data):
        mask = np.isnan(data) | np.isinf(data)
        self.data_queue[self.new_idx][~mask] = data[~mask]
        self.data_queue[self.new_idx][mask] = np.nan
        self.new_idx += 1
        if self.new_idx > self.window - 1:
            self.new_idx = 0
        return self
    
    def update_info(self):
        self._info = np.nanvar(self.data_queue, axis=0)
        self._info[np.isinf(self._info)] = np.nan
        return self.info
    
    @property
    def info(self):
        return self._info
    
    def adjust(self, new_size, old_index, new_index):
        self.size = new_size
        all_attr = list(self.__dict__.keys())
        for name in all_attr:
            if isinstance(getattr(self, name), np.ndarray):
                nd = len(getattr(self, name).shape)
                if nd == 2:
                    new_attr = np.zeros(shape=(self.window, new_size), dtype=getattr(self,name).dtype)
                    new_attr[:,new_index] = getattr(self,name)[:,old_index]
                    setattr(self, name, new_attr)
                elif nd == 1:
                    new_attr = np.zeros(shape=new_size, dtype=getattr(self,name).dtype)
                    new_attr[new_index] = getattr(self,name)[old_index]
                    setattr(self,name,new_attr)
                else:
                    raise Exception("A higher than 2-dimensional ndarray has appeared.")
         
# 标准差
class ts_std:
    def __init__(self, window, size, dtype):
        """
        window: 回溯窗口长度
        size: 合约数量
        dtype: 中间状态类型,近似为返回结果类似
        """
        self.window = window
        self.size = size
        # data queue
        self.data_queue = np.zeros(shape=(window, size), dtype=dtype)
        self.mask_queue = np.zeros(shape=(window, size), dtype=np.int32)
        # middle stats
        self.cnt = np.zeros(shape=size, dtype=np.int32)
        self.sum = np.zeros(shape=size, dtype=dtype)
        self.sum2 = np.zeros(shape=size, dtype=dtype)
        self.count = 0 
        self.new_idx = 0  # 当前更新到的某个窗口
        self._info = None

    def __call__(self, data):
        mask = np.isnan(data) | np.isinf(data)
        temp_data = np.where(mask, 0, data)
        if self.count >= self.window:
            self.sum +=  (temp_data - self.data_queue[self.new_idx])
            self.sum2 += (temp_data ** 2 - self.data_queue[self.new_idx] ** 2)
            self.cnt += (~mask - self.mask_queue[self.new_idx])
        else:
            self.sum[~mask] += data[~mask]
            self.sum2[~mask] += data[~mask] ** 2
            self.cnt += ~mask
            self.count += 1

        self.data_queue[self.new_idx] = temp_data
        self.mask_queue[self.new_idx] = ~mask
        self._info = np.sqrt(np.abs((self.sum2/self.cnt-(self.sum/self.cnt)**2)))
        self._info[np.isinf(self._info)] = np.nan
        self.new_idx += 1
        if self.new_idx > self.window - 1:
            self.new_idx = 0
        return self._info
    
    def update_base(self, data):
        mask = np.isnan(data) | np.isinf(data)
        self.data_queue[self.new_idx][~mask] = data[~mask]
        self.data_queue[self.new_idx][mask] = np.nan
        self.new_idx += 1
        if self.new_idx > self.window - 1:
            self.new_idx = 0
        return self
    
    def update_info(self):
        self._info = np.nanstd(self.data_queue, axis=0)
        self._info[np.isinf(self._info)] = np.nan
        return self.info
    
    @property
    def info(self):
        return self._info
    
    def adjust(self, new_size, old_index, new_index):
        self.size = new_size
        all_attr = list(self.__dict__.keys())
        for name in all_attr:
            if isinstance(getattr(self, name), np.ndarray):
                nd = len(getattr(self, name).shape)
                if nd == 2:
                    new_attr = np.zeros(shape=(self.window, new_size), dtype=getattr(self,name).dtype)
                    new_attr[:,new_index] = getattr(self,name)[:,old_index]
                    setattr(self, name, new_attr)
                elif nd == 1:
                    new_attr = np.zeros(shape=new_size, dtype=getattr(self,name).dtype)
                    new_attr[new_index] = getattr(self,name)[old_index]
                    setattr(self,name,new_attr)
                else:
                    raise Exception("A higher than 2-dimensional ndarray has appeared.")
            
# 偏度
class ts_skew:
    def __init__(self, window, size, dtype):
        """
        window: 回溯窗口长度
        size: 合约数量
        dtype: 中间状态类型,近似为返回结果类似
        """
        self.window = window
        self.size = size
        # data queue
        self.data_queue = np.zeros(shape=(window, size), dtype=dtype)
        self.mask_queue = np.zeros(shape=(window, size), dtype=np.int32)
        # middle stats
        self.cnt = np.zeros(shape=size, dtype=np.int32)
        self.sum = np.zeros(shape=size, dtype=dtype)
        self.sum2 = np.zeros(shape=size, dtype=dtype)
        self.sum3 = np.zeros(shape=size, dtype=dtype)
        self.count = 0 
        self.new_idx = 0  # 当前更新到的某个窗口
        self._info = None

    def __call__(self, data):
        mask = np.isnan(data) | np.isinf(data)
        temp_data = np.where(mask, 0, data)
        if self.count >= self.window:
            self.sum +=  (temp_data - self.data_queue[self.new_idx])
            self.sum2 += (temp_data ** 2 - self.data_queue[self.new_idx] ** 2)
            self.sum3 += (temp_data ** 3 - self.data_queue[self.new_idx] ** 3)
            self.cnt += (~mask - self.mask_queue[self.new_idx])
        else:
            self.sum[~mask] += data[~mask]
            self.sum2[~mask] += data[~mask] ** 2
            self.sum3[~mask] += data[~mask] ** 3
            self.cnt += ~mask
            self.count += 1
        self.data_queue[self.new_idx] = temp_data
        self.mask_queue[self.new_idx] = ~mask
        cnt = np.where(self.cnt == 0, np.nan, self.cnt)
        temp_data = (np.fabs((self.sum2/(cnt)) - (self.sum/(cnt)) ** 2) ** (3/2))
        self._info = (self.sum3/(cnt)-3*(self.sum/(cnt))*(self.sum2/(cnt))+2*(self.sum/(cnt))**3)/temp_data
        self.new_idx += 1
        if self.new_idx > self.window - 1:
            self.new_idx = 0
        return self._info
    
    def update_base(self, data):
        mask = np.isnan(data) | np.isinf(data)
        self.data_queue[self.new_idx][~mask] = data[~mask]
        self.data_queue[self.new_idx][mask] = np.nan
        self.new_idx += 1
        if self.new_idx > self.window - 1:
            self.new_idx = 0
        return self
    
    def update_info(self):
        self._info = skew(self.data_queue, axis=0)
        return self.info
    
    @property
    def info(self):
        return self._info
    
    def adjust(self, new_size, old_index, new_index):
        self.size = new_size
        all_attr = list(self.__dict__.keys())
        for name in all_attr:
            if isinstance(getattr(self, name), np.ndarray):
                nd = len(getattr(self, name).shape)
                if nd == 2:
                    new_attr = np.zeros(shape=(self.window, new_size), dtype=getattr(self,name).dtype)
                    new_attr[:,new_index] = getattr(self,name)[:,old_index]
                    setattr(self, name, new_attr)
                elif nd == 1:
                    new_attr = np.zeros(shape=new_size, dtype=getattr(self,name).dtype)
                    new_attr[new_index] = getattr(self,name)[old_index]
                    setattr(self,name,new_attr)
                else:
                    raise Exception("A higher than 2-dimensional ndarray has appeared.")
           
# 峰度
class ts_kurt:
    def __init__(self, window, size, dtype):
        """
        window: 回溯窗口长度
        size: 合约数量
        dtype: 中间状态类型,近似为返回结果类似
        """
        self.window = window
        self.size = size
        # data queue
        self.data_queue = np.zeros(shape=(window, size), dtype=dtype)
        self.mask_queue = np.zeros(shape=(window, size), dtype=np.int32)
        # middle stats
        self.cnt = np.zeros(shape=size, dtype=np.int32)
        self.sum = np.zeros(shape=size, dtype=dtype)
        self.sum2 = np.zeros(shape=size, dtype=dtype)
        self.sum3 = np.zeros(shape=size, dtype=dtype)
        self.sum4 = np.zeros(shape=size, dtype=dtype)
        self.count = 0 
        self.new_idx = 0  # 当前更新到的某个窗口
        self._info = None

    def __call__(self, data):
        mask = np.isnan(data) | np.isinf(data)
        temp_data = np.where(mask, 0, data)
        if self.count >= self.window:
            self.sum +=  (temp_data - self.data_queue[self.new_idx])
            self.sum2 += (temp_data ** 2 - self.data_queue[self.new_idx] ** 2)
            self.sum3 += (temp_data ** 3 - self.data_queue[self.new_idx] ** 3)
            self.sum4 += (temp_data ** 4 - self.data_queue[self.new_idx] ** 4)
            self.cnt += (~mask - self.mask_queue[self.new_idx])
        else:
            self.sum[~mask] += data[~mask]
            self.sum2[~mask] += data[~mask] ** 2
            self.sum3[~mask] += data[~mask] ** 3
            self.sum4[~mask] += data[~mask] ** 4
            self.cnt += ~mask
            self.count += 1
        self.data_queue[self.new_idx] = temp_data
        self.mask_queue[self.new_idx] = ~mask
        cnt = np.where(self.cnt == 0, np.nan, self.cnt)
        temp_data = np.fabs((self.sum2 - (cnt) * (self.sum/cnt) ** 2) ** 2)
        self._info = self.window * (self.sum4 - 4 * self.sum3 * (self.sum/cnt) + 6 * self.sum2 * (self.sum/cnt) ** 2 - 3 * cnt * (self.sum/cnt) ** 4) / temp_data - 3
        self.new_idx += 1
        if self.new_idx > self.window - 1:
            self.new_idx = 0
        return self._info
    
    def update_base(self, data):
        mask = np.isnan(data) | np.isinf(data)
        self.data_queue[self.new_idx][~mask] = data[~mask]
        self.data_queue[self.new_idx][mask] = np.nan
        self.new_idx += 1
        if self.new_idx > self.window - 1:
            self.new_idx = 0
        return self
    
    def update_info(self):
        self._info = kurtosis(self.data_queue, axis=0)
        return self.info
    
    @property
    def info(self):
        return self._info
    
    def adjust(self, new_size, old_index, new_index):
        self.size = new_size
        all_attr = list(self.__dict__.keys())
        for name in all_attr:
            if isinstance(getattr(self, name), np.ndarray):
                nd = len(getattr(self, name).shape)
                if nd == 2:
                    new_attr = np.zeros(shape=(self.window, new_size), dtype=getattr(self,name).dtype)
                    new_attr[:,new_index] = getattr(self,name)[:,old_index]
                    setattr(self, name, new_attr)
                elif nd == 1:
                    new_attr = np.zeros(shape=new_size, dtype=getattr(self,name).dtype)
                    new_attr[new_index] = getattr(self,name)[old_index]
                    setattr(self,name,new_attr)
                else:
                    raise Exception("A higher than 2-dimensional ndarray has appeared.")
          
# 均值 / 标准差
class ts_tcv:
    def __init__(self, window, size, dtype):
        """
        window: 回溯窗口长度
        size: 合约数量
        dtype: 中间状态类型,近似为返回结果类似
        """
        self.window = window
        self.size = size
        # data queue
        self.data_queue = np.zeros(shape=(window, size), dtype=dtype)
        self.mask_queue = np.zeros(shape=(window, size), dtype=np.int32)
        # middle stats
        self.cnt = np.zeros(shape=size, dtype=np.int32)
        self.sum = np.zeros(shape=size, dtype=dtype)
        self.sum2 = np.zeros(shape=size, dtype=dtype)
        self.count = 0 
        self.new_idx = 0  # 当前更新到的某个窗口
        self._info = None

    def __call__(self, data):
        mask = np.isnan(data) | np.isinf(data)
        temp_data = np.where(mask, 0, data)
        if self.count >= self.window:
            self.sum +=  (temp_data - self.data_queue[self.new_idx])
            self.sum2 += (temp_data ** 2 - self.data_queue[self.new_idx] ** 2)
            self.cnt += (~mask - self.mask_queue[self.new_idx])
        else:
            self.sum[~mask] += data[~mask]
            self.sum2[~mask] += data[~mask] ** 2
            self.cnt += ~mask
            self.count += 1

        self.data_queue[self.new_idx] = temp_data
        self.mask_queue[self.new_idx] = ~mask
        cnt = np.where(self.cnt == 0, np.nan, self.cnt)
        self._info = (self.sum/cnt) / np.sqrt(np.abs((self.sum2/cnt - (self.sum / cnt)**2)))
        self._info[np.isinf(self._info)] = np.nan
        self.new_idx += 1
        if self.new_idx > self.window - 1:
            self.new_idx = 0
        return self._info
    
    def update_base(self, data):
        mask = np.isnan(data) | np.isinf(data)
        self.data_queue[self.new_idx][~mask] = data[~mask]
        self.data_queue[self.new_idx][mask] = np.nan
        self.new_idx += 1
        if self.new_idx > self.window - 1:
            self.new_idx = 0
        return self
    
    def update_info(self):
        self._info = np.nanmean(self.data_queue, axis=0) / np.nanstd(self.data_queue, axis=0)
        self._info[np.isinf(self._info)] = np.nan
        return self.info
    
    @property
    def info(self):
        return self._info
    
    def adjust(self, new_size, old_index, new_index):
        self.size = new_size
        all_attr = list(self.__dict__.keys())
        for name in all_attr:
            if isinstance(getattr(self, name), np.ndarray):
                nd = len(getattr(self, name).shape)
                if nd == 2:
                    new_attr = np.zeros(shape=(self.window, new_size), dtype=getattr(self,name).dtype)
                    new_attr[:,new_index] = getattr(self,name)[:,old_index]
                    setattr(self, name, new_attr)
                elif nd == 1:
                    new_attr = np.zeros(shape=new_size, dtype=getattr(self,name).dtype)
                    new_attr[new_index] = getattr(self,name)[old_index]
                    setattr(self,name,new_attr)
                else:
                    raise Exception("A higher than 2-dimensional ndarray has appeared.")
          
# 标准差 / 均值
class ts_cv:
    def __init__(self, window, size, dtype):
        """
        window: 回溯窗口长度
        size: 合约数量
        dtype: 中间状态类型,近似为返回结果类似
        """
        self.window = window
        self.size = size
        # data queue
        self.data_queue = np.zeros(shape=(window, size), dtype=dtype)
        self.mask_queue = np.zeros(shape=(window, size), dtype=np.int32)
        # middle stats
        self.cnt = np.zeros(shape=size, dtype=np.int32)
        self.sum = np.zeros(shape=size, dtype=dtype)
        self.sum2 = np.zeros(shape=size, dtype=dtype)
        self.count = 0 
        self.new_idx = 0  # 当前更新到的某个窗口
        self._info = None

    def __call__(self, data):
        mask = np.isnan(data) | np.isinf(data)
        temp_data = np.where(mask, 0, data)
        if self.count >= self.window:
            self.sum +=  (temp_data - self.data_queue[self.new_idx])
            self.sum2 += (temp_data ** 2 - self.data_queue[self.new_idx] ** 2)
            self.cnt += (~mask - self.mask_queue[self.new_idx])
        else:
            self.sum[~mask] += data[~mask]
            self.sum2[~mask] += data[~mask] ** 2
            self.cnt += ~mask
            self.count += 1

        self.data_queue[self.new_idx] = temp_data
        self.mask_queue[self.new_idx] = ~mask
        cnt = np.where(self.cnt == 0, np.nan, self.cnt)
        self._info = np.sqrt(np.abs((self.sum2/cnt - (self.sum / cnt)**2))) / (self.sum/cnt) 
        self._info[np.isinf(self._info)] = np.nan
        self.new_idx += 1
        if self.new_idx > self.window - 1:
            self.new_idx = 0
        return self._info
    
    def update_base(self, data):
        mask = np.isnan(data) | np.isinf(data)
        self.data_queue[self.new_idx][~mask] = data[~mask]
        self.data_queue[self.new_idx][mask] = np.nan
        self.new_idx += 1
        if self.new_idx > self.window - 1:
            self.new_idx = 0
        return self
    
    def update_info(self):
        self._info = np.nanstd(self.data_queue, axis=0) / np.nanmean(self.data_queue, axis=0) 
        self._info[np.isinf(self._info)] = np.nan
        return self.info
    
    @property
    def info(self):
        return self._info
    
    def adjust(self, new_size, old_index, new_index):
        self.size = new_size
        all_attr = list(self.__dict__.keys())
        for name in all_attr:
            if isinstance(getattr(self, name), np.ndarray):
                nd = len(getattr(self, name).shape)
                if nd == 2:
                    new_attr = np.zeros(shape=(self.window, new_size), dtype=getattr(self,name).dtype)
                    new_attr[:,new_index] = getattr(self,name)[:,old_index]
                    setattr(self, name, new_attr)
                elif nd == 1:
                    new_attr = np.zeros(shape=new_size, dtype=getattr(self,name).dtype)
                    new_attr[new_index] = getattr(self,name)[old_index]
                    setattr(self,name,new_attr)
                else:
                    raise Exception("A higher than 2-dimensional ndarray has appeared.")
       
# 标准差 / (1 + abs(均值))  # TODO 检查一下
class ts_cv_abs:
    def __init__(self, window, size, dtype):
        """
        window: 回溯窗口长度
        size: 合约数量
        dtype: 中间状态类型,近似为返回结果类似
        """
        self.window = window
        self.size = size
        # data queue
        self.data_queue = np.zeros(shape=(window, size), dtype=dtype)
        self.mask_queue = np.zeros(shape=(window, size), dtype=np.int32)
        # middle stats
        self.cnt = np.zeros(shape=size, dtype=np.int32)
        self.sum = np.zeros(shape=size, dtype=dtype)
        self.sum2 = np.zeros(shape=size, dtype=dtype)
        self.count = 0 
        self.new_idx = 0  # 当前更新到的某个窗口
        self._info = None

    def __call__(self, data):
        mask = np.isnan(data) | np.isinf(data)
        temp_data = np.where(mask, 0, data)
        if self.count >= self.window:
            self.sum +=  (temp_data - self.data_queue[self.new_idx])
            self.sum2 += (temp_data ** 2 - self.data_queue[self.new_idx] ** 2)
            self.cnt += (~mask - self.mask_queue[self.new_idx])
        else:
            self.sum[~mask] += data[~mask]
            self.sum2[~mask] += data[~mask] ** 2
            self.cnt += ~mask
            self.count += 1

        self.data_queue[self.new_idx] = temp_data
        self.mask_queue[self.new_idx] = ~mask
        cnt = np.where(self.cnt == 0, np.nan, self.cnt)
        self._info = np.sqrt(np.abs((self.sum2/cnt - (self.sum / cnt)**2))) / (1+np.fabs((self.sum/cnt)))
        self._info[np.isinf(self._info)] = np.nan
        self.new_idx += 1
        if self.new_idx > self.window - 1:
            self.new_idx = 0
        return self._info
    
    def update_base(self, data):
        mask = np.isnan(data) | np.isinf(data)
        self.data_queue[self.new_idx][~mask] = data[~mask]
        self.data_queue[self.new_idx][mask] = np.nan
        self.new_idx += 1
        if self.new_idx > self.window - 1:
            self.new_idx = 0
        return self
    
    def update_info(self):
        self._info = np.nanstd(self.data_queue, axis=0) / (1 + np.fabs(np.nanmean(self.data_queue, axis=0)))
        self._info[np.isinf(self._info)] = np.nan
        return self.info
    
    @property
    def info(self):
        return self._info
    
    def adjust(self, new_size, old_index, new_index):
        self.size = new_size
        all_attr = list(self.__dict__.keys())
        for name in all_attr:
            if isinstance(getattr(self, name), np.ndarray):
                nd = len(getattr(self, name).shape)
                if nd == 2:
                    new_attr = np.zeros(shape=(self.window, new_size), dtype=getattr(self,name).dtype)
                    new_attr[:,new_index] = getattr(self,name)[:,old_index]
                    setattr(self, name, new_attr)
                elif nd == 1:
                    new_attr = np.zeros(shape=new_size, dtype=getattr(self,name).dtype)
                    new_attr[new_index] = getattr(self,name)[old_index]
                    setattr(self,name,new_attr)
                else:
                    raise Exception("A higher than 2-dimensional ndarray has appeared.")

# 滑动相关性 0.15秒/1000个因子/5000个股票/一个截面
class ts_corr:
    def __init__(self, window, size, dtype):
        """
        window: 回溯窗口长度
        size: 合约数量
        dtype: 中间状态类型,近似为返回结果类似
        """
        self.window = window
        self.size = size
        # data queue
        self.x_data_queue = np.zeros(shape=(window, size), dtype=dtype)
        self.y_data_queue = np.zeros(shape=(window, size), dtype=dtype)
        self.mask_queue = np.zeros(shape=(window, size), dtype=np.int32)
        # middle stats
        self.cnt = np.zeros(shape=size, dtype=np.int32)
        self.x_sum = np.zeros(shape=size, dtype=dtype)
        self.x_sum2 = np.zeros(shape=size, dtype=dtype)
        self.y_sum = np.zeros(shape=size, dtype=dtype)
        self.y_sum2 = np.zeros(shape=size, dtype=dtype)
        self.xy_sum = np.zeros(shape=size, dtype=dtype)
        self.count = 0 
        self.new_idx = 0  # 当前更新到的某个窗口
        self._info = None

    def __call__(self, x, y):
        # 计算过程中需要确保所有中间状态不为异常值
        mask = np.isnan(x) | np.isnan(y)
        new_x = x.copy()
        new_x[mask] = 0
        new_y = y.copy()
        new_y[mask] = 0
        # 更时间窗口超过window
        if self.count >= self.window:
            self.cnt += ((~ mask) - self.mask_queue[self.new_idx])
            self.x_sum += (new_x - self.x_data_queue[self.new_idx])
            self.x_sum2 += (new_x**2 - self.x_data_queue[self.new_idx]**2)
            self.y_sum += (new_y - self.y_data_queue[self.new_idx])
            self.y_sum2 += (new_y**2 - self.y_data_queue[self.new_idx]**2)
            self.xy_sum += ((new_x * new_y) - (self.x_data_queue[self.new_idx] * self.y_data_queue[self.new_idx]))
            cnt = np.where(self.cnt > 0, self.cnt, np.nan)
            self.x_data_queue[self.new_idx] = new_x
            self.y_data_queue[self.new_idx] = new_y
            self.mask_queue[self.new_idx] = ~ mask
        # 更新时间窗口不足window
        else:
            self.x_data_queue[self.new_idx] = new_x
            self.y_data_queue[self.new_idx] = new_y
            self.mask_queue[self.new_idx] = ~ mask
            self.cnt += ~ mask
            self.x_sum += new_x
            self.x_sum2 += new_x ** 2
            self.y_sum += new_y
            self.y_sum2 += new_y ** 2
            self.xy_sum += (new_x * new_y)
            cnt = np.where(self.cnt > 0, self.cnt, np.nan)
            self.count += 1
        # 计算相关系数 corr = cov / (x_std * y_std)
        cov = (self.xy_sum/cnt) - (self.x_sum/cnt) * (self.y_sum/cnt)
        x_std = np.sqrt(np.abs((self.x_sum2 - (self.x_sum ** 2) / cnt) / cnt))
        y_std = np.sqrt(np.abs((self.y_sum2 - (self.y_sum ** 2) / cnt) / cnt))
        self._info = cov / (x_std * y_std)
        self._info[(self.cnt <= self.window) | (np.isinf(self._info))] = np.nan
        # 更新坐标
        self.new_idx += 1 
        if self.new_idx > self.window - 1:
            self.new_idx = 0
        return self.info
    
    def update_base(self, x, y):
        self.x_data_queue[self.new_idx] = x
        self.y_data_queue[self.new_idx] = y
        self.mask_queue[self.new_idx] = np.isnan(x) | np.isnan(y)
        self.new_idx += 1 
        if self.new_idx > self.window - 1:
            self.new_idx = 0
        return self
    
    def update_info(self):
        self._info = pd.DataFrame(self.x_data_queue).T.corrwith(pd.DataFrame(self.y_data_queue).T, method='spearman').values
        return self.info
    
    @property
    def info(self):
        return self._info
    
    def adjust(self, new_size, old_index, new_index):
        self.size = new_size
        all_attr = list(self.__dict__.keys())
        for name in all_attr:
            if isinstance(getattr(self, name), np.ndarray):
                nd = len(getattr(self, name).shape)
                if nd == 2:
                    new_attr = np.zeros(shape=(self.window, new_size), dtype=getattr(self,name).dtype)
                    new_attr[:,new_index] = getattr(self,name)[:,old_index]
                    setattr(self, name, new_attr)
                elif nd == 1:
                    new_attr = np.zeros(shape=new_size, dtype=getattr(self,name).dtype)
                    new_attr[new_index] = getattr(self,name)[old_index]
                    setattr(self,name,new_attr)
                else:
                    raise Exception("A higher than 2-dimensional ndarray has appeared.")
            
# 自相关
class ts_delay_corr:
    def __init__(self, window, delay_window, size, dtype):
        """
        window: 回溯窗口长度
        size: 合约数量
        dtype: 中间状态类型,近似为返回结果类似
        """
        self.window = window
        self.delay_window = delay_window
        self.size = size
        self.ts_corr = ts_corr(window=window, size=size, dtype=dtype)
        self.ts_delay = ts_delay(window=delay_window, size=size, dtype=dtype)
        self.data_queue1 = np.zeros(shape=(window,size),dtype=dtype)
        self.data_queue2 = np.zeros(shape=(window,size),dtype=dtype)
        self.count = 0
        self.new_idx = 0
        self._info = np.zeros(shape=size, dtype=dtype)

    def __call__(self, data):
        self.ts_delay.__call__(data)
        self._info = self.ts_corr.__call__(data, self.ts_delay.info)
        return self.info
    
    def update_base(self, data):
        self.data_queue1[self.new_idx] = data
        self.data_queue2[self.new_idx] = self.data_queue1[self.new_idx - self.delay_window]
        self.new_idx += 1
        if self.new_idx > self.window - 1:
            self.new_idx = 0
        return self
    
    def update_info(self):
        self._info = pd.DataFrame(self.data_queue1).corrwith(pd.DataFrame(self.data_queue2), axis=0).values
        return self.info
    
    @property
    def info(self):
        return self._info
   
    def adjust(self, new_size, old_index, new_index):
        self.size = new_size
        self.ts_corr.adjust(new_size, old_index, new_index)  ## !! 不同于其他adjust函数
        self.ts_delay.adjust(new_size, old_index, new_index)
        all_attr = list(self.__dict__.keys())
        for name in all_attr:
            if isinstance(getattr(self, name), np.ndarray):
                nd = len(getattr(self, name).shape)
                if nd == 2:
                    new_attr = np.zeros(shape=(self.window, new_size), dtype=getattr(self,name).dtype)
                    new_attr[:,new_index] = getattr(self,name)[:,old_index]
                    setattr(self, name, new_attr)
                elif nd == 1:
                    new_attr = np.zeros(shape=new_size, dtype=getattr(self,name).dtype)
                    new_attr[new_index] = getattr(self,name)[old_index]
                    setattr(self,name,new_attr)
                else:
                    raise Exception("A higher than 2-dimensional ndarray has appeared.")

                
# 滑动协方差
class ts_cov:
    def __init__(self, window, size, dtype):
        """
        window: 回溯窗口长度
        size: 合约数量
        dtype: 中间状态类型,近似为返回结果类似
        """
        self.window = window
        self.size = size
        # data queue
        self.x_data_queue = np.zeros(shape=(window, size), dtype=dtype)
        self.y_data_queue = np.zeros(shape=(window, size), dtype=dtype)
        self.mask_queue = np.zeros(shape=(window, size), dtype=np.int32)
        # middle stats
        self.cnt = np.zeros(shape=size, dtype=np.int32)
        self.x_sum = np.zeros(shape=size, dtype=dtype)
        self.y_sum = np.zeros(shape=size, dtype=dtype)
        self.xy_sum = np.zeros(shape=size, dtype=dtype)
        self.count = 0 
        self.new_idx = 0  # 当前更新到的某个窗口
        self._info = None

    def __call__(self, x, y):
        # 计算过程中需要确保所有中间状态不为异常值
        mask = np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y)
        new_x = x.copy()
        new_x[mask] = 0
        new_y = y.copy()
        new_y[mask] = 0
        # 更时间窗口超过window
        if self.count >= self.window:
            self.cnt += ((~ mask) - self.mask_queue[self.new_idx])
            self.x_sum += (new_x - self.x_data_queue[self.new_idx])
            self.y_sum += (new_y - self.y_data_queue[self.new_idx])
            self.xy_sum += ((new_x * new_y) - (self.x_data_queue[self.new_idx] * self.y_data_queue[self.new_idx]))
            cnt = np.where(self.cnt > 0, self.cnt, np.nan)
            self.x_data_queue[self.new_idx] = new_x
            self.y_data_queue[self.new_idx] = new_y
            self.mask_queue[self.new_idx] = ~ mask
        # 更新时间窗口不足window
        else:
            self.x_data_queue[self.new_idx] = new_x
            self.y_data_queue[self.new_idx] = new_y
            self.mask_queue[self.new_idx] = ~ mask
            self.cnt += ~ mask
            self.x_sum += new_x
            self.y_sum += new_y
            self.xy_sum += (new_x * new_y)
            cnt = np.where(self.cnt > 0, self.cnt, np.nan)
            self.count += 1
        self._info = (self.xy_sum/cnt) - (self.x_sum/cnt) * (self.y_sum/cnt)
        self._info[np.isinf(self._info)] = np.nan
        # 更新坐标
        self.new_idx += 1 
        if self.new_idx > self.window - 1:
            self.new_idx = 0
        return self.info
    
    def update_base(self, x, y):
        self.x_data_queue[self.new_idx] = x
        self.y_data_queue[self.new_idx] = y
        self.mask_queue[self.new_idx] = np.isnan(x) | np.isnan(y)
        self.new_idx += 1 
        if self.new_idx > self.window - 1:
            self.new_idx = 0
        return self
    
    def update_info(self):
        self._info = (self.x_data_queue*self.y_data_queue).mean(axis=0)-(self.x_data_queue.mean(axis=0)*self.y_data_queue.mean(axis=0))
        return self.info
    
    @property
    def info(self):
        return self._info
    
    def adjust(self, new_size, old_index, new_index):
        self.size = new_size
        all_attr = list(self.__dict__.keys())
        for name in all_attr:
            if isinstance(getattr(self, name), np.ndarray):
                nd = len(getattr(self, name).shape)
                if nd == 2:
                    new_attr = np.zeros(shape=(self.window, new_size), dtype=getattr(self,name).dtype)
                    new_attr[:,new_index] = getattr(self,name)[:,old_index]
                    setattr(self, name, new_attr)
                elif nd == 1:
                    new_attr = np.zeros(shape=new_size, dtype=getattr(self,name).dtype)
                    new_attr[new_index] = getattr(self,name)[old_index]
                    setattr(self,name,new_attr)
                else:
                    raise Exception("A higher than 2-dimensional ndarray has appeared.")
                 
# region 涉及比较,该算符不适合2维数据,时间复杂度较高
# ==============================================
# 滑动最大值 6秒/1000个因子/5000个股票/一个截面,__call__为12s
# 这三个时间复杂度 股票数 * O(n/2)
class ts_max:
    def __init__(self, window, size, dtype):
        """
        window: 回溯窗口长度
        size: 合约数量
        dtype: 中间状态类型,近似为返回结果类似
        """
        self.window = window
        self.size = size
        self.data_queue = np.zeros(shape=(window, size), dtype=dtype)
        self.queue = [[] for _ in range(size)]
        self.pos_queue = [deque() for _ in range(size)]
        self._info = np.zeros(shape=size, dtype=dtype)
        self.new_idx = 0
        self.count = np.zeros(shape=size, dtype=np.int32)

    def __call__(self, data):
        mask = ~(np.isnan(data) | np.isinf(data))
        # 更时间窗口超过window
        for idx, val in enumerate(data):
            if mask[idx]:
                while self.pos_queue[idx] and (self.queue[idx][self.pos_queue[idx][-1]] <= val):
                    self.pos_queue[idx].pop()
                self.queue[idx].append(val)
                self.pos_queue[idx].append(self.count[idx])
                if self.count[idx] - self.pos_queue[idx][0] >= self.window:
                    self.pos_queue[idx].popleft()
                self._info[idx] = self.queue[idx][self.pos_queue[idx][0]]
            else:
                while self.pos_queue[idx] and (np.isinf(self.queue[idx][self.pos_queue[idx][-1]])):
                    self.pos_queue[idx].pop()
                self.queue[idx].append(-np.inf)
                self.pos_queue[idx].append(self.count[idx])
                if self.count[idx] - self.pos_queue[idx][0] >= self.window:
                    self.pos_queue[idx].popleft()
                if np.isinf(self.queue[idx][self.pos_queue[idx][0]]):
                    self._info[idx] = np.nan
                else:
                    self._info[idx] = self.queue[idx][self.pos_queue[idx][0]]
            self.count[idx] += 1
        return self.info
    
    def update_base(self, data):
        mask = np.isnan(data)
        self.data_queue[self.new_idx][mask] = - np.inf
        self.data_queue[self.new_idx][~mask] = data[~mask]
        self.new_idx += 1
        if self.new_idx > self.window - 1:
            self.new_idx = 0
        return self
    
    def update_info(self):
        self._info = np.max(self.data_queue,axis=0)
        return self.info
    
    @property
    def info(self):
        return self._info

    def adjust(self, new_size, old_index, new_index):
        self.size = new_size
        idx_list = range(len(self.queue))
        queue = [[] for _ in idx_list]
        pos_queue = [deque() for _ in idx_list]
        count = np.zeros_like(self.count)
        info = np.zeros(shape=new_size, dtype=self._info.dtype)
        for idx in idx_list:
            if len(self.queue) > self.window:
                _pos_diff = len(self.queue[idx]) - self.window
                queue[idx] = self.queue[idx][-self.window:]
                for _,j in enumerate(self.pos_queue[idx]):
                    pos_queue[idx].append(j - _pos_diff)
                count[idx] = self.window
            else:
                queue[idx] = self.queue[idx]
                pos_queue[idx] = self.pos_queue[idx]
                count[idx] = self.count[idx]
        self.queue = [[] for _ in range(new_size)]
        self.pos_queue = [deque() for _ in range(new_size)]
        self.count = np.zeros(shape=new_size, dtype=np.int32)
        for i, j  in zip(new_index, old_index):
            self.queue[i] = queue[j]
            self.pos_queue[i] = pos_queue[j]
            self.count[i] = count[j]
        info[new_index] = self._info[old_index]
        self._info = info


class ts_min:
    def __init__(self, window, size, dtype):
        """
        window: 回溯窗口长度
        size: 合约数量
        dtype: 中间状态类型,近似为返回结果类似
        """
        self.window = window
        self.size = size
        self.data_queue = np.zeros(shape=(window, size), dtype=dtype)
        self.queue = [[] for _ in range(size)]
        self.pos_queue = [deque() for _ in range(size)]
        self._info = np.zeros(shape=size, dtype=dtype)
        self.new_idx = 0
        self.count = np.zeros(shape=size, dtype=np.int32)

    def __call__(self, data):
        mask = ~(np.isnan(data) | np.isinf(data))
        # 更时间窗口超过window
        for idx, val in enumerate(data):
            if mask[idx]:
                while self.pos_queue[idx] and (self.queue[idx][self.pos_queue[idx][-1]] >= val):
                    self.pos_queue[idx].pop()
                self.queue[idx].append(val)
                self.pos_queue[idx].append(self.count[idx])
                if self.count[idx] - self.pos_queue[idx][0] >= self.window:
                    self.pos_queue[idx].popleft()
                self._info[idx] = self.queue[idx][self.pos_queue[idx][0]]
            else:
                while self.pos_queue[idx] and (np.isinf(self.queue[idx][self.pos_queue[idx][-1]])):
                    self.pos_queue[idx].pop()
                self.queue[idx].append(np.inf)
                self.pos_queue[idx].append(self.count[idx])
                if self.count[idx] - self.pos_queue[idx][0] >= self.window:
                    self.pos_queue[idx].popleft()
                if np.isinf(self.queue[idx][self.pos_queue[idx][0]]):
                    self._info[idx] = np.nan
                else:
                    self._info[idx] = self.queue[idx][self.pos_queue[idx][0]]
            self.count[idx] += 1
        return self.info
    
    def update_base(self, data):
        mask = np.isnan(data)
        self.data_queue[self.new_idx][mask] = - np.inf
        self.data_queue[self.new_idx][~mask] = data[~mask]
        self.new_idx += 1
        if self.new_idx > self.window - 1:
            self.new_idx = 0
        return self
    
    def update_info(self):
        self._info = np.max(self.data_queue,axis=0)
        return self.info
    
    @property
    def info(self):
        return self._info

    def adjust(self, new_size, old_index, new_index):
        self.size = new_size
        idx_list = range(len(self.queue))
        queue = [[] for _ in idx_list]
        pos_queue = [deque() for _ in idx_list]
        count = np.zeros_like(self.count)
        info = np.zeros(shape=new_size, dtype=self._info.dtype)
        for idx in idx_list:
            if len(self.queue) > self.window:
                _pos_diff = len(self.queue[idx]) - self.window
                queue[idx] = self.queue[idx][-self.window:]
                for _,j in enumerate(self.pos_queue[idx]):
                    pos_queue[idx].append(j - _pos_diff)
                count[idx] = self.window
            else:
                queue[idx] = self.queue[idx]
                pos_queue[idx] = self.pos_queue[idx]
                count[idx] = self.count[idx]
        self.queue = [[] for _ in range(new_size)]
        self.pos_queue = [deque() for _ in range(new_size)]
        self.count = np.zeros(shape=new_size, dtype=np.int32)
        for i, j in zip(new_index, old_index):
            self.queue[i] = queue[j]
            self.pos_queue[i] = pos_queue[j]
            self.count[i] = count[j]
        info[new_index] = self._info[old_index]
        self._info = info


# ts_max - ts_min
class ts_range:
    def __init__(self, window, size, dtype):
        self.window = window
        self.size = size
        self.max_queue = np.zeros(shape=(window, size), dtype=dtype)
        self.min_queue = np.zeros(shape=(window, size), dtype=dtype)
        self.count = 0 
        self.new_idx = 0  # 当前更新到的某个窗口
        self._info = np.zeros(shape=size, dtype=dtype)
        self.ts_max = ts_max(window=window, size=size, dtype=dtype)
        self.ts_min = ts_min(window=window, size=size, dtype=dtype)
    
    def __call__(self, data):
        self._info = self.ts_max(data) - self.ts_min(data)
        return self.info

    def base_update(self, data):
        mask = np.isnan(data)
        self.max_queue[self.new_idx][~mask] = data[~mask]
        self.max_queue[self.new_idx][mask] = - np.inf
        self.min_queue[self.new_idx][~mask] = data[~mask]
        self.min_queue[self.new_idx][mask] = np.inf
        self.new_idx += 1
        if self.new_idx > self.window - 1:
            self.new_idx = 0
        return self
    
    def update_info(self):
        self._info = np.max(self.max_queue, axis=0) - np.min(self.min_queue,axis=0)
        return self.info

    @property
    def info(self):
        return self._info

    def adjust(self, new_size, old_index, new_index):
        self.size = new_size
        self.ts_max.adjust(new_size, old_index, new_index)
        self.ts_min.adjust(new_size, old_index, new_index)

# 0.01s / iteration
class ts_argmax:
    def __init__(self, window, size, dtype):
        self.window = window
        self.size = size
        self.dtype = dtype
        self.rigid_queue = []
        self.data_queue = [[] for _ in range(size)]
        self.ordered_queue = [deque() for _ in range(size)]
        self.count = np.zeros(shape=size, dtype=np.int32)
        self._info = np.zeros(shape=size, dtype=np.float32)
        self.new_idx = 0
    
    def __call__(self, data):
        mask = ~ ((np.isnan(data)) | (np.isinf(data)))
        for idx, val in enumerate(data):
            if mask[idx]:
                while (self.ordered_queue[idx]) and (self.data_queue[idx][self.ordered_queue[idx][-1]] <= val):
                    self.ordered_queue[idx].pop()
                self.data_queue[idx].append(val)
                self.ordered_queue[idx].append(self.count[idx]+0)
                if self.count[idx] - self.ordered_queue[idx][0] >= self.window:
                    self.ordered_queue[idx].popleft()
                if self.count[idx] >= self.window - 1:
                    self._info[idx] = self.ordered_queue[idx][0] - self.count[idx] + self.window - 1
                else:
                    self._info[idx] = self.ordered_queue[idx][0]
            else:
                while (self.ordered_queue[idx]) and (np.isinf(self.data_queue[idx][self.ordered_queue[idx][-1]])):
                    self.ordered_queue[idx].pop()
                self.data_queue[idx].append(-np.inf)
                self.ordered_queue[idx].append(self.count[idx])
                if self.count[idx] - self.ordered_queue[idx][0] >= self.window:
                    self.ordered_queue[idx].popleft()
                if self.count[idx] >= self.window - 1:
                    if np.isinf(self.data_queue[idx][self.ordered_queue[idx][0]]):
                        self._info[idx] = np.nan
                    else:
                        self._info[idx] = self.ordered_queue[idx][0] - self.count[idx] + self.window - 1
                else:
                    if np.isinf(self.data_queue[idx][self.ordered_queue[idx][0]]):
                        self._info[idx] = np.nan
                    else:
                        self._info[idx] = self.ordered_queue[idx][0]
            self.count[idx] += 1
        return self.info
    
    def update_base(self, data):
        self.rigid_queue.append(data)
        if self.count > self.window:
            self.rigid_queue.pop(0)
        else:
            self.count += 1
        return self

    def update_info(self):
        self._info = np.argmax(self.rigid_queue, axis=0)
        return self.info

    @property
    def info(self):
        return self._info

    def adjust(self, new_size, old_index, new_index):
        self.size = new_size
        idx_list = range(len(self.data_queue))
        data_queue = [[] for _ in idx_list]
        ordered_queue = [deque() for _ in idx_list]
        count = np.zeros_like(self.count)
        info = np.zeros(shape=new_size, dtype=self._info.dtype)
        for idx in idx_list:
            if len(self.data_queue[idx]) > self.window:
                _pos_diff = len(self.data_queue[idx]) - self.window
                data_queue[idx] = self.data_queue[idx][-self.window:]
                for _,j in enumerate(self.ordered_queue[idx]):
                    ordered_queue[idx].append(j - _pos_diff)
                count[idx] = self.window
            else:
                data_queue[idx] = self.data_queue[idx]
                ordered_queue[idx] = self.ordered_queue[idx]
                count[idx] = self.count[idx]
        self.data_queue = [[] for _ in range(new_size)]
        self.ordered_queue = [deque() for _ in range(new_size)]
        self.count = np.zeros(shape=new_size, dtype=np.int32)
        for i, j  in zip(new_index, old_index):
            self.data_queue[i] = data_queue[j]
            self.ordered_queue[i] = ordered_queue[j]
            self.count[i] = count[j]
        info[new_index] = self._info[old_index]
        self._info = info


class ts_argmin:
    def __init__(self, window, size, dtype):
        self.window = window
        self.size = size
        self.dtype = dtype
        self.rigid_queue = []
        self.data_queue = [[] for _ in range(size)]
        self.ordered_queue = [deque() for _ in range(size)]
        self.count = np.zeros(shape=size, dtype=np.int32)
        self._info = np.zeros(shape=size, dtype=np.float32)
        self.new_idx = 0
    
    def __call__(self, data):
        mask = ~ ((np.isnan(data)) | (np.isinf(data)))
        for idx, val in enumerate(data):
            if mask[idx]:
                while (self.ordered_queue[idx]) and (self.data_queue[idx][self.ordered_queue[idx][-1]] >= val):
                    self.ordered_queue[idx].pop()
                self.data_queue[idx].append(val)
                self.ordered_queue[idx].append(self.count[idx]+0)
                if self.count[idx] - self.ordered_queue[idx][0] >= self.window:
                    self.ordered_queue[idx].popleft()
                if self.count[idx] >= self.window - 1:
                    self._info[idx] = self.ordered_queue[idx][0] - self.count[idx] + self.window - 1
                else:
                    self._info[idx] = self.ordered_queue[idx][0]
            else:
                while (self.ordered_queue[idx]) and (np.isinf(self.data_queue[idx][self.ordered_queue[idx][-1]])):
                    self.ordered_queue[idx].pop()
                self.data_queue[idx].append(np.inf)
                self.ordered_queue[idx].append(self.count[idx])
                if self.count[idx] - self.ordered_queue[idx][0] >= self.window:
                    self.ordered_queue[idx].popleft()
                if self.count[idx] >= self.window - 1:
                    if np.isinf(self.data_queue[idx][self.ordered_queue[idx][0]]):
                        self._info[idx] = np.nan
                    else:
                        self._info[idx] = self.ordered_queue[idx][0] - self.count[idx] + self.window - 1
                else:
                    if np.isinf(self.data_queue[idx][self.ordered_queue[idx][0]]):
                        self._info[idx] = np.nan
                    else:
                        self._info[idx] = self.ordered_queue[idx][0]
            self.count[idx] += 1
        return self.info
    
    def update_base(self, data):
        self.rigid_queue.append(data)
        if self.count > self.window:
            self.rigid_queue.pop(0)
        else:
            self.count += 1
        return self

    def update_info(self):
        self._info = np.argmin(self.rigid_queue, axis=0)
        return self.info

    @property
    def info(self):
        return self._info
    
    def adjust(self, new_size, old_index, new_index):
        self.size = new_size
        idx_list = range(len(self.data_queue))
        data_queue = [[] for _ in idx_list]
        ordered_queue = [deque() for _ in idx_list]
        count = np.zeros_like(self.count)
        info = np.zeros(shape=new_size, dtype=self._info.dtype)
        for idx in idx_list:
            if len(self.data_queue[idx]) > self.window:
                _pos_diff = len(self.data_queue[idx]) - self.window
                data_queue[idx] = self.data_queue[idx][-self.window:]
                for _,j in enumerate(self.ordered_queue[idx]):
                    ordered_queue[idx].append(j - _pos_diff)
                count[idx] = self.window
            else:
                data_queue[idx] = self.data_queue[idx]
                ordered_queue[idx] = self.ordered_queue[idx]
                count[idx] = self.count[idx]
        self.data_queue = [[] for _ in range(new_size)]
        self.ordered_queue = [deque() for _ in range(new_size)]
        self.count = np.zeros(shape=new_size, dtype=np.int32)
        for i, j  in zip(new_index, old_index):
            self.data_queue[i] = data_queue[j]
            self.ordered_queue[i] = ordered_queue[j]
            self.count[i] = count[j]
        info[new_index] = self._info[old_index]
        self._info = info


class ts_pre_action:
    def __init__(self, window, size, dtype):
        self.window = window
        self.size = size
        self.argmax1 = ts_argmax(window=window, size=size, dtype=dtype)
        self.argmax2 = ts_argmax(window=window, size=size, dtype=dtype)
        self._info = None
    
    def __call__(self, x, y):
        self._info = self.argmax1(x) - self.argmax2(y)
        return self.info
    
    def update_base(self, x, y):
        self.argmax1.update_base(x)
        self.argmax2.update_base(y)
        return self
    
    def update_info(self):
        self._info = self.argmax1.update_info() - self.argmax2.update_info()
        return self.info

    @property
    def info(self):
        return self._info

    def adjust(self, new_size, old_index, new_index):
        self.argmax1.adjust(new_size, old_index, new_index)
        self.argmax2.adjust(new_size, old_index, new_index)

# 滑动分位数 这两个时间复杂度fucking 股票数 * 2log(n)
class ts_quantile:
    def __init__(self, window, q, size, dtype):
        """
        window: 回溯窗口长度
        size: 合约数量
        dtype: 中间状态类型,近似为返回结果类似
        """
        self.window = window
        self.odd = True if (window % 2 == 1) else False
        self.q = q
        # like numpy method='midpoint'
        self.pos = int(self.q * (self.window - 1))
        self.size = size
        self.data_queue = np.zeros(shape=(window, size), dtype=dtype)
        self.queue = [[] for _ in range(size)]
        self.ordered_queue = [[] for _ in range(size)]
        self._info = np.zeros(shape=size, dtype=dtype)
        self.new_idx = 0
        self.count = np.zeros(shape=size, dtype=np.int32)

    def __call__(self, data):
        mask = (np.isnan(data) | np.isinf(data))
        for idx, val in enumerate(data):
            if self.count[idx] >= self.window:
                if ~mask[idx]:
                    self.ordered_queue[idx].insert(bisect.bisect_right(self.ordered_queue[idx], val, lo=0), val)
                    self.queue[idx].append(val)
                    if len(self.queue[idx]) > self.window:
                        self.ordered_queue[idx].pop(bisect.bisect_right(self.ordered_queue[idx], self.queue[idx].pop(0), lo=0)-1)
                        if self.odd:
                            self._info[idx] = self.ordered_queue[idx][self.pos]
                        else:
                            self._info[idx] = (self.ordered_queue[idx][self.pos] + self.ordered_queue[idx][self.pos + 1]) / 2
                    else:
                        self._info[idx] = np.nan
                else:
                    if self.queue[idx]:
                        self.ordered_queue[idx].pop(bisect.bisect_right(self.ordered_queue[idx], self.queue[idx].pop(0), lo=0)-1)
                    self._info[idx] = np.nan
            else:
                self.count[idx] += 1  
                if ~mask[idx]:
                    self.ordered_queue[idx].insert(bisect.bisect_right(self.ordered_queue[idx], val, lo=0), val)
                    self.queue[idx].append(val)
                    if self.odd:
                        self._info[idx] = self.ordered_queue[idx][self.pos]
                    else:
                        if self.pos + 1 < len(self.ordered_queue[idx]):
                            self._info[idx] = (self.ordered_queue[idx][self.pos] + self.ordered_queue[idx][self.pos + 1]) / 2
                        else:
                            self._info[idx] = np.nan
                else:
                    self._info[idx] = np.nan
        return self.info
    
    def update_base(self, data):
        self.data_queue[self.new_idx] = data
        self.new_idx += 1
        if self.new_idx > self.window - 1:
            self.new_idx = 0
        return self
    
    def update_info(self):
        self._info = np.percentile(self.data_queue, q=100*self.q, axis=0)
        return self.info
    
    @property
    def info(self):
        return self._info

    def adjust(self, new_size, old_index, new_index):
        self.size = new_size
        queue = [[] for _ in range(new_size)]
        ordered_queue = [[] for _ in range(new_size)]
        count = np.zeros(shape=new_size, dtype=np.int32)
        info = np.zeros(shape=new_size, dtype=self._info.dtype)
        for new_pos, old_pos in zip(new_index, old_index):
            queue[new_pos] = self.queue[old_pos]
            ordered_queue[new_pos] = self.ordered_queue[old_pos]
            count[new_pos] = self.count[old_pos]
            info[new_pos] = self._info[old_pos]
        self.ordered_queue = ordered_queue
        self.queue = queue
        self._info = info
        self.count = count


class ts_med:
    def __init__(self, window, size, dtype):
        """
        window: 回溯窗口长度
        size: 合约数量
        dtype: 中间状态类型,近似为返回结果类似
        """
        self.window = window
        self.odd = True if (window % 2 == 1) else False
        self.q = 0.5
        # like numpy method='midpoint'
        self.pos = int(self.q * (self.window - 1))
        self.size = size
        self.data_queue = np.zeros(shape=(window, size), dtype=dtype)
        self.queue = [[] for _ in range(size)]
        self.ordered_queue = [[] for _ in range(size)]
        self._info = np.zeros(shape=size, dtype=dtype)
        self.new_idx = 0
        self.count = np.zeros(shape=size, dtype=np.int32)

    def __call__(self, data):
        mask = (np.isnan(data) | np.isinf(data))
        for idx, val in enumerate(data):
            if self.count[idx] >= self.window:
                if ~mask[idx]:
                    self.ordered_queue[idx].insert(bisect.bisect_right(self.ordered_queue[idx], val, lo=0), val)
                    self.queue[idx].append(val)
                    if len(self.queue[idx]) > self.window:
                        self.ordered_queue[idx].pop(bisect.bisect_right(self.ordered_queue[idx], self.queue[idx].pop(0), lo=0)-1)
                        if self.odd:
                            self._info[idx] = self.ordered_queue[idx][self.pos]
                        else:
                            self._info[idx] = (self.ordered_queue[idx][self.pos] + self.ordered_queue[idx][self.pos + 1]) / 2
                    else:
                        self._info[idx] = np.nan
                else:
                    if self.queue[idx]:
                        self.ordered_queue[idx].pop(bisect.bisect_right(self.ordered_queue[idx], self.queue[idx].pop(0), lo=0)-1)
                    self._info[idx] = np.nan
            else:
                self.count[idx] += 1  
                if ~mask[idx]:
                    self.ordered_queue[idx].insert(bisect.bisect_right(self.ordered_queue[idx], val, lo=0), val)
                    self.queue[idx].append(val)
                    if self.odd:
                        self._info[idx] = self.ordered_queue[idx][self.pos]
                    else:
                        if self.pos + 1 < len(self.ordered_queue[idx]):
                            self._info[idx] = (self.ordered_queue[idx][self.pos] + self.ordered_queue[idx][self.pos + 1]) / 2
                        else:
                            self._info[idx] = np.nan
                else:
                    self._info[idx] = np.nan
        return self.info
    
    def update_base(self, data):
        self.data_queue[self.new_idx] = data
        self.new_idx += 1
        if self.new_idx > self.window - 1:
            self.new_idx = 0
        return self
    
    def update_info(self):
        self._info = np.percentile(self.data_queue, q=100*self.q, axis=0)
        return self.info
    
    @property
    def info(self):
        return self._info

    def adjust(self, new_size, old_index, new_index):
        self.size = new_size
        queue = [[] for _ in range(new_size)]
        ordered_queue = [[] for _ in range(new_size)]
        count = np.zeros(shape=new_size, dtype=np.int32)
        info = np.zeros(shape=new_size, dtype=self._info.dtype)
        for new_pos, old_pos in zip(new_index, old_index):
            queue[new_pos] = self.queue[old_pos]
            ordered_queue[new_pos] = self.ordered_queue[old_pos]
            count[new_pos] = self.count[old_pos]
            info[new_pos] = self._info[old_pos]
        self.ordered_queue = ordered_queue
        self.queue = queue
        self._info = info
        self.count = count
# ===============================================
# endregion


# region 涉及排序,在任何维度数据上的时间复杂度很高
# ===============================================
# 滑动排序
class ts_rank:
    def __init__(self, window, size, dtype, pct=False):
        """
        window: 回溯窗口长度
        size: 合约数量
        dtype: 中间状态类型,近似为返回结果类似
        """
        self.window = window
        self.size = size
        self.data_queue = np.zeros(shape=(window, size), dtype=dtype)
        self.count = 0 
        self.new_idx = 0  # 当前更新到的某个窗口
        self._info = None
        self.pct = pct

    def __call__(self, data):
        # 更时间窗口超过window
        if self.count >= self.window:
            self.data_queue[self.new_idx] = data
            if self.pct:
                self._info = (self.data_queue <= data).sum(axis=0) / self.window
            else:
                self._info = (self.data_queue <= data).sum(axis=0)
        # 更新时间窗口不足window
        else:
            self.data_queue[self.new_idx] = data
            if self.pct:
                self._info = (self.data_queue[:self.new_idx] <= data).sum(axis=0) / (self.new_idx+1)
            else:
                self._info = (self.data_queue[:self.new_idx] <= data).sum(axis=0)
            self.count += 1
        self.new_idx += 1
        if self.new_idx > self.window - 1:
            self.new_idx = 0
        return self.info
    
    def update_base(self, data):
        self.data_queue[self.new_idx] =  data
        self.new_idx += 1
        if self.new_idx > self.window - 1:
            self.new_idx = 0
        return self
    
    def update_info(self):
        if self.pct:
            self._info = (self.data_queue <= self.data_queue[self.new_idx - 1]).sum(axis=0) / self.window
        else:
            self._info = (self.data_queue <= self.data_queue[self.new_idx - 1]).sum(axis=0)
        return self.info
    
    @property
    def info(self):
        return self._info
    
    def adjust(self, new_size, old_index, new_index):
        self.size = new_index
        data_queue = np.zeros(shape=(self.window, new_size), dtype=self.data_queue.dtype)
        for i, j in zip(new_index, old_index):
            data_queue[:,i] = self.data_queue[:,j]
        self.data_queue = data_queue

# ===============================================
# endregion

# region 切割算符,又排序又聚合,时间复杂度最高
# ===============================================
# ts_var_top-1d
class _ts_var_top:
    def __init__(self, window:int, agg_num:int) -> None:
        """聚合头部或尾部数据
        window:int,回溯window个窗口
        agg_num:int, 聚合window个窗口内,头部agg_num个窗口对应位置的值
        """
        self.window = window
        self.agg_num = agg_num
        self.cut = self.window - self.agg_num
        # 记录值顺序
        self.val_queue1 = []
        self.val_queue2 = []
        # 记录聚合值
        self.agg_queue = []
        # 记录位置
        self.pos_queue = []
        # self.ordered_queue = []
        self.count = 0
        self._info = np.nan
        self.sum = 0
        self.sum2 = 0
        
    def update(self, data2, data1):
        if np.isnan(data1) | np.isinf(data1) | np.isnan(data2) | np.isinf(data2):
            return self.info
        if self.count >= self.window:
            # 需要丢掉的数值
            pop_val1 = self.val_queue1.pop(0)
            pop_val2 = self.val_queue2.pop(0)
            # 需要丢掉的val1在有序列表中的位置
            pop_pos = bisect.bisect_left(self.pos_queue, pop_val1)
            # 当需要剔除第0位,由于二分插入的计算问题,会返回最后一位+1,因此这里改为0
            if pop_pos == self.window:
                pop_pos = 0
            # 在有序列表中剔除pop_val1
            self.pos_queue.pop(pop_pos)
            # 按pop_val1在有序列表中的位置从agg_queue中剔除pop_val2
            self.agg_queue.pop(pop_pos)
            # 将新的值添加进时间顺序队列
            self.val_queue1.append(data1)
            self.val_queue2.append(data2)
            # 寻找data1在有序队列中的插入位置
            insert_pos = bisect.bisect_right(self.pos_queue, data1)
            self.pos_queue.insert(insert_pos, data1)
            self.agg_queue.insert(insert_pos, data2)
            # 判断是否需要将data2与pop_vol2加入求和
            # 判断进入与退出两个元素的位置
            # 剔除的val1处于头部
            if pop_pos >= self.cut:
                # 新入的data1也处于头部->两者相减
                if insert_pos >= self.cut:
                    self.sum += (data2 - pop_val2)
                    self.sum2 += (data2 - pop_val2) * (data2 + pop_val2)
                # 新入的data1不在头部->减去pop_val2加上self.agg_queue[self.cut]
                else:
                    self.sum += (self.agg_queue[self.cut] - pop_val2)
                    self.sum2 += (self.agg_queue[self.cut] - pop_val2) * (self.agg_queue[self.cut] + pop_val2)
            # 剔除的val1不在头部
            else:
                # data1在头部->加上data2减去被挤出去的self.cut-1位置的值
                if insert_pos >= self.cut:
                    self.sum += (data2 - self.agg_queue[self.cut - 1])
                    self.sum2 += (data2 - self.agg_queue[self.cut - 1]) * (data2 + self.agg_queue[self.cut - 1])
                # 若新加入的data1也不在头部,不做调整
            self._info = np.fabs(self.sum2 / self.agg_num - (self.sum / self.agg_num) ** 2)
        else:
            self.count += 1
            # 将data1、data2按时间顺序放入容器
            self.val_queue1.append(data1)
            self.val_queue2.append(data2)
            pos = bisect.bisect_right(self.pos_queue, data1)
            # 将data1按大小顺序放入pos_queue容器,data2按data1的位置放入agg_queue容器
            self.pos_queue.insert(pos, data1)
            self.agg_queue.insert(pos, data2)
            if self.count >= self.window:
                self._info = np.var(self.agg_queue[-self.agg_num:])
                self.sum += np.sum(self.agg_queue[-self.agg_num:])
                self.sum2 += np.sum([x**2 for x in self.agg_queue[-self.agg_num:]])
        return self.info

    @property
    def info(self):
        return self._info

# 0.084s/row  批量计算时 0.32s/row  在计算小批量时属于更加高效的存在
class ts_var_top:
    def __init__(self, window, agg_num, size, dtype):
        self.x_queue = np.zeros(shape=(window,size),dtype=dtype)
        self.y_queue = np.zeros(shape=(window,size),dtype=dtype)
        self.ts_obj = [_ts_var_top(window=window, agg_num=agg_num) for _ in range(size)]
        self.window = window
        self.agg_num = agg_num
        self.size = size
        self._info = np.full(shape=size, dtype=dtype, fill_value=np.nan)
        self.new_idx = 0
        self.count = 0
    
    def __call__(self, x, y):
        for idx, val in enumerate(x):
            self._info[idx] = self.ts_obj[idx].update(val, y[idx])
        return self.info
    
    def update_base(self, x, y):
        self.x_queue[self.new_idx] = x
        self.y_queue[self.new_idx] = y
        self.new_idx += 1
        if self.new_idx > self.window - 1:
            self.new_idx = 0
        return self
    
    def update_info(self):
        if self.count >= self.window:
            self._info = self.x_queue.T[np.where(np.argsort(np.argsort(self.y_queue,axis=0),axis=0).T>=self.window-self.agg_num)].reshape(self.size,self.agg_num).T.var(axis=0)
        else:
            self.count += 1
        return self.info

    @property
    def info(self):
        return self._info
    

class ts_std_top:
    def __init__(self, window, agg_num, size, dtype):
        self.x_queue = np.zeros(shape=(window,size),dtype=dtype)
        self.y_queue = np.zeros(shape=(window,size),dtype=dtype)
        self.ts_obj = [_ts_var_top(window=window, agg_num=agg_num) for _ in range(size)]
        self.window = window
        self.agg_num = agg_num
        self.size = size
        self._info = np.full(shape=size, dtype=dtype, fill_value=np.nan)
        self.new_idx = 0
        self.count = 0
    
    def __call__(self, x, y):
        for idx, val in enumerate(x):
            self._info[idx] = np.sqrt(self.ts_obj[idx].update(val, y[idx]))
        return self.info
    
    def update_base(self, x, y):
        self.x_queue[self.new_idx] = x
        self.y_queue[self.new_idx] = y
        self.new_idx += 1
        if self.new_idx > self.window - 1:
            self.new_idx = 0
        return self
    
    def update_info(self):
        if self.count >= self.window:
            self._info = self.x_queue.T[np.where(np.argsort(np.argsort(self.y_queue,axis=0),axis=0).T>=self.window-self.agg_num)].reshape(self.size,self.agg_num).T.std(axis=0)
        else:
            self.count += 1
        return self.info

    @property
    def info(self):
        return self._info

# ts_sum_top-1d
class _ts_sum_top:
    def __init__(self, window:int, agg_num:int) -> None:
        """聚合头部或尾部数据
        window:int,回溯window个窗口
        agg_num:int, 聚合window个窗口内,头部agg_num个窗口对应位置的值
        """
        self.window = window
        self.agg_num = agg_num
        self.cut = self.window - self.agg_num
        # 记录值顺序
        self.val_queue1 = []
        self.val_queue2 = []
        # 记录聚合值
        self.agg_queue = []
        # 记录位置
        self.pos_queue = []
        # self.ordered_queue = []
        self.count = 0
        self._info = 0
        
    def update(self, data2, data1):
        if np.isnan(data1) | np.isinf(data1) | np.isnan(data2) | np.isinf(data2):
            return self.info
        
        if self.count >= self.window:
            # 需要丢掉的数值
            pop_val1 = self.val_queue1.pop(0)
            pop_val2 = self.val_queue2.pop(0)
            # 需要丢掉的val1在有序列表中的位置
            pop_pos = bisect.bisect_left(self.pos_queue, pop_val1)
            # 当需要剔除第0位,由于二分插入的计算问题,会返回最后一位+1,因此这里改为0
            if pop_pos == self.window:
                pop_pos = 0
            # 在有序列表中剔除pop_val1
            self.pos_queue.pop(pop_pos)
            # 按pop_val1在有序列表中的位置从agg_queue中剔除pop_val2
            self.agg_queue.pop(pop_pos)
            # 将新的值添加进时间顺序队列
            self.val_queue1.append(data1)
            self.val_queue2.append(data2)
            # 寻找data1在有序队列中的插入位置
            insert_pos = bisect.bisect_right(self.pos_queue, data1)
            self.pos_queue.insert(insert_pos, data1)
            self.agg_queue.insert(insert_pos, data2)
            # 判断是否需要将data2与pop_vol2加入求和
            # 判断进入与退出两个元素的位置
            # 剔除的val1处于头部
            if pop_pos >= self.cut:
                # 新入的data1也处于头部->两者相减
                if insert_pos >= self.cut:
                    self._info += (data2 - pop_val2)
                # 新入的data1不在头部->减去pop_val2加上self.agg_queue[self.cut]
                else:
                    self._info += (self.agg_queue[self.cut] - pop_val2)
            # 剔除的val1不在头部
            else:
                # data1在头部->加上data2减去被挤出去的self.cut-1位置的值
                if insert_pos >= self.cut:
                    self._info += (data2 - self.agg_queue[self.cut - 1])
                # 若新加入的data1也不在头部,不做调整
        else:
            self.count += 1
            # 将data1、data2按时间顺序放入容器
            self.val_queue1.append(data1)
            self.val_queue2.append(data2)
            pos = bisect.bisect_right(self.pos_queue, data1)
            # 将data1按大小顺序放入pos_queue容器,data2按data1的位置放入agg_queue容器
            self.pos_queue.insert(pos, data1)
            self.agg_queue.insert(pos, data2)
            if self.count >= self.window:
                self._info = np.sum(self.agg_queue[-self.agg_num:])
        return self.info

    @property
    def info(self):
        return self._info 


class ts_sum_top:
    def __init__(self, window, agg_num, size, dtype):
        self.x_queue = np.zeros(shape=(window,size),dtype=dtype)
        self.y_queue = np.zeros(shape=(window,size),dtype=dtype)
        self.ts_obj = [_ts_sum_top(window=window, agg_num=agg_num) for _ in range(size)]
        self.window = window
        self.agg_num = agg_num
        self.size = size
        self._info = np.full(shape=size, dtype=dtype, fill_value=np.nan)
        self.new_idx = 0
        self.count = 0
    
    def __call__(self, x, y):
        for idx, val in enumerate(x):
            self._info[idx] = self.ts_obj[idx].update(val, y[idx])
        return self.info
    
    def update_base(self, x, y):
        self.x_queue[self.new_idx] = x
        self.y_queue[self.new_idx] = y
        self.new_idx += 1
        if self.new_idx > self.window - 1:
            self.new_idx = 0
        return self
    
    def update_info(self):
        if self.count >= self.window:
            self._info = self.x_queue.T[np.where(np.argsort(np.argsort(self.y_queue,axis=0),axis=0).T>=self.window-self.agg_num)].reshape(self.size,self.agg_num).T.sum(axis=0)
        else:
            self.count += 1
        return self.info

    @property
    def info(self):
        return self._info
# ===============================================
# endregin 