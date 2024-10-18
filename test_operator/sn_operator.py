import bisect
import numpy as np
import pandas as pd
from sn_platform.config import logger
from collections import Counter, defaultdict, deque
from functools import reduce


def div_cs(a, b): return ((a - b) / (a + b)) if (a + b != 0) else 0
def div_cs_abs(a, b): return abs((a - b) / (a + b)) if not (a + b == 0) else 0
def div(a, b): return (a / b) if (b != 0) else 0
def add(*args): return reduce(lambda a,b: a + b, args)
def sub(a, b): return a - b
def mul(a, b): return a * b
def Abs(a): return abs(a)
def if_big(a,b,c,d): return c if (a > b) else d
def if_small(a,b,c,d): return c if (b > a) else d
def if_equal(a,b,c,d): return c if (a == b) else d
def growth(a,b): return (a / b - 1) if (b != 0) else 0

# ts_sum
class TsSum:
    def __init__(self, window):
        self.window = window
        self.full_queue = False
        self.queue = []
        self.count = 0
        self._sum = 0
        
    def update(self, data):
        if self.full_queue:
            self._sum += (data - self.queue.pop(0))
            self.queue.append(data)
        else:
            self._sum += data
            self.count += 1
            self.queue.append(data)
            if self.count >= self.window:
                self.full_queue = True
        return self.info
    
    @property
    def info(self):
        return self._sum

# ts_avg
class TsAvg:
    def __init__(self, window):
        self.window = window
        self.full_queue = False
        self.queue = []
        self.count = 0
        self._sum = 0
        
    def update(self, data):
        if self.full_queue:
            self._sum += (data - self.queue.pop(0))
            self.queue.append(data)
        else:
            self._sum += data
            self.count += 1
            self.queue.append(data)
            if self.count >= self.window:
                self.full_queue = True
        return self.info
    
    @property
    def info(self):
        return self._sum / self.window
    
# ts_sum_div
class TsSumDiv:
    def __init__(self, window):
        self.window = window
        self.full_queue = False
        self.queue1 = []
        self.queue2 = []
        self.count = 0
        self._sum1 = 0
        self._sum2 = 0
        
    def update(self, data1, data2):
        if self.full_queue:
            self._sum1 += (data1 - self.queue1.pop(0))
            self._sum2 += (data2 - self.queue2.pop(0))
            self.queue1.append(data1)
            self.queue2.append(data2)
        else:
            self._sum1 += data1
            self._sum2 += data2
            self.count += 1
            self.queue1.append(data1)
            self.queue2.append(data2)
            if self.count >= self.window:
                self.full_queue = True
        return self.info
    @property
    def info(self):
        return (self._sum1 / self._sum2) if (self._sum2 != 0) else 0

# ts_sum_div_cs
class TsSumDivCs:
    def __init__(self, window):
        self.window = window
        self.full_queue = False
        self.queue1 = []
        self.queue2 = []
        self.count = 0
        self._sum1 = 0
        self._sum2 = 0
        
    def update(self, data1, data2):
        if self.full_queue:
            self._sum1 += (data1 - self.queue1.pop(0))
            self._sum2 += (data2 - self.queue2.pop(0))
            self.queue1.append(data1)
            self.queue2.append(data2)
        else:
            self._sum1 += data1
            self._sum2 += data2
            self.count += 1
            self.queue1.append(data1)
            self.queue2.append(data2)
            if self.count >= self.window:
                self.full_queue = True
        return self.info
    
    @property
    def info(self):
        return ((self._sum1 - self._sum2) / (self._sum1 + self._sum2)) if (self._sum1 + self._sum2 != 0) else 0

# ts_sum_div_cs_abs
class TsSumDivCsAbs:
    def __init__(self, window):
        self.window = window
        self.full_queue = False
        self.queue1 = []
        self.queue2 = []
        self.count = 0
        self._sum1 = 0
        self._sum2 = 0
        
    def update(self, data1, data2):
        if self.full_queue:
            self._sum1 += (data1 - self.queue1.pop(0))
            self._sum2 += (data2 - self.queue2.pop(0))
            self.queue1.append(data1)
            self.queue2.append(data2)
        else:
            self._sum1 += data1
            self._sum2 += data2
            self.count += 1
            self.queue1.append(data1)
            self.queue2.append(data2)
            if self.count >= self.window:
                self.full_queue = True
        return self.info
    
    @property
    def info(self):
        return abs(((self._sum1 - self._sum2) / (self._sum1 + self._sum2))) if (self._sum1 + self._sum2 != 0) else 0

# ts_growth
class TsGrowth:
    def __init__(self, window):
        self.window = window + 1
        self.full_queue = False
        self.queue = []
        self.count = 0
        self._growth = 0
        
    def update(self, data):
        if self.full_queue:
            self.queue.pop(0)
            self.queue.append(data)
            self._growth = (100 * (data / self.queue[0] - 1)) if self.queue[0] != 0 else 0
        else:
            self.queue.append(data)
            self.count += 1
            if self.count >= self.window:
                self.full_queue = True
                self._growth = (100 * (data / self.queue[0] - 1)) if self.queue[0] != 0 else 0
        return self.info
    @property
    def info(self):
        return self._growth

# ts_growth_abs
class TsGrowthAbs:
    def __init__(self, window):
        self.window = window + 1
        self.full_queue = False
        self.queue = []
        self.count = 0
        self._growth = 0
        
    def update(self, data):
        if self.full_queue:
            self.queue.pop(0)
            self.queue.append(data)
            self._growth = abs(100 * (data / self.queue[0] - 1)) if self.queue[0] != 0 else 0
        else:
            self.queue.append(data)
            self.count += 1
            if self.count >= self.window:
                self.full_queue = True
                self._growth = abs(100 * (data / self.queue[0] - 1)) if self.queue[0] != 0 else 0
        return self.info
    
    @property
    def info(self):
        return abs(self._growth)
  
# ts_delta
class TsDelta:
    def __init__(self, window):
        self.window = window + 1
        self.full_queue = False
        self.queue = []
        self.count = 0
        self._delta = 0
        
    def update(self, data):
        if self.full_queue:
            self.queue.pop(0)
            self.queue.append(data)
            self._delta = data - self.queue[0]
        else:
            self.queue.append(data)
            self._delta = data - self.queue[0]
            self.count += 1
            if self.count >= self.window:
                self.full_queue = True
        return self.info
    @property
    def info(self):
        return self._delta
    
# ts_delay 
class TsDelay:
    
    def __init__(self, window):
        self.window = window + 1  
        self.full_queue = False
        self.queue = []
        self.count = 0
        self._delay = 0
        
    def update(self, data):
        if self.full_queue:
            self.queue.pop(0)
            self.queue.append(data)
            self._delay = self.queue[0]
        else:
            self.queue.append(data)
            self._delay = self.queue[0]
            self.count += 1
            if self.count >= self.window:
                self.full_queue = True
        return self.info
    @property
    def info(self):
        return self._delay
    
# ts_std
class TsStd:
    _var = 0
    _std = 0
    def __init__(self, window):
        self.window = window
        self.queue = []
        self.count = 0
        self.full_queue = False
        self.window_sum = 0
        self.window_sum_squared = 0
    
    def update(self, data):
        if self.full_queue:
            del_data = self.queue.pop(0)
            self.queue.append(data)
            self.window_sum += (data - del_data)
            self.window_sum_squared += (data ** 2 - del_data ** 2)
            self._var = (self.window_sum_squared - (self.window_sum ** 2) / self.window) / self.window
            if self._var >= 0:
                self._std = np.sqrt(self._var)
            else:
                self._std = 0
        else:
            self.count += 1
            self.queue.append(data)
            if self.count >= self.window:
                self.window_sum = np.sum(self.queue)
                self.window_sum_squared = np.sum([x**2 for x in self.queue])
                self.full_queue = True
                self._std = np.std(self.queue)
        return self.info
    
    @property
    def info(self):
        return self._std

# ts_avg / ts_std
class TsTCV:
    def __init__(self, n):
        self._ts_std = TsStd(n)
        self._ts_avg = TsAvg(n)

    def update(self, data):
        self._ts_avg.update(data)
        self._ts_std.update(data)
        return self.info
    
    @property
    def info(self):
        return (self._ts_avg.info / self._ts_std.info) if self._ts_std.info > 0 else 0

# ts_std / ts_avg
class TsCV:
    def __init__(self, n):
        self._ts_std = TsStd(n)
        self._ts_avg = TsAvg(n)
        self._info = 0

    def update(self, data):
        self._ts_avg.update(data)
        self._ts_std.update(data)
        return self.info
    
    @property
    def info(self):
        return (self._ts_std.info / self._ts_avg.info) if self._ts_avg.info != 0 else 0
    
# ts_std / (abs(ts_avg) + 1)
class TsCVAbs:
    def __init__(self, n):
        self._ts_std = TsStd(n)
        self._ts_avg = TsAvg(n)
        self._info = 0

    def update(self, data):
        self._ts_avg.update(data)
        self._ts_std.update(data)
        return self.info
    
    @property
    def info(self):
        return self._ts_std.info / (abs(self._ts_avg.info) + 1)
      
# ts_cov
# class TsCov:
#     def __init__(self, window):
#         self.n = window
#         self.count = 0
#         self.full_queue = False
#         self.x_y = 0
#         self.x_queue = []
#         self.y_queue = []
#         self.x_sum = 0
#         self.y_sum = 0
#         self._cov = 0
    
#     def update(self, x, y):
#         if self.full_queue:
#             removed_x, removed_y = self.x_queue.pop(0), self.y_queue.pop(0)
#             self.x_y += (x * y - removed_x * removed_y)
#             new_x_sum = self.x_sum + x - removed_x
#             new_y_sum = self.y_sum + y - removed_y
#             if self.x_sum != 0:
#                 self.x_hat_y = (self.x_hat_y - removed_y * (self.x_sum / self.n)) * ((self.x_sum + x - removed_x) / self.x_sum) + y * new_x_sum / self.n
#             else:
#                 self.x_hat_y = 0
#             if self.y_sum != 0:
#                 self.y_hat_x = (self.y_hat_x - removed_x * (self.y_sum / self.n)) * ((self.y_sum + y - removed_y) / self.y_sum) + x * new_y_sum / self.n
#             else:
#                 self.y_hat_x = 0
#             self.x_hat_y_hat = new_x_sum * new_y_sum / self.n
#             self.x_sum = new_x_sum
#             self.y_sum = new_y_sum
#             self._cov = (self.x_y - self.x_hat_y - self.y_hat_x + self.x_hat_y_hat) / self.n
#             self.x_queue.append(x)
#             self.y_queue.append(y)
#         else:
#             self.count += 1
#             self.x_queue.append(x)
#             self.y_queue.append(y)
#             self.x_sum += x
#             self.y_sum += y
#             if self.count == self.n:
#                 self.full_queue = True
#                 self.x_y = np.sum([self.x_queue[_] * self.y_queue[_]  for _ in range(len(self.x_queue))])
#                 self.x_hat_y = (self.x_sum / self.n) * np.sum(self.y_queue)
#                 self.y_hat_x = (self.y_sum / self.n) * np.sum(self.x_queue)
#                 self.x_hat_y_hat = self.x_sum * self.y_sum / self.n
#                 self._cov = (self.x_y - self.x_hat_y - self.y_hat_x + self.x_hat_y_hat) / self.n
#         return self.info
                
#     @property
#     def info(self):
#         return self._cov
class TsCov:
    def __init__(self, window):
        self.widnow = window
        self.x_queue = []
        self.y_queue = []
        self.x_avg = 0
        self.y_avg = 0
        self.xy_avg = 0
        self.count = 0
        self.full_queue = False
        self._cov = 0
        
    def update(self, x, y):
        if self.full_queue:
            removed_x, removed_y = self.x_queue.pop(0), self.y_queue.pop(0)
            self.x_queue.append(x)
            self.y_queue.append(y)
            self.x_avg += (x - removed_x) / self.widnow
            self.y_avg += (y - removed_y) / self.widnow
            self.xy_avg += (x * y - removed_x * removed_y) / self.widnow
        else:
            self.count += 1
            if self.count >= self.widnow:
                self.full_queue = True
            self.x_queue.append(x)
            self.y_queue.append(y)
            self.x_avg = (x + self.x_avg * (self.count - 1)) / self.count
            self.y_avg = (y + self.y_avg * (self.count - 1)) / self.count
            self.xy_avg = (x * y + self.xy_avg * (self.count - 1)) / self.count
        self._cov = (self.xy_avg) - (self.x_avg * self.y_avg)
        return self.info
    
    @property
    def info(self):
        return self._cov 

# ts_corr
class TsCorr:
    def __init__(self, window):
        self.n = window
        self.full_queue = False
        self.ts_cov = TsCov(window)
        self.ts_x_std = TsStd(window)
        self.ts_y_std = TsStd(window)
        self._corr = 0
    
    def update(self, x, y):
        self.ts_cov.update(x,y)
        self.ts_x_std.update(x)
        self.ts_y_std.update(y)
        if self.ts_cov.full_queue:
            if (self.ts_x_std.info * self.ts_y_std.info) == 0:
                self._corr = 0
            else:
                # print('problem_tscorr')
                self._corr = self.ts_cov.info / (self.ts_x_std.info * self.ts_y_std.info)
        return self.info
    
    @property
    def info(self):
        return self._corr     

# ts_quantile
class TsQuantile:
    def __init__(self, window, q):
        self.q = q
        self.window = window
        self.queue = []
        self.ordered_queue = []
        self.full_queue = False
        self.count = 0
        if window % 2 == 0:
            self.odd = False
        else:
            self.odd = True
    
    def update(self, data):
        if self.full_queue:
            self.ordered_queue.pop(bisect.bisect_right(self.ordered_queue, self.queue.pop(0), lo=0) - 1)
            self.ordered_queue.insert(bisect.bisect(self.ordered_queue, data, lo=0), data)
            self.queue.append(data)
        else:
            self.count += 1
            self.queue.append(data)
            self.ordered_queue.insert(bisect.bisect_right(self.ordered_queue, data, lo=0), data)
            if self.count >= self.window:
                self.full_queue = True
        return self.info
    
    @property
    def info(self):
        if self.full_queue:
            idx = int(self.q * (self.window - 1))
            if self.odd:
                return self.ordered_queue[idx]
            else:
                return (self.ordered_queue[idx] + self.ordered_queue[idx + 1]) / 2
        else:
            return 0

# ts_max
class TsMax:
    def __init__(self, window):
        self.window = window
        self.queue = []
        self.pos_queue = deque()
        self.count = 0
        
    def update(self, data):
        # 入
        while self.pos_queue and (self.queue[self.pos_queue[-1]]) <= data:
            self.pos_queue.pop()
        self.queue.append(data)
        self.pos_queue.append(self.count)
        # 出
        if self.count - self.pos_queue[0] >= self.window:
            self.pos_queue.popleft()     
        self.count += 1
        return self.info
    
    @property
    def info(self):
        if self.count >= self.window:
            return self.queue[self.pos_queue[0]]
        else:
            return 0
# class TsMax:
#     def __init__(self, window):
#         self.window = window
#         self.queue = []
#         self.ordered_queue = []
#         self.full_queue = False
#         self.count = 0
    
#     def update(self, data):
#         if self.full_queue:
#             self.ordered_queue.pop(bisect.bisect_right(self.ordered_queue, self.queue.pop(0), lo=0) - 1)
#             self.ordered_queue.insert(bisect.bisect(self.ordered_queue, data, lo=0), data)
#             self.queue.append(data)
#         else:
#             self.count += 1
#             self.queue.append(data)
#             self.ordered_queue.insert(bisect.bisect_right(self.ordered_queue, data, lo=0), data)
#             if self.count >= self.window:
#                 self.full_queue = True
#         return self.info
    
#     @property
#     def info(self):
#         return self.ordered_queue[-1]
    
# ts_min
class TsMin:
    def __init__(self, window):
        self.window = window
        self.queue = []
        self.pos_queue = deque()
        self.count = 0
        
    def update(self, data):
        # 入
        while self.pos_queue and (self.queue[self.pos_queue[-1]]) <= - data:
            self.pos_queue.pop()
        self.queue.append(-data)
        self.pos_queue.append(self.count)
        # 出
        if self.count - self.pos_queue[0] >= self.window:
            self.pos_queue.popleft()     
        self.count += 1
        return self.info
    
    @property
    def info(self):
        if self.count >= self.window:
            return - self.queue[self.pos_queue[0]]
        else:
            return 0
# class TsMin:
#     def __init__(self, window):
#         self.window = window
#         self.queue = []
#         self.ordered_queue = []
#         self.full_queue = False
#         self.count = 0
    
#     def update(self, data):
#         if self.full_queue:
#             self.ordered_queue.pop(bisect.bisect_right(self.ordered_queue, self.queue.pop(0), lo=0) - 1)
#             self.ordered_queue.insert(bisect.bisect(self.ordered_queue, data, lo=0), data)
#             self.queue.append(data)
#         else:
#             self.count += 1
#             self.queue.append(data)
#             self.ordered_queue.insert(bisect.bisect_right(self.ordered_queue, data, lo=0), data)
#             if self.count >= self.window:
#                 self.full_queue = True
#         return self.info
    
#     @property
#     def info(self):
#         return self.ordered_queue[0]  

# ts_median
class TsMed:
    def __init__(self, window):
        self.q = 0.5
        self.loc = int(self.q * (window - 1))
        self.window = window
        self.queue = []
        self.ordered_queue = []
        self.full_queue = False
        self.count = 0
        if window % 2 == 0:
            self.odd = False
        else:
            self.odd = True
    
    def update(self, data):
        if self.full_queue:
            self.ordered_queue.pop(bisect.bisect_right(self.ordered_queue, self.queue.pop(0), lo=0) - 1)
            self.ordered_queue.insert(bisect.bisect(self.ordered_queue, data, lo=0), data)
            self.queue.append(data)
        else:
            self.count += 1
            self.queue.append(data)
            self.ordered_queue.insert(bisect.bisect_right(self.ordered_queue, data, lo=0), data)
            if self.count >= self.window:
                self.full_queue = True
        return self.info
    
    @property
    def info(self):
        if self.full_queue:  
            if self.odd:
                return self.ordered_queue[self.loc]
            else:
                return (self.ordered_queue[self.loc] + self.ordered_queue[self.loc + 1]) / 2
        else:
            return 0

# ts_range
class TsRange:
    def __init__(self, window):
        self.window = window
        self.max = TsMax(window)
        self.min = TsMin(window)
    
    def update(self, data):
        self.max.update(data)
        self.min.update(data)
        return self.info
    
    @property
    def info(self):
        return self.max.info - self.min.info

# class TsRange:
#     def __init__(self, window):
#         self.window = window
#         self.queue = []
#         self.ordered_queue = []
#         self.full_queue = False
#         self.count = 0
    
#     def update(self, data):
#         if self.full_queue:
#             self.ordered_queue.pop(bisect.bisect_right(self.ordered_queue, self.queue.pop(0), lo=0) - 1)
#             self.ordered_queue.insert(bisect.bisect(self.ordered_queue, data, lo=0), data)
#             self.queue.append(data)
#         else:
#             self.count += 1
#             self.queue.append(data)
#             self.ordered_queue.insert(bisect.bisect_right(self.ordered_queue, data, lo=0), data)
#             if self.count >= self.window:
#                 self.full_queue = True
#         return self.info
    
#     @property
#     def info(self):
#         return self.ordered_queue[-1] - self.ordered_queue[0]

# ts_up_ratio
class TsUpRatio:
    def __init__(self, window):
        self.window = window
        self.queue = []
        self.up_queue = []
        self.full_queue = False
        self.count = 0
        self._info = 0
    
    def update(self, data):
        if self.full_queue:
            self.queue.pop(0)
            self.up_queue.append(data > self.queue[-1])
            self._info += (self.up_queue[-1] - self.up_queue.pop(0))
            self.queue.append(data)
            
        else:
            self.count += 1
            self.queue.append(data)
            self.up_queue.append(data > self.queue[-1])
            self._info += self.up_queue[-1]
            if self.count >= self.window:
                self.full_queue = True
        return self.info
    
    @property
    def info(self):
        return self._info / self.window
    