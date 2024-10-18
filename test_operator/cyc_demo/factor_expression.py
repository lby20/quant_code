"""
@author: caiyucheng
@contact: caiyucheng23@163.com
@date: 2024-07-02
"""
from multiprocessing import shared_memory, Array
from .cyc_operator import *
from loguru import logger
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# DataContainer(multiple calls are needed later so use an abbreviation)
class DC:
    shared_memories = {}
    shm_map = {}
    factor_array = np.ndarray
    factor_ready = np.ndarray
    @staticmethod
    def set_params(stock_nums:int, field_names:list, dtypes:list, factor_list:list, n_jobs:int):
        """verify the specifications of the static data. 
        dtypes:the type of element of field_names
        """
        DC.factor_array = Array('f', stock_nums * len(factor_list))
        DC.factor_array = np.frombuffer(DC.factor_array.get_obj(), dtype='float32').reshape((stock_nums, len(factor_list)))
        # factor_size = np.dtype(np.float32).itemsize
        # signal_size = np.dtype(np.int32).itemsize
        # factor_total_size = stock_nums * len(factor_list) * factor_size
        # signal_total_size = n_jobs * signal_size
        # shm1 = shared_memory.SharedMemory(create=True, size=factor_total_size)
        # shm2 = shared_memory.SharedMemory(create=True, size=signal_total_size)
        # DC.factor_array = np.ndarray(shape=(stock_nums,len(factor_list)), dtype=np.float32, buffer=shm1.buf)
        # DC.factor_array[:,:] = 0
        # DC.factor_ready =  np.ndarray(n_jobs, dtype=np.int32, buffer=shm2.buf)
        # DC.factor_ready[:] = 0
        # DC.shm_map['factor_array'] = (shm1,)
        # DC.shm_map['factor_ready'] = (shm2,)
        DC.stock_nums = stock_nums
        for idx, fn in enumerate(field_names):
            DC.create_shared_memory(var_name=fn, dtype=dtypes[idx], shape=stock_nums)

    @staticmethod
    def create_shared_memory(var_name, dtype, shape):
        size = np.dtype(dtype).itemsize
        total_size = np.prod(shape) * size
        shm = shared_memory.SharedMemory(create=True, size=total_size)
        array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        DC.shared_memories[var_name] = array
        DC.shm_map[var_name] = (shm, shape, dtype)

    @staticmethod
    def update_shared_memory(data:pd.DataFrame):
        for var_name in data.columns:
            if not var_name in DC.shm_map: 
                logger.info(f"The field:{var_name} has not been initialized.")      
                # 为什么当别的函数再次调用这个函数(create_shared_memory)时,只能
                # DC.create_shared_memory(var_name=var_name, dtype=data[var_name].dtype, shape=data.shape[0])
                continue
            np.copyto(DC.shared_memories[var_name], data[var_name].values)

    # just calling this function is sufficient, eg DC.get('raw_close')
    @staticmethod
    def get(var_name):
        return DC.shared_memories[var_name]
    
    @staticmethod
    def close():
        shm_name = list(DC.shm_map.keys())
        for name in shm_name:
            DC.shm_map[name][0].close()
            DC.shm_map[name][0].unlink()


get = DC.get


class cyc_k1_alpha0001:
    time_consume=1
    def __init__(self):
        self.ts_avg1 = ts_avg(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return div(self.ts_avg1(get('amount_cyc')),get('market_cap_cyc'))


class cyc_k1_alpha0002:
    time_consume=11
    def __init__(self):
        self.ts_rank1 = ts_rank(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum1 = ts_sum(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_rank1(
            div(
                self.ts_sum1(get('amount_cyc')),
                get('market_cap_cyc')
                )
            )


class cyc_k1_alpha0003:
    time_consume=2
    def __init__(self):
        self.ts_sum1 = ts_sum(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum2 = ts_sum(240,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return div(self.ts_sum1(get('volume_cyc')),self.ts_sum2(get('volume_cyc')))
