"""
@author: caiyucheng
@contact: caiyucheng23@163.com
@date: 2024-07-02
"""
from multiprocessing import shared_memory, Array
from .cyc_operator import *
# from cyc_operator import *

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



class alpha_vboth_fx_0:
    time_consume=8
    def __init__(self):
        self.ts_tcv1 = ts_tcv(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_tcv1(self.ts_sum_div1(div(get('small_actS_equi_fx'),get('small_actS_cnt_equi_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_1:
    time_consume=8
    def __init__(self):
        self.ts_tcv1 = ts_tcv(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_tcv1(self.ts_sum_div1(div(get('small_actB_equi_fx'),get('small_actB_cnt_equi_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_2:
    time_consume=5
    def __init__(self):
        self.ts_corr1 = ts_corr(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_corr1(div(get('small_actS_equi_fx'),get('small_actS_cnt_equi_fx')),get('amount_trade_fx'))


class alpha_vboth_fx_3:
    time_consume=5
    def __init__(self):
        self.ts_corr1 = ts_corr(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_corr1(div(div(get('small_actS_equi_fx'),get('small_actS_cnt_equi_fx')),get('circulating_market_cap')),get('amount_trade_fx'))


class alpha_vboth_fx_4:
    time_consume=22
    def __init__(self):
        self.ts_min1 = ts_min(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(div_cs(self.ts_sum_div1(get('small_actS_down_fx'),get('small_actS_cnt_down_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_5:
    time_consume=5
    def __init__(self):
        self.ts_corr1 = ts_corr(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_corr1(div(div(get('small_actS_down_fx'),get('small_actS_cnt_down_fx')),get('circulating_market_cap')),get('amount_trade_fx'))


class alpha_vboth_fx_8:
    time_consume=22
    def __init__(self):
        self.ts_min1 = ts_min(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(div_cs(self.ts_sum_div1(get('sell_small_order_down_fx'),get('sell_small_order_down_cnt_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_9:
    time_consume=22
    def __init__(self):
        self.ts_min1 = ts_min(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(div_cs(self.ts_sum_div1(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_10:
    time_consume=25
    def __init__(self):
        self.ts_min1 = ts_min(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_avg1 = ts_avg(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div2 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_avg1(div(self.ts_sum_div1(add(get('buy_small_order_up_fx'),get('buy_big_order_up_fx')),add(get('buy_small_order_up_cnt_fx'),get('buy_big_order_up_cnt_fx'))),self.ts_sum_div2(add(get('buy_small_order_up_fx'),get('buy_big_order_up_fx')),add(get('buy_small_order_up_cnt_fx'),get('buy_big_order_up_cnt_fx'))))))


class alpha_vboth_fx_11:
    time_consume=22
    def __init__(self):
        self.ts_min1 = ts_min(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(div_cs(self.ts_sum_div1(get('small_actB_equi_fx'),get('small_actB_cnt_equi_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_12:
    time_consume=22
    def __init__(self):
        self.ts_min1 = ts_min(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_sum_div1(div(get('small_actB_equi_fx'),get('small_actB_cnt_equi_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_13:
    time_consume=22
    def __init__(self):
        self.ts_min1 = ts_min(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_sum_div1(div(get('small_actS_down_fx'),get('small_actS_cnt_down_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_15:
    time_consume=22
    def __init__(self):
        self.ts_min1 = ts_min(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_sum_div1(div(get('small_actB_up_fx'),get('small_actB_cnt_up_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_24:
    time_consume=8
    def __init__(self):
        self.ts_tcv1 = ts_tcv(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_tcv1(self.ts_sum_div1(div(get('small_actS_down_fx'),get('small_actS_cnt_down_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_28:
    time_consume=23
    def __init__(self):
        self.ts_min1 = ts_min(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_avg1 = ts_avg(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_avg1(div(self.ts_sum_div1(get('buy_small_order_down_fx'),get('buy_small_order_down_cnt_fx')),get('amount_trade_fx'))))


class alpha_vboth_fx_29:
    time_consume=5
    def __init__(self):
        self.ts_corr1 = ts_corr(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_corr1(div(div(get('small_actB_equi_fx'),get('small_actB_cnt_equi_fx')),get('circulating_market_cap')),get('amount_trade_fx'))


class alpha_vboth_fx_36:
    time_consume=22
    def __init__(self):
        self.ts_min1 = ts_min(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(div_cs(self.ts_sum_div1(get('small_actS_up_fx'),get('small_actS_cnt_up_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_39:
    time_consume=8
    def __init__(self):
        self.ts_tcv1 = ts_tcv(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_tcv1(self.ts_sum_div1(div(get('small_actS_down_fx'),get('small_actS_cnt_down_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_40:
    time_consume=22
    def __init__(self):
        self.ts_min1 = ts_min(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(div_cs(self.ts_sum_div1(get('small_actB_down_fx'),get('small_actB_cnt_down_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_41:
    time_consume=22
    def __init__(self):
        self.ts_min1 = ts_min(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_sum_div_cs1(div(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_45:
    time_consume=22
    def __init__(self):
        self.ts_min1 = ts_min(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_sum_div1(div(get('small_actS_equi_fx'),get('small_actS_cnt_equi_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_47:
    time_consume=23
    def __init__(self):
        self.ts_min1 = ts_min(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_avg1 = ts_avg(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_avg1(div(self.ts_sum_div1(get('sell_small_order_up_fx'),get('sell_small_order_up_cnt_fx')),get('amount_trade_fx'))))


class alpha_vboth_fx_50:
    time_consume=21
    def __init__(self):
        self.ts_min1 = ts_min(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_avg1 = ts_avg(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_avg1(div(div(get('small_actB_equi_fx'),get('small_actB_cnt_equi_fx')),get('amount_trade_fx'))))


class alpha_vboth_fx_64:
    time_consume=22
    def __init__(self):
        self.ts_min1 = ts_min(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_sum_div_cs1(div(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),div(get('sell_small_order_up_fx'),get('sell_small_order_up_cnt_fx'))))


class alpha_vboth_fx_70:
    time_consume=21
    def __init__(self):
        self.ts_min1 = ts_min(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_avg1 = ts_avg(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_avg1(div_cs(div(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),get('amount_trade_fx'))))


class alpha_vboth_fx_71:
    time_consume=21
    def __init__(self):
        self.ts_min1 = ts_min(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_avg1 = ts_avg(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_avg1(div_cs(div(get('sell_small_order_down_fx'),get('sell_small_order_down_cnt_fx')),get('amount_trade_fx'))))


class alpha_vboth_fx_72:
    time_consume=23
    def __init__(self):
        self.ts_min1 = ts_min(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_avg1 = ts_avg(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_avg1(div(self.ts_sum_div1(get('small_actS_down_fx'),get('small_actS_cnt_down_fx')),get('amount_trade_fx'))))


class alpha_vboth_fx_73:
    time_consume=23
    def __init__(self):
        self.ts_min1 = ts_min(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_avg1 = ts_avg(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_avg1(div(self.ts_sum_div1(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),get('amount_trade_fx'))))


class alpha_vboth_fx_84:
    time_consume=9
    def __init__(self):
        self.ts_std1 = ts_std(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div2 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_std1(div(self.ts_sum_div1(add(get('sell_small_order_down_fx'),get('sell_big_order_down_fx')),add(get('sell_small_order_down_cnt_fx'),get('sell_big_order_down_cnt_fx'))),self.ts_sum_div2(add(get('sell_small_order_up_fx'),get('sell_big_order_up_fx'),get('sell_small_order_down_fx'),get('sell_big_order_down_fx')),add(get('sell_small_order_up_cnt_fx'),get('sell_big_order_up_cnt_fx'),get('sell_small_order_down_cnt_fx'),get('sell_big_order_down_cnt_fx')))))


class alpha_vboth_fx_85:
    time_consume=22
    def __init__(self):
        self.ts_min1 = ts_min(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_sum_div1(div(get('small_actS_down_fx'),get('small_actS_cnt_down_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_88:
    time_consume=5
    def __init__(self):
        self.ts_corr1 = ts_corr(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_corr1(div(get('sell_big_order_act_tot_fx'),get('amount_trade_fx')),get('amount_trade_fx'))


class alpha_vboth_fx_89:
    time_consume=22
    def __init__(self):
        self.ts_min1 = ts_min(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(div_cs(self.ts_sum_div1(get('small_actS_down_fx'),get('small_actS_cnt_down_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_92:
    time_consume=4
    def __init__(self):
        self.ts_sum_div1 = ts_sum_div(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div2 = ts_sum_div(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_sum_div1(self.ts_sum_div2(get('sell_big_order_down_fx'),get('sell_big_order_down_cnt_fx')),get('amount_trade_fx'))


class alpha_vboth_fx_100:
    time_consume=9
    def __init__(self):
        self.ts_std1 = ts_std(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div2 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_std1(div(self.ts_sum_div1(get('sell_small_order_down_fx'),get('sell_small_order_down_cnt_fx')),self.ts_sum_div2(get('sell_small_order_up_fx'),get('sell_small_order_up_cnt_fx'))))


class alpha_vboth_fx_104:
    time_consume=22
    def __init__(self):
        self.ts_min1 = ts_min(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(div_cs(self.ts_sum_div1(get('sell_small_order_down_fx'),get('sell_small_order_down_cnt_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_105:
    time_consume=22
    def __init__(self):
        self.ts_min1 = ts_min(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_sum_div_cs1(div(add(get('buy_small_order_up_fx'),get('buy_big_order_up_fx')),add(get('buy_small_order_up_cnt_fx'),get('buy_big_order_up_cnt_fx'))),div(add(get('sell_small_order_up_fx'),get('sell_big_order_up_fx')),add(get('sell_small_order_up_cnt_fx'),get('sell_big_order_up_cnt_fx')))))


class alpha_vboth_fx_111:
    time_consume=5
    def __init__(self):
        self.ts_corr1 = ts_corr(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_corr1(div(div(get('sell_big_order_down_fx'),get('sell_big_order_down_cnt_fx')),get('amount_trade_fx')),get('amount_trade_fx'))


class alpha_vboth_fx_112:
    time_consume=21
    def __init__(self):
        self.ts_min1 = ts_min(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_avg1 = ts_avg(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_avg1(div_cs(div(get('small_actB_equi_fx'),get('small_actB_cnt_equi_fx')),get('amount_trade_fx'))))


class alpha_vboth_fx_114:
    time_consume=22
    def __init__(self):
        self.ts_min1 = ts_min(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(div_cs(self.ts_sum_div1(get('buy_small_order_down_fx'),get('buy_small_order_down_cnt_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_119:
    time_consume=23
    def __init__(self):
        self.ts_min1 = ts_min(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_avg1 = ts_avg(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_avg1(div_cs(self.ts_sum_div1(get('buy_small_order_down_fx'),get('buy_small_order_down_cnt_fx')),get('amount_trade_fx'))))


class alpha_vboth_fx_122:
    time_consume=22
    def __init__(self):
        self.ts_min1 = ts_min(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_sum_div1(div(add(get('buy_small_order_up_fx'),get('buy_big_order_up_fx')),add(get('buy_small_order_up_cnt_fx'),get('buy_big_order_up_cnt_fx'))),div(add(get('sell_small_order_up_fx'),get('sell_big_order_up_fx')),add(get('sell_small_order_up_cnt_fx'),get('sell_big_order_up_cnt_fx')))))


class alpha_vboth_fx_132:
    time_consume=10
    def __init__(self):
        self.ts_tcv1 = ts_tcv(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div2 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_tcv1(div_cs(self.ts_sum_div1(get('sell_small_order_down_fx'),get('sell_small_order_down_cnt_fx')),self.ts_sum_div2(add(get('sell_small_order_up_fx'),get('sell_big_order_up_fx'),get('sell_small_order_down_fx'),get('sell_big_order_down_fx')),add(get('sell_small_order_up_cnt_fx'),get('sell_big_order_up_cnt_fx'),get('sell_small_order_down_cnt_fx'),get('sell_big_order_down_cnt_fx')))))


class alpha_vboth_fx_133:
    time_consume=22
    def __init__(self):
        self.ts_min1 = ts_min(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_sum_div1(div(get('small_actB_equi_fx'),get('small_actB_cnt_equi_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_134:
    time_consume=10
    def __init__(self):
        self.ts_cv_abs1 = ts_cv_abs(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div2 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_cv_abs1(div_cs(self.ts_sum_div1(get('sell_small_order_down_fx'),get('sell_small_order_down_cnt_fx')),self.ts_sum_div2(add(get('sell_small_order_up_fx'),get('sell_big_order_up_fx'),get('sell_small_order_down_fx'),get('sell_big_order_down_fx')),add(get('sell_small_order_up_cnt_fx'),get('sell_big_order_up_cnt_fx'),get('sell_small_order_down_cnt_fx'),get('sell_big_order_down_cnt_fx')))))


class alpha_vboth_fx_141:
    time_consume=22
    def __init__(self):
        self.ts_min1 = ts_min(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(div_cs(self.ts_sum_div1(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_154:
    time_consume=23
    def __init__(self):
        self.ts_min1 = ts_min(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_avg1 = ts_avg(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_avg1(div_cs(self.ts_sum_div1(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),get('amount_trade_fx'))))


class alpha_vboth_fx_161:
    time_consume=21
    def __init__(self):
        self.ts_max1 = ts_max(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_avg1 = ts_avg(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_max1(self.ts_avg1(div_cs(add(get('sell_small_order_tot_fx'),get('sell_big_order_tot_fx')),add(get('sell_small_order_act_tot_fx'),get('sell_big_order_act_tot_fx')))))


class alpha_vboth_fx_163:
    time_consume=10
    def __init__(self):
        self.ts_tcv1 = ts_tcv(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div2 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_tcv1(div_cs(self.ts_sum_div1(add(get('sell_small_order_down_fx'),get('sell_big_order_down_fx')),add(get('sell_small_order_down_cnt_fx'),get('sell_big_order_down_cnt_fx'))),self.ts_sum_div2(add(get('sell_small_order_up_fx'),get('sell_big_order_up_fx'),get('sell_small_order_down_fx'),get('sell_big_order_down_fx')),add(get('sell_small_order_up_cnt_fx'),get('sell_big_order_up_cnt_fx'),get('sell_small_order_down_cnt_fx'),get('sell_big_order_down_cnt_fx')))))


class alpha_vboth_fx_164:
    time_consume=10
    def __init__(self):
        self.ts_cv_abs1 = ts_cv_abs(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div2 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_cv_abs1(div(self.ts_sum_div1(get('sell_small_order_down_fx'),get('sell_small_order_down_cnt_fx')),self.ts_sum_div2(get('sell_small_order_up_fx'),get('sell_small_order_up_cnt_fx'))))


class alpha_vboth_fx_165:
    time_consume=10
    def __init__(self):
        self.ts_cv_abs1 = ts_cv_abs(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div2 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_cv_abs1(div(self.ts_sum_div1(get('sell_small_order_down_fx'),get('sell_small_order_down_cnt_fx')),self.ts_sum_div2(get('sell_small_order_up_fx'),get('sell_small_order_up_cnt_fx'))))


class alpha_vboth_fx_168:
    time_consume=23
    def __init__(self):
        self.ts_min1 = ts_min(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_avg1 = ts_avg(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_avg1(div_cs(self.ts_sum_div1(get('sell_small_order_up_fx'),get('sell_small_order_up_cnt_fx')),get('amount_trade_fx'))))


class alpha_vboth_fx_170:
    time_consume=6
    def __init__(self):
        self.ts_cv_abs1 = ts_cv_abs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_cv_abs1(div(div(add(get('buy_small_order_up_fx'),get('buy_big_order_up_fx')),add(get('buy_small_order_up_cnt_fx'),get('buy_big_order_up_cnt_fx'))),div(add(get('buy_small_order_up_fx'),get('buy_big_order_up_fx'),get('buy_small_order_down_fx'),get('buy_big_order_down_fx')),add(get('buy_small_order_up_cnt_fx'),get('buy_big_order_up_cnt_fx'),get('buy_small_order_down_cnt_fx'),get('buy_big_order_down_cnt_fx')))))


class alpha_vboth_fx_175:
    time_consume=6
    def __init__(self):
        self.ts_cv_abs1 = ts_cv_abs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_cv_abs1(div(div(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),div(add(get('buy_small_order_up_fx'),get('buy_big_order_up_fx'),get('buy_small_order_down_fx'),get('buy_big_order_down_fx')),add(get('buy_small_order_up_cnt_fx'),get('buy_big_order_up_cnt_fx'),get('buy_small_order_down_cnt_fx'),get('buy_big_order_down_cnt_fx')))))


class alpha_vboth_fx_184:
    time_consume=6
    def __init__(self):
        self.ts_cv_abs1 = ts_cv_abs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_cv_abs1(div(div(get('sell_small_order_down_fx'),get('sell_small_order_down_cnt_fx')),div(add(get('sell_small_order_down_fx'),get('sell_big_order_down_fx')),add(get('sell_small_order_down_cnt_fx'),get('sell_big_order_down_cnt_fx')))))


class alpha_vboth_fx_186:
    time_consume=6
    def __init__(self):
        self.ts_cv_abs1 = ts_cv_abs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_cv_abs1(div(div(get('sell_small_order_down_fx'),get('sell_small_order_down_cnt_fx')),div(add(get('sell_small_order_down_fx'),get('sell_big_order_down_fx')),add(get('sell_small_order_down_cnt_fx'),get('sell_big_order_down_cnt_fx')))))


class alpha_vboth_fx_190:
    time_consume=2
    def __init__(self):
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_sum_div1(div(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),get('amount_trade_fx'))


class alpha_vboth_fx_193:
    time_consume=8
    def __init__(self):
        self.ts_tcv1 = ts_tcv(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_tcv1(self.ts_sum_div_cs1(get('buy_small_order_down_fx'),get('amount_trade_fx')))


class alpha_vboth_fx_195:
    time_consume=4
    def __init__(self):
        self.ts_sum_div1 = ts_sum_div(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div2 = ts_sum_div(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_sum_div1(self.ts_sum_div2(get('small_actB_down_fx'),get('small_actB_cnt_down_fx')),get('amount_trade_fx'))


class alpha_vboth_fx_196:
    time_consume=22
    def __init__(self):
        self.ts_min1 = ts_min(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_sum_div1(div(get('small_actS_equi_fx'),get('small_actS_cnt_equi_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_198:
    time_consume=6
    def __init__(self):
        self.ts_cv_abs1 = ts_cv_abs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_cv_abs1(div_cs(div(get('sell_small_order_down_fx'),get('sell_small_order_down_cnt_fx')),div(add(get('sell_small_order_up_fx'),get('sell_big_order_up_fx'),get('sell_small_order_down_fx'),get('sell_big_order_down_fx')),add(get('sell_small_order_up_cnt_fx'),get('sell_big_order_up_cnt_fx'),get('sell_small_order_down_cnt_fx'),get('sell_big_order_down_cnt_fx')))))


class alpha_vboth_fx_207:
    time_consume=6
    def __init__(self):
        self.ts_cv_abs1 = ts_cv_abs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_cv_abs1(div(div(add(get('sell_small_order_down_fx'),get('sell_big_order_down_fx')),add(get('sell_small_order_down_cnt_fx'),get('sell_big_order_down_cnt_fx'))),div(add(get('sell_small_order_up_fx'),get('sell_big_order_up_fx'),get('sell_small_order_down_fx'),get('sell_big_order_down_fx')),add(get('sell_small_order_up_cnt_fx'),get('sell_big_order_up_cnt_fx'),get('sell_small_order_down_cnt_fx'),get('sell_big_order_down_cnt_fx')))))


class alpha_vboth_fx_211:
    time_consume=6
    def __init__(self):
        self.ts_cv_abs1 = ts_cv_abs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_cv_abs1(div(div(get('sell_small_order_down_fx'),get('sell_small_order_down_cnt_fx')),div(add(get('sell_small_order_up_fx'),get('sell_big_order_up_fx'),get('sell_small_order_down_fx'),get('sell_big_order_down_fx')),add(get('sell_small_order_up_cnt_fx'),get('sell_big_order_up_cnt_fx'),get('sell_small_order_down_cnt_fx'),get('sell_big_order_down_cnt_fx')))))


class alpha_vboth_fx_212:
    time_consume=8
    def __init__(self):
        self.ts_cv_abs1 = ts_cv_abs(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_cv_abs1(self.ts_sum_div_cs1(get('buy_small_order_down_fx'),get('amount_trade_fx')))


class alpha_vboth_fx_220:
    time_consume=21
    def __init__(self):
        self.ts_min1 = ts_min(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_avg1 = ts_avg(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_avg1(div(div(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),get('amount_trade_fx'))))


class alpha_vboth_fx_230:
    time_consume=6
    def __init__(self):
        self.ts_tcv1 = ts_tcv(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_tcv1(div_cs(div(get('sell_small_order_down_fx'),get('sell_small_order_down_cnt_fx')),div(add(get('sell_small_order_up_fx'),get('sell_big_order_up_fx'),get('sell_small_order_down_fx'),get('sell_big_order_down_fx')),add(get('sell_small_order_up_cnt_fx'),get('sell_big_order_up_cnt_fx'),get('sell_small_order_down_cnt_fx'),get('sell_big_order_down_cnt_fx')))))


class alpha_vboth_fx_233:
    time_consume=44
    def __init__(self):
        self.ts_range1 = ts_range(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div2 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_range1(div_cs(self.ts_sum_div1(add(get('sell_small_order_up_fx'),get('sell_big_order_up_fx')),add(get('sell_small_order_up_cnt_fx'),get('sell_big_order_up_cnt_fx'))),self.ts_sum_div2(add(get('sell_small_order_down_fx'),get('sell_big_order_down_fx')),add(get('sell_small_order_down_cnt_fx'),get('sell_big_order_down_cnt_fx')))))


class alpha_vboth_fx_243:
    time_consume=3
    def __init__(self):
        self.ts_avg1 = ts_avg(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_avg1(div_cs(self.ts_sum_div1(get('sell_big_order_down_fx'),get('sell_big_order_down_cnt_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_250:
    time_consume=23
    def __init__(self):
        self.ts_min1 = ts_min(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_avg1 = ts_avg(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_avg1(div(self.ts_sum_div1(get('small_actS_down_fx'),get('small_actS_cnt_down_fx')),get('amount_trade_fx'))))


class alpha_vboth_fx_255:
    time_consume=5
    def __init__(self):
        self.ts_skew1 = ts_skew(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_skew1(div(div(get('small_actB_equi_fx'),get('small_actB_cnt_equi_fx')),get('circulating_market_cap')))


class alpha_vboth_fx_259:
    time_consume=21
    def __init__(self):
        self.ts_min1 = ts_min(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_avg1 = ts_avg(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_avg1(div_cs(div(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),get('amount_trade_fx'))))


class alpha_vboth_fx_263:
    time_consume=5
    def __init__(self):
        self.ts_std1 = ts_std(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_std1(div(get('buy_small_order_down_fx'),get('amount_trade_fx')))


class alpha_vboth_fx_264:
    time_consume=7
    def __init__(self):
        self.ts_corr1 = ts_corr(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_corr1(div_cs(self.ts_sum_div1(get('sell_small_order_up_fx'),get('sell_small_order_up_cnt_fx')),get('amount_trade_fx')),mul(get('close_trade_fx'),get('fq_factor_trade_fx')))


class alpha_vboth_fx_265:
    time_consume=4
    def __init__(self):
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div2 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_sum_div1(self.ts_sum_div2(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),get('amount_trade_fx'))


class alpha_vboth_fx_267:
    time_consume=8
    def __init__(self):
        self.ts_cv_abs1 = ts_cv_abs(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_cv_abs1(self.ts_sum_div1(get('small_actS_equi_fx'),get('small_actS_cnt_equi_fx')))


class alpha_vboth_fx_275:
    time_consume=22
    def __init__(self):
        self.ts_min1 = ts_min(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_sum_div_cs1(div(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),div(add(get('sell_small_order_up_fx'),get('sell_big_order_up_fx')),add(get('sell_small_order_up_cnt_fx'),get('sell_big_order_up_cnt_fx')))))


class alpha_vboth_fx_277:
    time_consume=5
    def __init__(self):
        self.ts_corr1 = ts_corr(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_corr1(div(div(get('small_actB_up_fx'),get('small_actB_cnt_up_fx')),get('amount_trade_fx')),mul(get('close_trade_fx'),get('fq_factor_trade_fx')))


class alpha_vboth_fx_282:
    time_consume=9
    def __init__(self):
        self.ts_std1 = ts_std(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div2 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_std1(div(self.ts_sum_div1(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),self.ts_sum_div2(add(get('buy_small_order_up_fx'),get('buy_big_order_up_fx'),get('buy_small_order_down_fx'),get('buy_big_order_down_fx')),add(get('buy_small_order_up_cnt_fx'),get('buy_big_order_up_cnt_fx'),get('buy_small_order_down_cnt_fx'),get('buy_big_order_down_cnt_fx')))))


class alpha_vboth_fx_287:
    time_consume=23
    def __init__(self):
        self.ts_min1 = ts_min(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_avg1 = ts_avg(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_avg1(div(self.ts_sum_div1(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),get('amount_trade_fx'))))


class alpha_vboth_fx_292:
    time_consume=9
    def __init__(self):
        self.ts_std1 = ts_std(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div2 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_std1(div_cs(self.ts_sum_div1(get('sell_small_order_down_fx'),get('sell_small_order_down_cnt_fx')),self.ts_sum_div2(add(get('sell_small_order_up_fx'),get('sell_big_order_up_fx'),get('sell_small_order_down_fx'),get('sell_big_order_down_fx')),add(get('sell_small_order_up_cnt_fx'),get('sell_big_order_up_cnt_fx'),get('sell_small_order_down_cnt_fx'),get('sell_big_order_down_cnt_fx')))))


class alpha_vboth_fx_301:
    time_consume=5
    def __init__(self):
        self.ts_corr1 = ts_corr(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_corr1(div(get('small_actS_fx'),get('amount_trade_fx')),div(get('high_trade_fx'),get('low_trade_fx')))


class alpha_vboth_fx_302:
    time_consume=42
    def __init__(self):
        self.ts_range1 = ts_range(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_range1(self.ts_sum_div1(div(get('sell_small_order_down_fx'),get('sell_small_order_down_cnt_fx')),div(add(get('sell_small_order_up_fx'),get('sell_big_order_up_fx'),get('sell_small_order_down_fx'),get('sell_big_order_down_fx')),add(get('sell_small_order_up_cnt_fx'),get('sell_big_order_up_cnt_fx'),get('sell_small_order_down_cnt_fx'),get('sell_big_order_down_cnt_fx')))))


class alpha_vboth_fx_303:
    time_consume=10
    def __init__(self):
        self.ts_cv_abs1 = ts_cv_abs(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div2 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_cv_abs1(div_cs(self.ts_sum_div1(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),self.ts_sum_div2(add(get('buy_small_order_up_fx'),get('buy_big_order_up_fx'),get('buy_small_order_down_fx'),get('buy_big_order_down_fx')),add(get('buy_small_order_up_cnt_fx'),get('buy_big_order_up_cnt_fx'),get('buy_small_order_down_cnt_fx'),get('buy_big_order_down_cnt_fx')))))


class alpha_vboth_fx_304:
    time_consume=10
    def __init__(self):
        self.ts_tcv1 = ts_tcv(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div2 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_tcv1(div_cs(self.ts_sum_div1(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),self.ts_sum_div2(add(get('buy_small_order_up_fx'),get('buy_big_order_up_fx'),get('buy_small_order_down_fx'),get('buy_big_order_down_fx')),add(get('buy_small_order_up_cnt_fx'),get('buy_big_order_up_cnt_fx'),get('buy_small_order_down_cnt_fx'),get('buy_big_order_down_cnt_fx')))))


class alpha_vboth_fx_306:
    time_consume=8
    def __init__(self):
        self.ts_cv_abs1 = ts_cv_abs(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_cv_abs1(self.ts_sum_div1(get('small_actB_equi_fx'),get('small_actB_cnt_equi_fx')))


class alpha_vboth_fx_312:
    time_consume=8
    def __init__(self):
        self.ts_tcv1 = ts_tcv(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_tcv1(div_cs(self.ts_sum_div1(get('sell_big_order_down_fx'),get('sell_big_order_down_cnt_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_313:
    time_consume=23
    def __init__(self):
        self.ts_min1 = ts_min(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_avg1 = ts_avg(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_avg1(div(self.ts_sum_div1(get('small_actB_up_fx'),get('small_actB_cnt_up_fx')),get('amount_trade_fx'))))


class alpha_vboth_fx_314:
    time_consume=5
    def __init__(self):
        self.ts_std1 = ts_std(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_std1(div_cs(div(add(get('buy_small_order_up_fx'),get('buy_big_order_up_fx')),add(get('buy_small_order_up_cnt_fx'),get('buy_big_order_up_cnt_fx'))),div(add(get('buy_small_order_up_fx'),get('buy_big_order_up_fx'),get('buy_small_order_down_fx'),get('buy_big_order_down_fx')),add(get('buy_small_order_up_cnt_fx'),get('buy_big_order_up_cnt_fx'),get('buy_small_order_down_cnt_fx'),get('buy_big_order_down_cnt_fx')))))


class alpha_vboth_fx_316:
    time_consume=5
    def __init__(self):
        self.ts_std1 = ts_std(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_std1(div_cs(div(add(get('buy_small_order_up_fx'),get('buy_big_order_up_fx')),add(get('buy_small_order_up_cnt_fx'),get('buy_big_order_up_cnt_fx'))),div(add(get('buy_small_order_up_fx'),get('buy_big_order_up_fx'),get('buy_small_order_down_fx'),get('buy_big_order_down_fx')),add(get('buy_small_order_up_cnt_fx'),get('buy_big_order_up_cnt_fx'),get('buy_small_order_down_cnt_fx'),get('buy_big_order_down_cnt_fx')))))


class alpha_vboth_fx_321:
    time_consume=9
    def __init__(self):
        self.ts_std1 = ts_std(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div2 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_std1(div_cs(self.ts_sum_div1(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),self.ts_sum_div2(add(get('sell_small_order_up_fx'),get('sell_big_order_up_fx'),get('sell_small_order_down_fx'),get('sell_big_order_down_fx')),add(get('sell_small_order_up_cnt_fx'),get('sell_big_order_up_cnt_fx'),get('sell_small_order_down_cnt_fx'),get('sell_big_order_down_cnt_fx')))))


class alpha_vboth_fx_324:
    time_consume=6
    def __init__(self):
        self.ts_cv_abs1 = ts_cv_abs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_cv_abs1(div_cs(get('sell_small_order_down_fx'),get('amount_trade_fx')))


class alpha_vboth_fx_325:
    time_consume=7
    def __init__(self):
        self.ts_std1 = ts_std(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_std1(self.ts_sum_div_cs1(add(get('buy_small_order_act_up_fx'),get('buy_big_order_act_up_fx'),get('buy_small_order_act_down_fx'),get('buy_big_order_act_down_fx')),add(get('buy_small_order_up_fx'),get('buy_big_order_up_fx'),get('buy_small_order_down_fx'),get('buy_big_order_down_fx'))))


class alpha_vboth_fx_326:
    time_consume=1
    def __init__(self):
        self.ts_avg1 = ts_avg(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_avg1(div_cs(div(get('sell_small_order_down_fx'),get('sell_small_order_down_cnt_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_329:
    time_consume=6
    def __init__(self):
        self.ts_tcv1 = ts_tcv(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_tcv1(div_cs(get('buy_small_order_down_fx'),get('amount_trade_fx')))


class alpha_vboth_fx_332:
    time_consume=8
    def __init__(self):
        self.ts_cv_abs1 = ts_cv_abs(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_cv_abs1(self.ts_sum_div_cs1(div(get('sell_small_order_down_fx'),get('sell_small_order_down_cnt_fx')),div(add(get('sell_small_order_up_fx'),get('sell_big_order_up_fx'),get('sell_small_order_down_fx'),get('sell_big_order_down_fx')),add(get('sell_small_order_up_cnt_fx'),get('sell_big_order_up_cnt_fx'),get('sell_small_order_down_cnt_fx'),get('sell_big_order_down_cnt_fx')))))


class alpha_vboth_fx_334:
    time_consume=5
    def __init__(self):
        self.ts_std1 = ts_std(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_std1(div(get('sell_small_order_down_fx'),get('amount_trade_fx')))


class alpha_vboth_fx_337:
    time_consume=6
    def __init__(self):
        self.ts_cv_abs1 = ts_cv_abs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_cv_abs1(div_cs(get('buy_small_order_down_fx'),get('amount_trade_fx')))


class alpha_vboth_fx_341:
    time_consume=22
    def __init__(self):
        self.ts_min1 = ts_min(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_sum_div1(div(add(get('sell_small_order_down_fx'),get('sell_big_order_down_fx')),add(get('sell_small_order_down_cnt_fx'),get('sell_big_order_down_cnt_fx'))),div(add(get('sell_small_order_up_fx'),get('sell_big_order_up_fx')),add(get('sell_small_order_up_cnt_fx'),get('sell_big_order_up_cnt_fx')))))


class alpha_vboth_fx_352:
    time_consume=3
    def __init__(self):
        self.ts_avg1 = ts_avg(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_avg1(div(self.ts_sum_div1(get('sell_big_order_down_fx'),get('sell_big_order_down_cnt_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_355:
    time_consume=24
    def __init__(self):
        self.ts_max1 = ts_max(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div2 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_max1(div(self.ts_sum_div1(get('sell_small_order_down_fx'),get('sell_small_order_down_cnt_fx')),self.ts_sum_div2(get('sell_small_order_up_fx'),get('sell_small_order_up_cnt_fx'))))


class alpha_vboth_fx_356:
    time_consume=24
    def __init__(self):
        self.ts_max1 = ts_max(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div2 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_max1(div(self.ts_sum_div1(get('sell_small_order_down_fx'),get('sell_small_order_down_cnt_fx')),self.ts_sum_div2(get('sell_small_order_up_fx'),get('sell_small_order_up_cnt_fx'))))


class alpha_vboth_fx_359:
    time_consume=26
    def __init__(self):
        self.ts_min1 = ts_min(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div2 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_sum_div_cs1(self.ts_sum_div1(get('buy_small_order_down_fx'),get('buy_small_order_down_cnt_fx')),self.ts_sum_div2(get('buy_big_order_down_fx'),get('buy_big_order_down_cnt_fx'))))


class alpha_vboth_fx_363:
    time_consume=6
    def __init__(self):
        self.ts_cv_abs1 = ts_cv_abs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_cv_abs1(div_cs(div(add(get('sell_small_order_down_fx'),get('sell_big_order_down_fx')),add(get('sell_small_order_down_cnt_fx'),get('sell_big_order_down_cnt_fx'))),div(add(get('sell_small_order_up_fx'),get('sell_big_order_up_fx'),get('sell_small_order_down_fx'),get('sell_big_order_down_fx')),add(get('sell_small_order_up_cnt_fx'),get('sell_big_order_up_cnt_fx'),get('sell_small_order_down_cnt_fx'),get('sell_big_order_down_cnt_fx')))))


class alpha_vboth_fx_367:
    time_consume=26
    def __init__(self):
        self.ts_min1 = ts_min(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div2 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_sum_div_cs1(self.ts_sum_div1(get('buy_small_order_down_fx'),get('buy_small_order_down_cnt_fx')),self.ts_sum_div2(get('sell_big_order_up_fx'),get('sell_big_order_up_cnt_fx'))))


class alpha_vboth_fx_369:
    time_consume=22
    def __init__(self):
        self.ts_min1 = ts_min(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_sum_div_cs1(div(get('buy_small_order_down_fx'),get('buy_small_order_down_cnt_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_371:
    time_consume=22
    def __init__(self):
        self.ts_min1 = ts_min(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(div_cs(self.ts_sum_div1(get('sell_big_order_down_fx'),get('sell_big_order_down_cnt_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_375:
    time_consume=6
    def __init__(self):
        self.ts_cv_abs1 = ts_cv_abs(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_cv_abs1(div(div(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),div(add(get('buy_small_order_up_fx'),get('buy_big_order_up_fx'),get('buy_small_order_down_fx'),get('buy_big_order_down_fx')),add(get('buy_small_order_up_cnt_fx'),get('buy_big_order_up_cnt_fx'),get('buy_small_order_down_cnt_fx'),get('buy_big_order_down_cnt_fx')))))


class alpha_vboth_fx_380:
    time_consume=6
    def __init__(self):
        self.ts_cv_abs1 = ts_cv_abs(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_cv_abs1(div(div(get('buy_small_order_down_fx'),get('buy_small_order_down_cnt_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_381:
    time_consume=2
    def __init__(self):
        self.ts_sum_div_cs1 = ts_sum_div_cs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_sum_div_cs1(div(add(get('sell_small_order_down_fx'),get('sell_big_order_down_fx')),add(get('sell_small_order_down_cnt_fx'),get('sell_big_order_down_cnt_fx'))),div(add(get('sell_small_order_up_fx'),get('sell_big_order_up_fx')),add(get('sell_small_order_up_cnt_fx'),get('sell_big_order_up_cnt_fx'))))


class alpha_vboth_fx_389:
    time_consume=26
    def __init__(self):
        self.ts_min1 = ts_min(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div2 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_sum_div_cs1(self.ts_sum_div1(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),self.ts_sum_div2(get('sell_big_order_up_fx'),get('sell_big_order_up_cnt_fx'))))


class alpha_vboth_fx_390:
    time_consume=44
    def __init__(self):
        self.ts_range1 = ts_range(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div2 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_range1(div_cs(self.ts_sum_div1(get('sell_small_order_down_fx'),get('sell_small_order_down_cnt_fx')),self.ts_sum_div2(add(get('sell_small_order_up_fx'),get('sell_big_order_up_fx'),get('sell_small_order_down_fx'),get('sell_big_order_down_fx')),add(get('sell_small_order_up_cnt_fx'),get('sell_big_order_up_cnt_fx'),get('sell_small_order_down_cnt_fx'),get('sell_big_order_down_cnt_fx')))))


class alpha_vboth_fx_397:
    time_consume=4
    def __init__(self):
        self.ts_sum_div1 = ts_sum_div(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div2 = ts_sum_div(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_sum_div1(self.ts_sum_div2(get('buy_big_order_up_fx'),get('buy_big_order_up_cnt_fx')),get('amount_trade_fx'))


class alpha_vboth_fx_400:
    time_consume=22
    def __init__(self):
        self.ts_min1 = ts_min(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_sum_div_cs1(div(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),div(get('buy_big_order_up_fx'),get('buy_big_order_up_cnt_fx'))))


class alpha_vboth_fx_402:
    time_consume=10
    def __init__(self):
        self.ts_tcv1 = ts_tcv(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div2 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_tcv1(div_cs(self.ts_sum_div1(get('sell_small_order_down_fx'),get('sell_small_order_down_cnt_fx')),self.ts_sum_div2(add(get('sell_small_order_up_fx'),get('sell_big_order_up_fx')),add(get('sell_small_order_up_cnt_fx'),get('sell_big_order_up_cnt_fx')))))


class alpha_vboth_fx_407:
    time_consume=44
    def __init__(self):
        self.ts_range1 = ts_range(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div2 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_range1(div(self.ts_sum_div1(add(get('buy_small_order_up_fx'),get('buy_big_order_up_fx')),add(get('buy_small_order_up_cnt_fx'),get('buy_big_order_up_cnt_fx'))),self.ts_sum_div2(add(get('buy_small_order_up_fx'),get('buy_big_order_up_fx'),get('buy_small_order_down_fx'),get('buy_big_order_down_fx')),add(get('buy_small_order_up_cnt_fx'),get('buy_big_order_up_cnt_fx'),get('buy_small_order_down_cnt_fx'),get('buy_big_order_down_cnt_fx')))))


class alpha_vboth_fx_412:
    time_consume=42
    def __init__(self):
        self.ts_range1 = ts_range(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_range1(div_cs(self.ts_sum_div1(get('small_actB_down_fx'),get('small_actB_cnt_down_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_417:
    time_consume=22
    def __init__(self):
        self.ts_min1 = ts_min(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_sum_div_cs1(div(add(get('sell_small_order_down_fx'),get('sell_big_order_down_fx')),add(get('sell_small_order_down_cnt_fx'),get('sell_big_order_down_cnt_fx'))),div(add(get('sell_small_order_up_fx'),get('sell_big_order_up_fx')),add(get('sell_small_order_up_cnt_fx'),get('sell_big_order_up_cnt_fx')))))


class alpha_vboth_fx_422:
    time_consume=5
    def __init__(self):
        self.ts_std1 = ts_std(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_std1(div(get('buy_small_order_up_fx'),get('amount_trade_fx')))


class alpha_vboth_fx_428:
    time_consume=5
    def __init__(self):
        self.ts_std1 = ts_std(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_std1(div_cs(get('sell_small_order_act_down_fx'),get('amount_trade_fx')))


class alpha_vboth_fx_438:
    time_consume=6
    def __init__(self):
        self.ts_cv_abs1 = ts_cv_abs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_cv_abs1(div(get('small_actS_equi_fx'),get('small_actS_cnt_equi_fx')))


class alpha_vboth_fx_439:
    time_consume=3
    def __init__(self):
        self.ts_avg1 = ts_avg(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_avg1(div(self.ts_sum_div1(get('small_actB_down_fx'),get('small_actB_cnt_down_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_440:
    time_consume=22
    def __init__(self):
        self.ts_min1 = ts_min(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_sum_div1(div(get('buy_small_order_down_fx'),get('buy_small_order_down_cnt_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_441:
    time_consume=7
    def __init__(self):
        self.ts_corr1 = ts_corr(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_corr1(div(self.ts_sum_div1(get('sell_small_order_up_fx'),get('sell_small_order_up_cnt_fx')),get('amount_trade_fx')),mul(get('close_trade_fx'),get('fq_factor_trade_fx')))


class alpha_vboth_fx_448:
    time_consume=22
    def __init__(self):
        self.ts_min1 = ts_min(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_sum_div1(div(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),div(get('sell_small_order_up_fx'),get('sell_small_order_up_cnt_fx'))))


class alpha_vboth_fx_452:
    time_consume=5
    def __init__(self):
        self.ts_std1 = ts_std(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_std1(div_cs(div(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),div(add(get('sell_small_order_down_fx'),get('sell_big_order_down_fx')),add(get('sell_small_order_down_cnt_fx'),get('sell_big_order_down_cnt_fx')))))


class alpha_vboth_fx_453:
    time_consume=21
    def __init__(self):
        self.ts_max1 = ts_max(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_avg1 = ts_avg(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_max1(self.ts_avg1(div(get('sell_small_order_act_down_fx'),get('amount_trade_fx'))))


class alpha_vboth_fx_458:
    time_consume=6
    def __init__(self):
        self.ts_cv_abs1 = ts_cv_abs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_cv_abs1(div_cs(get('buy_small_order_up_fx'),get('amount_trade_fx')))


class alpha_vboth_fx_459:
    time_consume=5
    def __init__(self):
        self.ts_std1 = ts_std(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_std1(div(get('sell_small_order_up_fx'),get('amount_trade_fx')))


class alpha_vboth_fx_460:
    time_consume=6
    def __init__(self):
        self.ts_tcv1 = ts_tcv(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_tcv1(div_cs(get('buy_small_order_up_fx'),get('amount_trade_fx')))


class alpha_vboth_fx_462:
    time_consume=8
    def __init__(self):
        self.ts_tcv1 = ts_tcv(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_tcv1(self.ts_sum_div_cs1(get('sell_small_order_down_fx'),get('amount_trade_fx')))


class alpha_vboth_fx_469:
    time_consume=3
    def __init__(self):
        self.ts_avg1 = ts_avg(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_avg1(div(self.ts_sum_div1(get('small_actS_up_fx'),get('small_actS_cnt_up_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_480:
    time_consume=26
    def __init__(self):
        self.ts_min1 = ts_min(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div2 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_sum_div_cs1(self.ts_sum_div1(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),self.ts_sum_div2(get('buy_big_order_down_fx'),get('buy_big_order_down_cnt_fx'))))


class alpha_vboth_fx_484:
    time_consume=5
    def __init__(self):
        self.ts_std1 = ts_std(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_std1(div_cs(div(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),div(add(get('sell_small_order_up_fx'),get('sell_big_order_up_fx'),get('sell_small_order_down_fx'),get('sell_big_order_down_fx')),add(get('sell_small_order_up_cnt_fx'),get('sell_big_order_up_cnt_fx'),get('sell_small_order_down_cnt_fx'),get('sell_big_order_down_cnt_fx')))))


class alpha_vboth_fx_486:
    time_consume=8
    def __init__(self):
        self.ts_cv_abs1 = ts_cv_abs(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_cv_abs1(div_cs(self.ts_sum_div1(get('small_actB_down_fx'),get('small_actB_cnt_down_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_500:
    time_consume=5
    def __init__(self):
        self.ts_std1 = ts_std(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_std1(div_cs(div(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),div(add(get('sell_small_order_up_fx'),get('sell_big_order_up_fx'),get('sell_small_order_down_fx'),get('sell_big_order_down_fx')),add(get('sell_small_order_up_cnt_fx'),get('sell_big_order_up_cnt_fx'),get('sell_small_order_down_cnt_fx'),get('sell_big_order_down_cnt_fx')))))


class alpha_vboth_fx_501:
    time_consume=6
    def __init__(self):
        self.ts_cv_abs1 = ts_cv_abs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_cv_abs1(div_cs(div(get('buy_small_order_down_fx'),get('buy_small_order_down_cnt_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_506:
    time_consume=6
    def __init__(self):
        self.ts_cv_abs1 = ts_cv_abs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_cv_abs1(div_cs(div(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),div(add(get('buy_small_order_up_fx'),get('buy_big_order_up_fx'),get('buy_small_order_down_fx'),get('buy_big_order_down_fx')),add(get('buy_small_order_up_cnt_fx'),get('buy_big_order_up_cnt_fx'),get('buy_small_order_down_cnt_fx'),get('buy_big_order_down_cnt_fx')))))


class alpha_vboth_fx_512:
    time_consume=5
    def __init__(self):
        self.ts_std1 = ts_std(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_std1(div_cs(get('buy_small_order_up_fx'),get('amount_trade_fx')))


class alpha_vboth_fx_513:
    time_consume=5
    def __init__(self):
        self.ts_std1 = ts_std(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_std1(div(get('sell_small_order_down_fx'),get('amount_trade_fx')))


class alpha_vboth_fx_517:
    time_consume=22
    def __init__(self):
        self.ts_min1 = ts_min(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_sum_div1(div(get('sell_small_order_down_fx'),get('sell_small_order_down_cnt_fx')),div(get('sell_small_order_up_fx'),get('sell_small_order_up_cnt_fx'))))


class alpha_vboth_fx_521:
    time_consume=22
    def __init__(self):
        self.ts_min1 = ts_min(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_sum_div1(div(add(get('sell_small_order_down_fx'),get('sell_big_order_down_fx')),add(get('sell_small_order_down_cnt_fx'),get('sell_big_order_down_cnt_fx'))),div(add(get('sell_small_order_up_fx'),get('sell_big_order_up_fx')),add(get('sell_small_order_up_cnt_fx'),get('sell_big_order_up_cnt_fx')))))


class alpha_vboth_fx_552:
    time_consume=6
    def __init__(self):
        self.ts_cv_abs1 = ts_cv_abs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_cv_abs1(div_cs(get('sell_small_order_up_fx'),get('amount_trade_fx')))


class alpha_vboth_fx_553:
    time_consume=6
    def __init__(self):
        self.ts_tcv1 = ts_tcv(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_tcv1(div_cs(get('sell_small_order_up_fx'),get('amount_trade_fx')))


class alpha_vboth_fx_555:
    time_consume=6
    def __init__(self):
        self.ts_cv_abs1 = ts_cv_abs(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_cv_abs1(div(div(get('sell_small_order_down_fx'),get('sell_small_order_down_cnt_fx')),div(add(get('sell_small_order_up_fx'),get('sell_big_order_up_fx'),get('sell_small_order_down_fx'),get('sell_big_order_down_fx')),add(get('sell_small_order_up_cnt_fx'),get('sell_big_order_up_cnt_fx'),get('sell_small_order_down_cnt_fx'),get('sell_big_order_down_cnt_fx')))))


class alpha_vboth_fx_559:
    time_consume=3
    def __init__(self):
        self.ts_avg1 = ts_avg(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_avg1(div(self.ts_sum_div1(get('small_actS_down_fx'),get('small_actS_cnt_down_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_560:
    time_consume=5
    def __init__(self):
        self.ts_std1 = ts_std(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_std1(div_cs(get('buy_small_order_down_fx'),get('amount_trade_fx')))


class alpha_vboth_fx_566:
    time_consume=7
    def __init__(self):
        self.ts_corr1 = ts_corr(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_corr1(div(self.ts_sum_div1(get('buy_small_order_down_fx'),get('buy_small_order_down_cnt_fx')),get('amount_trade_fx')),mul(get('close_trade_fx'),get('fq_factor_trade_fx')))


class alpha_vboth_fx_567:
    time_consume=5
    def __init__(self):
        self.ts_std1 = ts_std(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_std1(div(get('buy_small_order_down_fx'),get('amount_trade_fx')))


class alpha_vboth_fx_569:
    time_consume=2
    def __init__(self):
        self.ts_sum_div_cs1 = ts_sum_div_cs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_sum_div_cs1(div(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),div(get('sell_small_order_up_fx'),get('sell_small_order_up_cnt_fx')))


class alpha_vboth_fx_572:
    time_consume=6
    def __init__(self):
        self.ts_tcv1 = ts_tcv(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_tcv1(div_cs(div(get('buy_big_order_down_fx'),get('buy_big_order_down_cnt_fx')),div(get('buy_big_order_up_fx'),get('buy_big_order_up_cnt_fx'))))


class alpha_vboth_fx_573:
    time_consume=5
    def __init__(self):
        self.ts_corr1 = ts_corr(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_corr1(div(div(get('sell_small_order_down_fx'),get('sell_small_order_down_cnt_fx')),get('amount_trade_fx')),mul(get('close_trade_fx'),get('fq_factor_trade_fx')))


class alpha_vboth_fx_574:
    time_consume=6
    def __init__(self):
        self.ts_tcv1 = ts_tcv(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_tcv1(div_cs(get('buy_big_order_down_fx'),get('amount_trade_fx')))


class alpha_vboth_fx_575:
    time_consume=7
    def __init__(self):
        self.ts_std1 = ts_std(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_std1(self.ts_sum_div_cs1(get('buy_small_order_act_up_fx'),add(get('buy_small_order_up_fx'),get('buy_big_order_up_fx'),get('buy_small_order_down_fx'),get('buy_big_order_down_fx'))))


class alpha_vboth_fx_582:
    time_consume=22
    def __init__(self):
        self.ts_min1 = ts_min(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_sum_div_cs1(div(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),div(get('sell_small_order_up_fx'),get('sell_small_order_up_cnt_fx'))))


class alpha_vboth_fx_584:
    time_consume=5
    def __init__(self):
        self.ts_std1 = ts_std(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_std1(div_cs(get('sell_small_order_up_fx'),get('amount_trade_fx')))


class alpha_vboth_fx_587:
    time_consume=6
    def __init__(self):
        self.ts_corr1 = ts_corr(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_growth1 = ts_growth(1,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_corr1(div(get('buy_small_order_tot_fx'),get('amount_trade_fx')),self.ts_growth1(mul(get('close_trade_fx'),get('fq_factor_trade_fx'))))


class alpha_vboth_fx_593:
    time_consume=6
    def __init__(self):
        self.ts_tcv1 = ts_tcv(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_tcv1(div_cs(div(get('buy_small_order_down_fx'),get('buy_small_order_down_cnt_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_595:
    time_consume=6
    def __init__(self):
        self.ts_cv_abs1 = ts_cv_abs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_cv_abs1(div_cs(div(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_598:
    time_consume=7
    def __init__(self):
        self.ts_std1 = ts_std(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_std1(self.ts_sum_div_cs1(get('buy_small_order_act_up_fx'),get('buy_small_order_down_fx')))


class alpha_vboth_fx_607:
    time_consume=22
    def __init__(self):
        self.ts_min1 = ts_min(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_sum_div_cs1(div(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),div(get('buy_big_order_up_fx'),get('buy_big_order_up_cnt_fx'))))


class alpha_vboth_fx_610:
    time_consume=8
    def __init__(self):
        self.ts_tcv1 = ts_tcv(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_tcv1(self.ts_sum_div_cs1(div(add(get('sell_small_order_down_fx'),get('sell_big_order_down_fx')),add(get('sell_small_order_down_cnt_fx'),get('sell_big_order_down_cnt_fx'))),div(add(get('sell_small_order_up_fx'),get('sell_big_order_up_fx')),add(get('sell_small_order_up_cnt_fx'),get('sell_big_order_up_cnt_fx')))))


class alpha_vboth_fx_612:
    time_consume=8
    def __init__(self):
        self.ts_tcv1 = ts_tcv(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_tcv1(self.ts_sum_div_cs1(div(add(get('sell_small_order_down_fx'),get('sell_big_order_down_fx')),add(get('sell_small_order_down_cnt_fx'),get('sell_big_order_down_cnt_fx'))),div(add(get('sell_small_order_up_fx'),get('sell_big_order_up_fx')),add(get('sell_small_order_up_cnt_fx'),get('sell_big_order_up_cnt_fx')))))


class alpha_vboth_fx_614:
    time_consume=8
    def __init__(self):
        self.ts_cv_abs1 = ts_cv_abs(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_cv_abs1(div_cs(self.ts_sum_div1(get('small_actS_up_fx'),get('small_actS_cnt_up_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_617:
    time_consume=8
    def __init__(self):
        self.ts_cv_abs1 = ts_cv_abs(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_cv_abs1(div_cs(self.ts_sum_div1(get('small_actS_down_fx'),get('small_actS_cnt_down_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_620:
    time_consume=5
    def __init__(self):
        self.ts_std1 = ts_std(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_std1(div_cs(div(get('sell_small_order_down_fx'),get('sell_small_order_down_cnt_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_624:
    time_consume=21
    def __init__(self):
        self.ts_max1 = ts_max(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_avg1 = ts_avg(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_max1(self.ts_avg1(div_cs(get('buy_small_order_act_tot_fx'),add(get('sell_small_order_act_tot_fx'),get('sell_big_order_act_tot_fx')))))


class alpha_vboth_fx_628:
    time_consume=8
    def __init__(self):
        self.ts_cv_abs1 = ts_cv_abs(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_cv_abs1(div_cs(self.ts_sum_div1(get('small_actS_down_fx'),get('small_actS_cnt_down_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_630:
    time_consume=5
    def __init__(self):
        self.ts_std1 = ts_std(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_std1(div(get('buy_small_order_up_fx'),get('amount_trade_fx')))


class alpha_vboth_fx_632:
    time_consume=7
    def __init__(self):
        self.ts_std1 = ts_std(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_std1(self.ts_sum_div_cs1(get('sell_small_order_down_fx'),get('amount_trade_fx')))


class alpha_vboth_fx_635:
    time_consume=22
    def __init__(self):
        self.ts_min1 = ts_min(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_sum_div1(div(add(get('buy_small_order_up_fx'),get('buy_big_order_up_fx')),add(get('buy_small_order_up_cnt_fx'),get('buy_big_order_up_cnt_fx'))),div(add(get('buy_small_order_down_fx'),get('buy_big_order_down_fx')),add(get('buy_small_order_down_cnt_fx'),get('buy_big_order_down_cnt_fx')))))


class alpha_vboth_fx_638:
    time_consume=2
    def __init__(self):
        self.ts_sum_div1 = ts_sum_div(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_sum_div1(div(add(get('sell_small_order_down_fx'),get('sell_big_order_down_fx')),add(get('sell_small_order_down_cnt_fx'),get('sell_big_order_down_cnt_fx'))),div(add(get('sell_small_order_up_fx'),get('sell_big_order_up_fx')),add(get('sell_small_order_up_cnt_fx'),get('sell_big_order_up_cnt_fx'))))


class alpha_vboth_fx_642:
    time_consume=5
    def __init__(self):
        self.ts_skew1 = ts_skew(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_skew1(div(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')))


class alpha_vboth_fx_644:
    time_consume=23
    def __init__(self):
        self.ts_min1 = ts_min(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_avg1 = ts_avg(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_avg1(div(self.ts_sum_div1(get('buy_small_order_down_fx'),get('buy_small_order_down_cnt_fx')),get('amount_trade_fx'))))


class alpha_vboth_fx_654:
    time_consume=21
    def __init__(self):
        self.ts_max1 = ts_max(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_avg1 = ts_avg(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_max1(self.ts_avg1(div(get('buy_small_order_act_up_fx'),get('amount_trade_fx'))))


class alpha_vboth_fx_659:
    time_consume=8
    def __init__(self):
        self.ts_cv_abs1 = ts_cv_abs(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_cv_abs1(div_cs(self.ts_sum_div1(get('small_actB_up_fx'),get('small_actB_cnt_up_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_661:
    time_consume=26
    def __init__(self):
        self.ts_min1 = ts_min(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div2 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_sum_div_cs1(self.ts_sum_div1(get('sell_small_order_up_fx'),get('sell_small_order_up_cnt_fx')),self.ts_sum_div2(get('sell_big_order_up_fx'),get('sell_big_order_up_cnt_fx'))))


class alpha_vboth_fx_665:
    time_consume=8
    def __init__(self):
        self.ts_cv_abs1 = ts_cv_abs(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_cv_abs1(div_cs(self.ts_sum_div1(get('small_actB_down_fx'),get('small_actB_cnt_down_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_673:
    time_consume=6
    def __init__(self):
        self.ts_cv_abs1 = ts_cv_abs(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_cv_abs1(div(div(get('sell_small_order_up_fx'),get('sell_small_order_up_cnt_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_689:
    time_consume=7
    def __init__(self):
        self.ts_std1 = ts_std(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_std1(self.ts_sum_div_cs1(div(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),div(add(get('sell_small_order_up_fx'),get('sell_big_order_up_fx'),get('sell_small_order_down_fx'),get('sell_big_order_down_fx')),add(get('sell_small_order_up_cnt_fx'),get('sell_big_order_up_cnt_fx'),get('sell_small_order_down_cnt_fx'),get('sell_big_order_down_cnt_fx')))))


class alpha_vboth_fx_693:
    time_consume=22
    def __init__(self):
        self.ts_min1 = ts_min(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(div_cs(self.ts_sum_div1(get('buy_small_order_down_fx'),get('buy_small_order_down_cnt_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_696:
    time_consume=8
    def __init__(self):
        self.ts_cv_abs1 = ts_cv_abs(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_cv_abs1(div(self.ts_sum_div1(get('small_actB_down_fx'),get('small_actB_cnt_down_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_700:
    time_consume=22
    def __init__(self):
        self.ts_min1 = ts_min(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_sum_div1(div(get('sell_small_order_down_fx'),get('sell_small_order_down_cnt_fx')),div(get('sell_small_order_up_fx'),get('sell_small_order_up_cnt_fx'))))


class alpha_vboth_fx_711:
    time_consume=7
    def __init__(self):
        self.ts_std1 = ts_std(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_std1(self.ts_sum_div_cs1(div(add(get('buy_small_order_up_fx'),get('buy_big_order_up_fx'),get('buy_small_order_down_fx'),get('buy_big_order_down_fx')),add(get('buy_small_order_up_cnt_fx'),get('buy_big_order_up_cnt_fx'),get('buy_small_order_down_cnt_fx'),get('buy_big_order_down_cnt_fx'))),div(add(get('sell_small_order_up_fx'),get('sell_big_order_up_fx'),get('sell_small_order_down_fx'),get('sell_big_order_down_fx')),add(get('sell_small_order_up_cnt_fx'),get('sell_big_order_up_cnt_fx'),get('sell_small_order_down_cnt_fx'),get('sell_big_order_down_cnt_fx')))))


class alpha_vboth_fx_712:
    time_consume=5
    def __init__(self):
        self.ts_std1 = ts_std(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_std1(div_cs(get('sell_small_order_act_tot_fx'),get('amount_trade_fx')))


class alpha_vboth_fx_714:
    time_consume=8
    def __init__(self):
        self.ts_cv_abs1 = ts_cv_abs(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_cv_abs1(div_cs(self.ts_sum_div1(get('small_actB_up_fx'),get('small_actB_cnt_up_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_723:
    time_consume=5
    def __init__(self):
        self.ts_std1 = ts_std(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_std1(div(get('sell_small_order_up_fx'),get('amount_trade_fx')))


class alpha_vboth_fx_733:
    time_consume=6
    def __init__(self):
        self.ts_tcv1 = ts_tcv(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_tcv1(div_cs(get('buy_small_order_act_tot_fx'),add(get('sell_small_order_act_tot_fx'),get('sell_big_order_act_tot_fx'))))


class alpha_vboth_fx_734:
    time_consume=3
    def __init__(self):
        self.ts_avg1 = ts_avg(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_avg1(self.ts_sum_div_cs1(div(get('sell_small_order_down_fx'),get('sell_small_order_down_cnt_fx')),div(get('sell_small_order_up_fx'),get('sell_small_order_up_cnt_fx'))))


class alpha_vboth_fx_744:
    time_consume=22
    def __init__(self):
        self.ts_min1 = ts_min(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_sum_div_cs1(div(add(get('buy_small_order_up_fx'),get('buy_big_order_up_fx')),add(get('buy_small_order_up_cnt_fx'),get('buy_big_order_up_cnt_fx'))),div(add(get('buy_small_order_down_fx'),get('buy_big_order_down_fx')),add(get('buy_small_order_down_cnt_fx'),get('buy_big_order_down_cnt_fx')))))


class alpha_vboth_fx_746:
    time_consume=22
    def __init__(self):
        self.ts_min1 = ts_min(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_sum_div1(div(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),div(get('buy_small_order_down_fx'),get('buy_small_order_down_cnt_fx'))))


class alpha_vboth_fx_754:
    time_consume=6
    def __init__(self):
        self.ts_tcv1 = ts_tcv(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_tcv1(div_cs(add(get('buy_small_order_act_down_fx'),get('buy_big_order_act_down_fx')),add(get('buy_small_order_up_fx'),get('buy_big_order_up_fx'))))


class alpha_vboth_fx_757:
    time_consume=8
    def __init__(self):
        self.ts_cv_abs1 = ts_cv_abs(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_cv_abs1(div_cs(self.ts_sum_div1(get('small_actS_equi_fx'),get('small_actS_cnt_equi_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_758:
    time_consume=21
    def __init__(self):
        self.ts_min1 = ts_min(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_avg1 = ts_avg(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_avg1(div_cs(div(get('buy_big_order_down_fx'),get('buy_big_order_down_cnt_fx')),div(get('sell_big_order_down_fx'),get('sell_big_order_down_cnt_fx')))))


class alpha_vboth_fx_764:
    time_consume=42
    def __init__(self):
        self.ts_range1 = ts_range(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_range1(self.ts_sum_div_cs1(div(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),div(add(get('buy_small_order_up_fx'),get('buy_big_order_up_fx'),get('buy_small_order_down_fx'),get('buy_big_order_down_fx')),add(get('buy_small_order_up_cnt_fx'),get('buy_big_order_up_cnt_fx'),get('buy_small_order_down_cnt_fx'),get('buy_big_order_down_cnt_fx')))))


class alpha_vboth_fx_766:
    time_consume=42
    def __init__(self):
        self.ts_range1 = ts_range(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_range1(self.ts_sum_div_cs1(div(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),div(add(get('buy_small_order_up_fx'),get('buy_big_order_up_fx'),get('buy_small_order_down_fx'),get('buy_big_order_down_fx')),add(get('buy_small_order_up_cnt_fx'),get('buy_big_order_up_cnt_fx'),get('buy_small_order_down_cnt_fx'),get('buy_big_order_down_cnt_fx')))))


class alpha_vboth_fx_770:
    time_consume=8
    def __init__(self):
        self.ts_cv_abs1 = ts_cv_abs(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_cv_abs1(div_cs(self.ts_sum_div1(get('small_actS_equi_fx'),get('small_actS_cnt_equi_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_771:
    time_consume=5
    def __init__(self):
        self.ts_std1 = ts_std(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_std1(div(div(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),div(add(get('buy_small_order_up_fx'),get('buy_big_order_up_fx')),add(get('buy_small_order_up_cnt_fx'),get('buy_big_order_up_cnt_fx')))))


class alpha_vboth_fx_784:
    time_consume=8
    def __init__(self):
        self.ts_cv_abs1 = ts_cv_abs(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_cv_abs1(div_cs(self.ts_sum_div1(get('small_actS_up_fx'),get('small_actS_cnt_up_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_787:
    time_consume=3
    def __init__(self):
        self.ts_avg1 = ts_avg(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_avg1(div(self.ts_sum_div1(get('small_actS_up_fx'),get('small_actS_cnt_up_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_790:
    time_consume=9
    def __init__(self):
        self.ts_std1 = ts_std(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div2 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_std1(div(self.ts_sum_div1(get('sell_small_order_down_fx'),get('sell_small_order_down_cnt_fx')),self.ts_sum_div2(get('sell_small_order_up_fx'),get('sell_small_order_up_cnt_fx'))))


class alpha_vboth_fx_801:
    time_consume=22
    def __init__(self):
        self.ts_min1 = ts_min(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_sum_div_cs1(div(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),div(get('buy_big_order_up_fx'),get('buy_big_order_up_cnt_fx'))))


class alpha_vboth_fx_809:
    time_consume=8
    def __init__(self):
        self.ts_tcv1 = ts_tcv(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_tcv1(self.ts_sum_div_cs1(div(get('sell_small_order_down_fx'),get('sell_small_order_down_cnt_fx')),div(get('sell_small_order_up_fx'),get('sell_small_order_up_cnt_fx'))))


class alpha_vboth_fx_815:
    time_consume=6
    def __init__(self):
        self.ts_cv_abs1 = ts_cv_abs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_cv_abs1(div(get('sell_small_order_act_tot_fx'),get('amount_trade_fx')))


class alpha_vboth_fx_816:
    time_consume=5
    def __init__(self):
        self.ts_std1 = ts_std(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_std1(div_cs(add(get('buy_small_order_act_up_fx'),get('buy_big_order_act_up_fx'),get('buy_small_order_act_down_fx'),get('buy_big_order_act_down_fx')),add(get('buy_small_order_up_fx'),get('buy_big_order_up_fx'),get('buy_small_order_down_fx'),get('buy_big_order_down_fx'))))


class alpha_vboth_fx_820:
    time_consume=8
    def __init__(self):
        self.ts_tcv1 = ts_tcv(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_tcv1(self.ts_sum_div_cs1(div(get('sell_small_order_down_fx'),get('sell_small_order_down_cnt_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_830:
    time_consume=42
    def __init__(self):
        self.ts_range1 = ts_range(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_range1(self.ts_sum_div_cs1(div(get('sell_small_order_down_fx'),get('sell_small_order_down_cnt_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_837:
    time_consume=7
    def __init__(self):
        self.ts_std1 = ts_std(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_std1(div_cs(self.ts_sum_div1(get('sell_big_order_down_fx'),get('sell_big_order_down_cnt_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_838:
    time_consume=5
    def __init__(self):
        self.ts_std1 = ts_std(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_std1(div_cs(div(get('sell_small_order_down_fx'),get('sell_small_order_down_cnt_fx')),div(add(get('sell_small_order_up_fx'),get('sell_big_order_up_fx'),get('sell_small_order_down_fx'),get('sell_big_order_down_fx')),add(get('sell_small_order_up_cnt_fx'),get('sell_big_order_up_cnt_fx'),get('sell_small_order_down_cnt_fx'),get('sell_big_order_down_cnt_fx')))))


class alpha_vboth_fx_858:
    time_consume=6
    def __init__(self):
        self.ts_cv_abs1 = ts_cv_abs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_cv_abs1(div(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')))


class alpha_vboth_fx_865:
    time_consume=8
    def __init__(self):
        self.ts_cv_abs1 = ts_cv_abs(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_cv_abs1(div_cs(self.ts_sum_div1(get('small_actB_equi_fx'),get('small_actB_cnt_equi_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_873:
    time_consume=23
    def __init__(self):
        self.ts_min1 = ts_min(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_avg1 = ts_avg(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_avg1(div(self.ts_sum_div1(get('sell_small_order_up_fx'),get('sell_small_order_up_cnt_fx')),get('amount_trade_fx'))))


class alpha_vboth_fx_884:
    time_consume=8
    def __init__(self):
        self.ts_cv_abs1 = ts_cv_abs(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_cv_abs1(self.ts_sum_div1(get('small_actB_up_fx'),get('small_actB_cnt_up_fx')))


class alpha_vboth_fx_896:
    time_consume=23
    def __init__(self):
        self.ts_max1 = ts_max(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_avg1 = ts_avg(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_max1(self.ts_avg1(div(self.ts_sum_div1(get('small_actS_down_fx'),get('small_actS_cnt_down_fx')),get('amount_trade_fx'))))


class alpha_vboth_fx_898:
    time_consume=6
    def __init__(self):
        self.ts_cv_abs1 = ts_cv_abs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_cv_abs1(div_cs(div(get('sell_small_order_up_fx'),get('sell_small_order_up_cnt_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_899:
    time_consume=8
    def __init__(self):
        self.ts_cv_abs1 = ts_cv_abs(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_cv_abs1(self.ts_sum_div_cs1(div(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_901:
    time_consume=23
    def __init__(self):
        self.ts_min1 = ts_min(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_avg1 = ts_avg(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_avg1(div_cs(self.ts_sum_div1(get('buy_small_order_down_fx'),get('buy_small_order_down_cnt_fx')),get('amount_trade_fx'))))


class alpha_vboth_fx_902:
    time_consume=8
    def __init__(self):
        self.ts_tcv1 = ts_tcv(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_tcv1(self.ts_sum_div_cs1(div(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),div(get('buy_big_order_up_fx'),get('buy_big_order_up_cnt_fx'))))


class alpha_vboth_fx_903:
    time_consume=8
    def __init__(self):
        self.ts_tcv1 = ts_tcv(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_tcv1(self.ts_sum_div_cs1(div(add(get('sell_small_order_down_fx'),get('sell_big_order_down_fx')),add(get('sell_small_order_down_cnt_fx'),get('sell_big_order_down_cnt_fx'))),div(add(get('sell_small_order_up_fx'),get('sell_big_order_up_fx')),add(get('sell_small_order_up_cnt_fx'),get('sell_big_order_up_cnt_fx')))))


class alpha_vboth_fx_910:
    time_consume=8
    def __init__(self):
        self.ts_tcv1 = ts_tcv(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_tcv1(div_cs(self.ts_sum_div1(get('small_actB_down_fx'),get('small_actB_cnt_down_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_917:
    time_consume=7
    def __init__(self):
        self.ts_std1 = ts_std(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_std1(self.ts_sum_div1(div(add(get('buy_small_order_up_fx'),get('buy_big_order_up_fx'),get('buy_small_order_down_fx'),get('buy_big_order_down_fx')),add(get('buy_small_order_up_cnt_fx'),get('buy_big_order_up_cnt_fx'),get('buy_small_order_down_cnt_fx'),get('buy_big_order_down_cnt_fx'))),div(add(get('sell_small_order_up_fx'),get('sell_big_order_up_fx'),get('sell_small_order_down_fx'),get('sell_big_order_down_fx')),add(get('sell_small_order_up_cnt_fx'),get('sell_big_order_up_cnt_fx'),get('sell_small_order_down_cnt_fx'),get('sell_big_order_down_cnt_fx')))))


class alpha_vboth_fx_921:
    time_consume=5
    def __init__(self):
        self.ts_std1 = ts_std(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_std1(div_cs(div(add(get('buy_small_order_up_fx'),get('buy_big_order_up_fx'),get('buy_small_order_down_fx'),get('buy_big_order_down_fx')),add(get('buy_small_order_up_cnt_fx'),get('buy_big_order_up_cnt_fx'),get('buy_small_order_down_cnt_fx'),get('buy_big_order_down_cnt_fx'))),div(add(get('sell_small_order_up_fx'),get('sell_big_order_up_fx'),get('sell_small_order_down_fx'),get('sell_big_order_down_fx')),add(get('sell_small_order_up_cnt_fx'),get('sell_big_order_up_cnt_fx'),get('sell_small_order_down_cnt_fx'),get('sell_big_order_down_cnt_fx')))))


class alpha_vboth_fx_926:
    time_consume=8
    def __init__(self):
        self.ts_cv_abs1 = ts_cv_abs(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_cv_abs1(div_cs(self.ts_sum_div1(get('small_actB_equi_fx'),get('small_actB_cnt_equi_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_937:
    time_consume=6
    def __init__(self):
        self.ts_cv_abs1 = ts_cv_abs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_cv_abs1(div_cs(div(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),div(add(get('sell_small_order_up_fx'),get('sell_big_order_up_fx'),get('sell_small_order_down_fx'),get('sell_big_order_down_fx')),add(get('sell_small_order_up_cnt_fx'),get('sell_big_order_up_cnt_fx'),get('sell_small_order_down_cnt_fx'),get('sell_big_order_down_cnt_fx')))))


class alpha_vboth_fx_941:
    time_consume=23
    def __init__(self):
        self.ts_max1 = ts_max(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_avg1 = ts_avg(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_max1(self.ts_avg1(div_cs(self.ts_sum_div1(get('small_actB_down_fx'),get('small_actB_cnt_down_fx')),get('amount_trade_fx'))))


class alpha_vboth_fx_944:
    time_consume=44
    def __init__(self):
        self.ts_range1 = ts_range(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div2 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_range1(div_cs(self.ts_sum_div1(get('sell_small_order_up_fx'),get('sell_small_order_up_cnt_fx')),self.ts_sum_div2(add(get('sell_small_order_down_fx'),get('sell_big_order_down_fx')),add(get('sell_small_order_down_cnt_fx'),get('sell_big_order_down_cnt_fx')))))


class alpha_vboth_fx_949:
    time_consume=5
    def __init__(self):
        self.ts_skew1 = ts_skew(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_skew1(div(div(get('small_actB_up_fx'),get('small_actB_cnt_up_fx')),get('circulating_market_cap')))


class alpha_vboth_fx_959:
    time_consume=6
    def __init__(self):
        self.ts_tcv1 = ts_tcv(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_tcv1(div_cs(div(get('sell_small_order_up_fx'),get('sell_small_order_up_cnt_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_975:
    time_consume=5
    def __init__(self):
        self.ts_std1 = ts_std(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_std1(div_cs(get('sell_small_order_act_tot_fx'),add(get('buy_small_order_act_tot_fx'),get('buy_big_order_act_tot_fx'))))


class alpha_vboth_fx_977:
    time_consume=42
    def __init__(self):
        self.ts_range1 = ts_range(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_range1(self.ts_sum_div1(div(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),div(add(get('sell_small_order_up_fx'),get('sell_big_order_up_fx'),get('sell_small_order_down_fx'),get('sell_big_order_down_fx')),add(get('sell_small_order_up_cnt_fx'),get('sell_big_order_up_cnt_fx'),get('sell_small_order_down_cnt_fx'),get('sell_big_order_down_cnt_fx')))))


class alpha_vboth_fx_984:
    time_consume=5
    def __init__(self):
        self.ts_std1 = ts_std(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_std1(div(add(get('buy_small_order_act_up_fx'),get('buy_big_order_act_up_fx'),get('buy_small_order_act_down_fx'),get('buy_big_order_act_down_fx')),add(get('buy_small_order_up_fx'),get('buy_big_order_up_fx'),get('buy_small_order_down_fx'),get('buy_big_order_down_fx'))))


class alpha_vboth_fx_986:
    time_consume=44
    def __init__(self):
        self.ts_range1 = ts_range(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div2 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_range1(div_cs(self.ts_sum_div1(add(get('buy_small_order_up_fx'),get('buy_big_order_up_fx')),add(get('buy_small_order_up_cnt_fx'),get('buy_big_order_up_cnt_fx'))),self.ts_sum_div2(add(get('buy_small_order_up_fx'),get('buy_big_order_up_fx'),get('buy_small_order_down_fx'),get('buy_big_order_down_fx')),add(get('buy_small_order_up_cnt_fx'),get('buy_big_order_up_cnt_fx'),get('buy_small_order_down_cnt_fx'),get('buy_big_order_down_cnt_fx')))))


class alpha_vboth_fx_990:
    time_consume=21
    def __init__(self):
        self.ts_max1 = ts_max(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_avg1 = ts_avg(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_max1(self.ts_avg1(div_cs(add(get('buy_small_order_tot_fx'),get('buy_big_order_tot_fx')),add(get('sell_small_order_act_tot_fx'),get('sell_big_order_act_tot_fx')))))


class alpha_vboth_fx_996:
    time_consume=8
    def __init__(self):
        self.ts_cv_abs1 = ts_cv_abs(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_cv_abs1(self.ts_sum_div1(get('sell_small_order_down_fx'),get('sell_small_order_down_cnt_fx')))


class alpha_vboth_fx_1003:
    time_consume=9
    def __init__(self):
        self.ts_std1 = ts_std(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div2 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_std1(div_cs(self.ts_sum_div1(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),self.ts_sum_div2(get('sell_small_order_down_fx'),get('sell_small_order_down_cnt_fx'))))


class alpha_vboth_fx_1020:
    time_consume=5
    def __init__(self):
        self.ts_std1 = ts_std(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_std1(div_cs(div(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),div(add(get('buy_small_order_up_fx'),get('buy_big_order_up_fx'),get('buy_small_order_down_fx'),get('buy_big_order_down_fx')),add(get('buy_small_order_up_cnt_fx'),get('buy_big_order_up_cnt_fx'),get('buy_small_order_down_cnt_fx'),get('buy_big_order_down_cnt_fx')))))


class alpha_vboth_fx_1022:
    time_consume=9
    def __init__(self):
        self.ts_std1 = ts_std(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div2 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_std1(div_cs(self.ts_sum_div1(add(get('buy_small_order_up_fx'),get('buy_big_order_up_fx')),add(get('buy_small_order_up_cnt_fx'),get('buy_big_order_up_cnt_fx'))),self.ts_sum_div2(add(get('sell_small_order_down_fx'),get('sell_big_order_down_fx')),add(get('sell_small_order_down_cnt_fx'),get('sell_big_order_down_cnt_fx')))))


class alpha_vboth_fx_1023:
    time_consume=24
    def __init__(self):
        self.ts_max1 = ts_max(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div2 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_max1(div(self.ts_sum_div1(get('sell_small_order_down_fx'),get('sell_small_order_down_cnt_fx')),self.ts_sum_div2(add(get('sell_small_order_up_fx'),get('sell_big_order_up_fx'),get('sell_small_order_down_fx'),get('sell_big_order_down_fx')),add(get('sell_small_order_up_cnt_fx'),get('sell_big_order_up_cnt_fx'),get('sell_small_order_down_cnt_fx'),get('sell_big_order_down_cnt_fx')))))


class alpha_vboth_fx_1024:
    time_consume=5
    def __init__(self):
        self.ts_std1 = ts_std(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_std1(div(get('buy_small_order_act_tot_fx'),add(get('buy_small_order_tot_fx'),get('buy_big_order_tot_fx'))))


class alpha_vboth_fx_1034:
    time_consume=5
    def __init__(self):
        self.ts_std1 = ts_std(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_std1(div_cs(get('buy_small_order_act_tot_fx'),get('sell_small_order_act_tot_fx')))


class alpha_vboth_fx_1051:
    time_consume=22
    def __init__(self):
        self.ts_min1 = ts_min(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_sum_div_cs1(div(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),div(get('sell_big_order_up_fx'),get('sell_big_order_up_cnt_fx'))))


class alpha_vboth_fx_1059:
    time_consume=6
    def __init__(self):
        self.ts_tcv1 = ts_tcv(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_tcv1(div_cs(get('buy_big_order_act_up_fx'),get('amount_trade_fx')))


class alpha_vboth_fx_1064:
    time_consume=6
    def __init__(self):
        self.ts_tcv1 = ts_tcv(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_tcv1(div_cs(div(get('buy_big_order_down_fx'),get('buy_big_order_down_cnt_fx')),div(get('sell_big_order_down_fx'),get('sell_big_order_down_cnt_fx'))))


class alpha_vboth_fx_1066:
    time_consume=21
    def __init__(self):
        self.ts_min1 = ts_min(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_avg1 = ts_avg(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_avg1(div(div(get('small_actB_equi_fx'),get('small_actB_cnt_equi_fx')),get('amount_trade_fx'))))


class alpha_vboth_fx_1074:
    time_consume=9
    def __init__(self):
        self.ts_std1 = ts_std(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div2 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_std1(div_cs(self.ts_sum_div1(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),self.ts_sum_div2(add(get('sell_small_order_down_fx'),get('sell_big_order_down_fx')),add(get('sell_small_order_down_cnt_fx'),get('sell_big_order_down_cnt_fx')))))


class alpha_vboth_fx_1076:
    time_consume=4
    def __init__(self):
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div2 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_sum_div1(self.ts_sum_div2(get('buy_small_order_down_fx'),get('buy_small_order_down_cnt_fx')),get('amount_trade_fx'))


class alpha_vboth_fx_1078:
    time_consume=6
    def __init__(self):
        self.ts_tcv1 = ts_tcv(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_tcv1(div_cs(get('buy_big_order_act_down_fx'),get('buy_big_order_up_fx')))


class alpha_vboth_fx_1089:
    time_consume=22
    def __init__(self):
        self.ts_min1 = ts_min(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_sum_div_cs1(div(get('sell_small_order_down_fx'),get('sell_small_order_down_cnt_fx')),div(get('sell_big_order_up_fx'),get('sell_big_order_up_cnt_fx'))))


class alpha_vboth_fx_1101:
    time_consume=22
    def __init__(self):
        self.ts_min1 = ts_min(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_sum_div_cs1(div(get('buy_small_order_down_fx'),get('buy_small_order_down_cnt_fx')),div(get('sell_big_order_up_fx'),get('sell_big_order_up_cnt_fx'))))


class alpha_vboth_fx_1102:
    time_consume=6
    def __init__(self):
        self.ts_cv_abs1 = ts_cv_abs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_cv_abs1(div(get('sell_small_order_down_fx'),get('sell_small_order_down_cnt_fx')))


class alpha_vboth_fx_1106:
    time_consume=6
    def __init__(self):
        self.ts_cv_abs1 = ts_cv_abs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_cv_abs1(div_cs(get('buy_small_order_act_tot_fx'),add(get('sell_small_order_tot_fx'),get('sell_big_order_tot_fx'))))


class alpha_vboth_fx_1114:
    time_consume=7
    def __init__(self):
        self.ts_std1 = ts_std(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_std1(div_cs(self.ts_sum_div1(get('buy_big_order_up_fx'),get('buy_big_order_up_cnt_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_1115:
    time_consume=8
    def __init__(self):
        self.ts_tcv1 = ts_tcv(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_tcv1(div_cs(self.ts_sum_div1(get('small_actS_up_fx'),get('small_actS_cnt_up_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_1118:
    time_consume=21
    def __init__(self):
        self.ts_min1 = ts_min(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_avg1 = ts_avg(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_avg1(div_cs(div(add(get('buy_small_order_up_fx'),get('buy_big_order_up_fx')),add(get('buy_small_order_up_cnt_fx'),get('buy_big_order_up_cnt_fx'))),div(add(get('sell_small_order_up_fx'),get('sell_big_order_up_fx')),add(get('sell_small_order_up_cnt_fx'),get('sell_big_order_up_cnt_fx'))))))


class alpha_vboth_fx_1124:
    time_consume=41
    def __init__(self):
        self.ts_range1 = ts_range(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_avg1 = ts_avg(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_range1(self.ts_avg1(div(get('buy_small_order_tot_fx'),get('sell_small_order_act_tot_fx'))))


class alpha_vboth_fx_1144:
    time_consume=23
    def __init__(self):
        self.ts_min1 = ts_min(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_avg1 = ts_avg(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_avg1(div_cs(self.ts_sum_div1(get('sell_small_order_up_fx'),get('sell_small_order_up_cnt_fx')),get('amount_trade_fx'))))


class alpha_vboth_fx_1145:
    time_consume=22
    def __init__(self):
        self.ts_max1 = ts_max(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_max1(self.ts_sum_div1(div(add(get('sell_small_order_down_fx'),get('sell_big_order_down_fx')),add(get('sell_small_order_down_cnt_fx'),get('sell_big_order_down_cnt_fx'))),div(add(get('sell_small_order_up_fx'),get('sell_big_order_up_fx')),add(get('sell_small_order_up_cnt_fx'),get('sell_big_order_up_cnt_fx')))))


class alpha_vboth_fx_1158:
    time_consume=22
    def __init__(self):
        self.ts_min1 = ts_min(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_sum_div1(div(add(get('buy_small_order_up_fx'),get('buy_big_order_up_fx')),add(get('buy_small_order_up_cnt_fx'),get('buy_big_order_up_cnt_fx'))),div(add(get('buy_small_order_down_fx'),get('buy_big_order_down_fx')),add(get('buy_small_order_down_cnt_fx'),get('buy_big_order_down_cnt_fx')))))


class alpha_vboth_fx_1171:
    time_consume=22
    def __init__(self):
        self.ts_max1 = ts_max(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_max1(self.ts_sum_div1(div(add(get('sell_small_order_down_fx'),get('sell_big_order_down_fx')),add(get('sell_small_order_down_cnt_fx'),get('sell_big_order_down_cnt_fx'))),div(add(get('sell_small_order_up_fx'),get('sell_big_order_up_fx')),add(get('sell_small_order_up_cnt_fx'),get('sell_big_order_up_cnt_fx')))))


class alpha_vboth_fx_1179:
    time_consume=10
    def __init__(self):
        self.ts_tcv1 = ts_tcv(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div2 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_tcv1(div_cs(self.ts_sum_div1(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),self.ts_sum_div2(get('buy_big_order_up_fx'),get('buy_big_order_up_cnt_fx'))))


class alpha_vboth_fx_1185:
    time_consume=6
    def __init__(self):
        self.ts_cv_abs1 = ts_cv_abs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_cv_abs1(div_cs(add(get('buy_small_order_act_tot_fx'),get('buy_big_order_act_tot_fx')),add(get('buy_small_order_tot_fx'),get('buy_big_order_tot_fx'))))


class alpha_vboth_fx_1188:
    time_consume=8
    def __init__(self):
        self.ts_tcv1 = ts_tcv(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_tcv1(self.ts_sum_div_cs1(div(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),div(get('buy_big_order_up_fx'),get('buy_big_order_up_cnt_fx'))))


class alpha_vboth_fx_1190:
    time_consume=21
    def __init__(self):
        self.ts_max1 = ts_max(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_avg1 = ts_avg(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_max1(self.ts_avg1(div_cs(get('buy_small_order_act_tot_fx'),add(get('sell_small_order_act_tot_fx'),get('sell_big_order_act_tot_fx')))))


class alpha_vboth_fx_1209:
    time_consume=22
    def __init__(self):
        self.ts_min1 = ts_min(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_sum_div1(div(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),div(get('buy_big_order_up_fx'),get('buy_big_order_up_cnt_fx'))))


class alpha_vboth_fx_1213:
    time_consume=9
    def __init__(self):
        self.ts_std1 = ts_std(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div2 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_std1(div_cs(self.ts_sum_div1(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),self.ts_sum_div2(get('sell_small_order_down_fx'),get('sell_small_order_down_cnt_fx'))))


class alpha_vboth_fx_1214:
    time_consume=10
    def __init__(self):
        self.ts_cv_abs1 = ts_cv_abs(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div2 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_cv_abs1(div(self.ts_sum_div1(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),self.ts_sum_div2(get('sell_small_order_down_fx'),get('sell_small_order_down_cnt_fx'))))


class alpha_vboth_fx_1218:
    time_consume=8
    def __init__(self):
        self.ts_cv_abs1 = ts_cv_abs(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_cv_abs1(div_cs(self.ts_sum_div1(get('buy_small_order_down_fx'),get('buy_small_order_down_cnt_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_1219:
    time_consume=45
    def __init__(self):
        self.ts_range1 = ts_range(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_avg1 = ts_avg(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div2 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_range1(self.ts_avg1(div_cs(self.ts_sum_div1(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),self.ts_sum_div2(get('sell_small_order_down_fx'),get('sell_small_order_down_cnt_fx')))))


class alpha_vboth_fx_1220:
    time_consume=5
    def __init__(self):
        self.ts_std1 = ts_std(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_std1(div_cs(get('buy_small_order_act_tot_fx'),add(get('sell_small_order_act_tot_fx'),get('sell_big_order_act_tot_fx'))))


class alpha_vboth_fx_1222:
    time_consume=6
    def __init__(self):
        self.ts_tcv1 = ts_tcv(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_tcv1(div_cs(add(get('buy_small_order_act_tot_fx'),get('buy_big_order_act_tot_fx')),add(get('buy_small_order_tot_fx'),get('buy_big_order_tot_fx'))))


class alpha_vboth_fx_1232:
    time_consume=22
    def __init__(self):
        self.ts_min1 = ts_min(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_sum_div_cs1(div(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),div(add(get('sell_small_order_up_fx'),get('sell_big_order_up_fx')),add(get('sell_small_order_up_cnt_fx'),get('sell_big_order_up_cnt_fx')))))


class alpha_vboth_fx_1234:
    time_consume=8
    def __init__(self):
        self.ts_tcv1 = ts_tcv(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_tcv1(self.ts_sum_div_cs1(get('buy_small_order_act_up_fx'),add(get('buy_small_order_up_fx'),get('buy_big_order_up_fx'),get('buy_small_order_down_fx'),get('buy_big_order_down_fx'))))


class alpha_vboth_fx_1235:
    time_consume=5
    def __init__(self):
        self.ts_std1 = ts_std(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_std1(div_cs(get('buy_small_order_act_tot_fx'),add(get('sell_small_order_tot_fx'),get('sell_big_order_tot_fx'))))


class alpha_vboth_fx_1242:
    time_consume=22
    def __init__(self):
        self.ts_min1 = ts_min(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_sum_div1(div(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),div(get('buy_small_order_down_fx'),get('buy_small_order_down_cnt_fx'))))


class alpha_vboth_fx_1254:
    time_consume=8
    def __init__(self):
        self.ts_tcv1 = ts_tcv(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_tcv1(self.ts_sum_div_cs1(div(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),div(get('sell_small_order_up_fx'),get('sell_small_order_up_cnt_fx'))))


class alpha_vboth_fx_1261:
    time_consume=6
    def __init__(self):
        self.ts_tcv1 = ts_tcv(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_tcv1(div_cs(div(get('small_actS_up_fx'),get('small_actS_cnt_up_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_1274:
    time_consume=41
    def __init__(self):
        self.ts_range1 = ts_range(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_avg1 = ts_avg(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_range1(self.ts_avg1(div(div(get('sell_small_order_down_fx'),get('sell_small_order_down_cnt_fx')),get('amount_trade_fx'))))


class alpha_vboth_fx_1277:
    time_consume=42
    def __init__(self):
        self.ts_range1 = ts_range(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_range1(self.ts_sum_div_cs1(div(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),div(add(get('sell_small_order_down_fx'),get('sell_big_order_down_fx')),add(get('sell_small_order_down_cnt_fx'),get('sell_big_order_down_cnt_fx')))))


class alpha_vboth_fx_1282:
    time_consume=4
    def __init__(self):
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div2 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_sum_div1(self.ts_sum_div2(get('sell_small_order_up_fx'),get('sell_small_order_up_cnt_fx')),get('amount_trade_fx'))


class alpha_vboth_fx_1293:
    time_consume=8
    def __init__(self):
        self.ts_tcv1 = ts_tcv(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_tcv1(div_cs(self.ts_sum_div1(get('small_actB_down_fx'),get('small_actB_cnt_down_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_1302:
    time_consume=8
    def __init__(self):
        self.ts_tcv1 = ts_tcv(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_tcv1(self.ts_sum_div_cs1(div(add(get('sell_small_order_down_fx'),get('sell_big_order_down_fx')),add(get('sell_small_order_down_cnt_fx'),get('sell_big_order_down_cnt_fx'))),div(add(get('sell_small_order_up_fx'),get('sell_big_order_up_fx')),add(get('sell_small_order_up_cnt_fx'),get('sell_big_order_up_cnt_fx')))))


class alpha_vboth_fx_1309:
    time_consume=42
    def __init__(self):
        self.ts_range1 = ts_range(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_range1(div_cs(self.ts_sum_div1(get('buy_small_order_down_fx'),get('buy_small_order_down_cnt_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_1311:
    time_consume=22
    def __init__(self):
        self.ts_max1 = ts_max(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_max1(self.ts_sum_div1(div(get('sell_small_order_down_fx'),get('sell_small_order_down_cnt_fx')),div(get('sell_small_order_up_fx'),get('sell_small_order_up_cnt_fx'))))


class alpha_vboth_fx_1319:
    time_consume=21
    def __init__(self):
        self.ts_max1 = ts_max(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_avg1 = ts_avg(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_max1(self.ts_avg1(div_cs(div(get('buy_small_order_down_fx'),get('buy_small_order_down_cnt_fx')),get('amount_trade_fx'))))


class alpha_vboth_fx_1324:
    time_consume=41
    def __init__(self):
        self.ts_range1 = ts_range(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_avg1 = ts_avg(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_range1(self.ts_avg1(div(div(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),get('amount_trade_fx'))))


class alpha_vboth_fx_1326:
    time_consume=22
    def __init__(self):
        self.ts_max1 = ts_max(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_max1(self.ts_sum_div_cs1(get('sell_small_order_act_down_fx'),get('amount_trade_fx')))


class alpha_vboth_fx_1331:
    time_consume=2
    def __init__(self):
        self.ts_sum_div1 = ts_sum_div(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_sum_div1(div(add(get('buy_small_order_up_fx'),get('buy_big_order_up_fx')),add(get('buy_small_order_up_cnt_fx'),get('buy_big_order_up_cnt_fx'))),div(add(get('buy_small_order_down_fx'),get('buy_big_order_down_fx')),add(get('buy_small_order_down_cnt_fx'),get('buy_big_order_down_cnt_fx'))))


class alpha_vboth_fx_1336:
    time_consume=8
    def __init__(self):
        self.ts_cv_abs1 = ts_cv_abs(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_cv_abs1(div_cs(self.ts_sum_div1(get('sell_small_order_up_fx'),get('sell_small_order_up_cnt_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_1350:
    time_consume=8
    def __init__(self):
        self.ts_cv_abs1 = ts_cv_abs(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_cv_abs1(self.ts_sum_div_cs1(div(get('small_actB_equi_fx'),get('small_actB_cnt_equi_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_1351:
    time_consume=5
    def __init__(self):
        self.ts_corr1 = ts_corr(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_corr1(div(div(get('small_actB_equi_fx'),get('small_actB_cnt_equi_fx')),get('amount_trade_fx')),get('amount_trade_fx'))


class alpha_vboth_fx_1352:
    time_consume=6
    def __init__(self):
        self.ts_cv_abs1 = ts_cv_abs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_cv_abs1(div_cs(get('sell_small_order_act_tot_fx'),add(get('buy_small_order_tot_fx'),get('buy_big_order_tot_fx'))))


class alpha_vboth_fx_1364:
    time_consume=9
    def __init__(self):
        self.ts_std1 = ts_std(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div2 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_std1(div_cs(self.ts_sum_div1(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),self.ts_sum_div2(add(get('sell_small_order_up_fx'),get('sell_big_order_up_fx'),get('sell_small_order_down_fx'),get('sell_big_order_down_fx')),add(get('sell_small_order_up_cnt_fx'),get('sell_big_order_up_cnt_fx'),get('sell_small_order_down_cnt_fx'),get('sell_big_order_down_cnt_fx')))))


class alpha_vboth_fx_1366:
    time_consume=6
    def __init__(self):
        self.ts_tcv1 = ts_tcv(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_tcv1(div_cs(add(get('small_actS_fx'),get('big_actS_fx')),add(get('small_actB_fx'),get('big_actB_fx'),get('small_actS_fx'),get('big_actS_fx'))))


class alpha_vboth_fx_1370:
    time_consume=6
    def __init__(self):
        self.ts_cv_abs1 = ts_cv_abs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_cv_abs1(div(get('sell_small_order_act_tot_fx'),get('sell_small_order_tot_fx')))


class alpha_vboth_fx_1373:
    time_consume=42
    def __init__(self):
        self.ts_range1 = ts_range(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_range1(div_cs(self.ts_sum_div1(get('sell_small_order_up_fx'),get('sell_small_order_up_cnt_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_1377:
    time_consume=6
    def __init__(self):
        self.ts_tcv1 = ts_tcv(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_tcv1(div_cs(get('buy_small_order_act_down_fx'),add(get('buy_small_order_up_fx'),get('buy_big_order_up_fx'),get('buy_small_order_down_fx'),get('buy_big_order_down_fx'))))


class alpha_vboth_fx_1381:
    time_consume=6
    def __init__(self):
        self.ts_tcv1 = ts_tcv(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_tcv1(div_cs(get('buy_small_order_act_tot_fx'),add(get('buy_small_order_act_tot_fx'),get('buy_big_order_act_tot_fx'))))


class alpha_vboth_fx_1383:
    time_consume=8
    def __init__(self):
        self.ts_cv_abs1 = ts_cv_abs(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_cv_abs1(div_cs(self.ts_sum_div1(get('buy_small_order_down_fx'),get('buy_small_order_down_cnt_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_1389:
    time_consume=22
    def __init__(self):
        self.ts_max1 = ts_max(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_max1(self.ts_sum_div1(div(get('sell_small_order_down_fx'),get('sell_small_order_down_cnt_fx')),div(get('sell_small_order_up_fx'),get('sell_small_order_up_cnt_fx'))))


class alpha_vboth_fx_1403:
    time_consume=8
    def __init__(self):
        self.ts_tcv1 = ts_tcv(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_tcv1(self.ts_sum_div_cs1(get('buy_small_order_act_tot_fx'),get('amount_trade_fx')))


class alpha_vboth_fx_1407:
    time_consume=5
    def __init__(self):
        self.ts_std1 = ts_std(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_std1(div(get('buy_small_order_act_tot_fx'),get('amount_trade_fx')))


class alpha_vboth_fx_1410:
    time_consume=3
    def __init__(self):
        self.ts_avg1 = ts_avg(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_avg1(div(self.ts_sum_div1(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_1415:
    time_consume=22
    def __init__(self):
        self.ts_max1 = ts_max(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_max1(self.ts_sum_div_cs1(div(get('sell_small_order_down_fx'),get('sell_small_order_down_cnt_fx')),div(get('sell_small_order_up_fx'),get('sell_small_order_up_cnt_fx'))))


class alpha_vboth_fx_1435:
    time_consume=8
    def __init__(self):
        self.ts_cv_abs1 = ts_cv_abs(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_cv_abs1(self.ts_sum_div_cs1(get('buy_small_order_act_tot_fx'),add(get('buy_small_order_tot_fx'),get('buy_big_order_tot_fx'))))


class alpha_vboth_fx_1438:
    time_consume=5
    def __init__(self):
        self.ts_corr1 = ts_corr(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_corr1(div(div(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),get('amount_trade_fx')),get('amount_trade_fx'))


class alpha_vboth_fx_1440:
    time_consume=21
    def __init__(self):
        self.ts_min1 = ts_min(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_avg1 = ts_avg(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_avg1(div_cs(div(get('sell_small_order_down_fx'),get('sell_small_order_down_cnt_fx')),get('amount_trade_fx'))))


class alpha_vboth_fx_1441:
    time_consume=21
    def __init__(self):
        self.ts_min1 = ts_min(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_avg1 = ts_avg(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_avg1(div_cs(div(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),get('amount_trade_fx'))))


class alpha_vboth_fx_1451:
    time_consume=3
    def __init__(self):
        self.ts_avg1 = ts_avg(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_avg1(div(self.ts_sum_div1(get('buy_small_order_down_fx'),get('buy_small_order_down_cnt_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_1452:
    time_consume=5
    def __init__(self):
        self.ts_std1 = ts_std(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_std1(div_cs(get('sell_small_order_act_tot_fx'),get('sell_small_order_tot_fx')))


class alpha_vboth_fx_1455:
    time_consume=5
    def __init__(self):
        self.ts_std1 = ts_std(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_std1(div_cs(get('buy_small_order_act_down_fx'),add(get('buy_small_order_up_fx'),get('buy_big_order_up_fx'),get('buy_small_order_down_fx'),get('buy_big_order_down_fx'))))


class alpha_vboth_fx_1458:
    time_consume=4
    def __init__(self):
        self.ts_sum_div1 = ts_sum_div(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div2 = ts_sum_div(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_sum_div1(self.ts_sum_div2(get('sell_small_order_up_fx'),get('sell_small_order_up_cnt_fx')),get('amount_trade_fx'))


class alpha_vboth_fx_1466:
    time_consume=5
    def __init__(self):
        self.ts_std1 = ts_std(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_std1(div_cs(add(get('buy_small_order_act_down_fx'),get('buy_big_order_act_down_fx')),add(get('buy_small_order_up_fx'),get('buy_big_order_up_fx'),get('buy_small_order_down_fx'),get('buy_big_order_down_fx'))))


class alpha_vboth_fx_1467:
    time_consume=6
    def __init__(self):
        self.ts_cv_abs1 = ts_cv_abs(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_cv_abs1(div_cs(get('sell_small_order_down_fx'),get('amount_trade_fx')))


class alpha_vboth_fx_1491:
    time_consume=3
    def __init__(self):
        self.ts_avg1 = ts_avg(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_avg1(self.ts_sum_div1(div(add(get('sell_small_order_down_fx'),get('sell_big_order_down_fx')),add(get('sell_small_order_down_cnt_fx'),get('sell_big_order_down_cnt_fx'))),div(add(get('sell_small_order_up_fx'),get('sell_big_order_up_fx')),add(get('sell_small_order_up_cnt_fx'),get('sell_big_order_up_cnt_fx')))))


class alpha_vboth_fx_1505:
    time_consume=8
    def __init__(self):
        self.ts_tcv1 = ts_tcv(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_tcv1(self.ts_sum_div_cs1(div(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),div(get('sell_big_order_up_fx'),get('sell_big_order_up_cnt_fx'))))


class alpha_vboth_fx_1515:
    time_consume=8
    def __init__(self):
        self.ts_tcv1 = ts_tcv(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_tcv1(self.ts_sum_div_cs1(div(get('sell_small_order_down_fx'),get('sell_small_order_down_cnt_fx')),div(get('sell_big_order_up_fx'),get('sell_big_order_up_cnt_fx'))))


class alpha_vboth_fx_1519:
    time_consume=8
    def __init__(self):
        self.ts_cv_abs1 = ts_cv_abs(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_cv_abs1(self.ts_sum_div1(div(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),div(get('sell_small_order_down_fx'),get('sell_small_order_down_cnt_fx'))))


class alpha_vboth_fx_1521:
    time_consume=42
    def __init__(self):
        self.ts_range1 = ts_range(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_range1(self.ts_sum_div1(get('buy_small_order_down_fx'),get('amount_trade_fx')))


class alpha_vboth_fx_1525:
    time_consume=41
    def __init__(self):
        self.ts_range1 = ts_range(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_avg1 = ts_avg(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_range1(self.ts_avg1(div(get('buy_small_order_act_tot_fx'),add(get('sell_small_order_act_tot_fx'),get('sell_big_order_act_tot_fx')))))


class alpha_vboth_fx_1528:
    time_consume=8
    def __init__(self):
        self.ts_cv_abs1 = ts_cv_abs(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_cv_abs1(div_cs(self.ts_sum_div1(get('sell_small_order_up_fx'),get('sell_small_order_up_cnt_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_1552:
    time_consume=7
    def __init__(self):
        self.ts_std1 = ts_std(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_std1(self.ts_sum_div1(div(add(get('sell_small_order_down_fx'),get('sell_big_order_down_fx')),add(get('sell_small_order_down_cnt_fx'),get('sell_big_order_down_cnt_fx'))),div(add(get('sell_small_order_up_fx'),get('sell_big_order_up_fx')),add(get('sell_small_order_up_cnt_fx'),get('sell_big_order_up_cnt_fx')))))


class alpha_vboth_fx_1561:
    time_consume=8
    def __init__(self):
        self.ts_tcv1 = ts_tcv(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_tcv1(self.ts_sum_div_cs1(get('buy_small_order_down_fx'),get('amount_trade_fx')))


class alpha_vboth_fx_1562:
    time_consume=8
    def __init__(self):
        self.ts_cv_abs1 = ts_cv_abs(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_cv_abs1(self.ts_sum_div_cs1(get('buy_small_order_down_fx'),get('amount_trade_fx')))


class alpha_vboth_fx_1566:
    time_consume=9
    def __init__(self):
        self.ts_std1 = ts_std(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div2 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_std1(div_cs(self.ts_sum_div1(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),self.ts_sum_div2(add(get('buy_small_order_up_fx'),get('buy_big_order_up_fx'),get('buy_small_order_down_fx'),get('buy_big_order_down_fx')),add(get('buy_small_order_up_cnt_fx'),get('buy_big_order_up_cnt_fx'),get('buy_small_order_down_cnt_fx'),get('buy_big_order_down_cnt_fx')))))


class alpha_vboth_fx_1567:
    time_consume=8
    def __init__(self):
        self.ts_cv_abs1 = ts_cv_abs(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_cv_abs1(self.ts_sum_div1(get('small_actS_equi_fx'),get('small_actS_cnt_equi_fx')))


class alpha_vboth_fx_1569:
    time_consume=5
    def __init__(self):
        self.ts_skew1 = ts_skew(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_skew1(div(get('small_actB_up_fx'),get('amount_trade_fx')))


class alpha_vboth_fx_1574:
    time_consume=41
    def __init__(self):
        self.ts_range1 = ts_range(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_avg1 = ts_avg(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_range1(self.ts_avg1(div(add(get('small_actB_fx'),get('big_actB_fx')),add(get('small_actS_fx'),get('big_actS_fx')))))


class alpha_vboth_fx_1587:
    time_consume=6
    def __init__(self):
        self.ts_tcv1 = ts_tcv(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_tcv1(div_cs(get('sell_small_order_act_up_fx'),get('amount_trade_fx')))


class alpha_vboth_fx_1590:
    time_consume=3
    def __init__(self):
        self.ts_avg1 = ts_avg(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_avg1(div(self.ts_sum_div1(get('sell_small_order_up_fx'),get('sell_small_order_up_cnt_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_1592:
    time_consume=8
    def __init__(self):
        self.ts_tcv1 = ts_tcv(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_tcv1(self.ts_sum_div_cs1(div(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),div(get('sell_small_order_up_fx'),get('sell_small_order_up_cnt_fx'))))


class alpha_vboth_fx_1600:
    time_consume=12
    def __init__(self):
        self.ts_tcv1 = ts_tcv(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div2 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_tcv1(self.ts_sum_div_cs1(self.ts_sum_div1(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),self.ts_sum_div2(get('buy_big_order_up_fx'),get('buy_big_order_up_cnt_fx'))))


class alpha_vboth_fx_1603:
    time_consume=21
    def __init__(self):
        self.ts_min1 = ts_min(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_avg1 = ts_avg(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_avg1(div_cs(div(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),div(add(get('sell_small_order_up_fx'),get('sell_big_order_up_fx')),add(get('sell_small_order_up_cnt_fx'),get('sell_big_order_up_cnt_fx'))))))


class alpha_vboth_fx_1606:
    time_consume=22
    def __init__(self):
        self.ts_max1 = ts_max(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_max1(self.ts_sum_div1(get('buy_small_order_act_tot_fx'),add(get('buy_small_order_tot_fx'),get('buy_big_order_tot_fx'))))


class alpha_vboth_fx_1607:
    time_consume=43
    def __init__(self):
        self.ts_range1 = ts_range(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_avg1 = ts_avg(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_range1(self.ts_avg1(div_cs(self.ts_sum_div1(get('small_actS_up_fx'),get('small_actS_cnt_up_fx')),get('amount_trade_fx'))))


class alpha_vboth_fx_1608:
    time_consume=22
    def __init__(self):
        self.ts_max1 = ts_max(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_max1(self.ts_sum_div1(get('buy_small_order_act_tot_fx'),add(get('sell_small_order_tot_fx'),get('sell_big_order_tot_fx'))))


class alpha_vboth_fx_1610:
    time_consume=6
    def __init__(self):
        self.ts_tcv1 = ts_tcv(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_tcv1(div_cs(div(get('buy_small_order_down_fx'),get('buy_small_order_down_cnt_fx')),div(add(get('buy_small_order_down_fx'),get('buy_big_order_down_fx')),add(get('buy_small_order_down_cnt_fx'),get('buy_big_order_down_cnt_fx')))))


class alpha_vboth_fx_1624:
    time_consume=22
    def __init__(self):
        self.ts_min1 = ts_min(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_sum_div_cs1(div(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),div(get('buy_big_order_down_fx'),get('buy_big_order_down_cnt_fx'))))


class alpha_vboth_fx_1631:
    time_consume=3
    def __init__(self):
        self.ts_avg1 = ts_avg(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_avg1(self.ts_sum_div1(div(get('sell_small_order_down_fx'),get('sell_small_order_down_cnt_fx')),div(get('sell_small_order_up_fx'),get('sell_small_order_up_cnt_fx'))))


class alpha_vboth_fx_1641:
    time_consume=8
    def __init__(self):
        self.ts_tcv1 = ts_tcv(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_tcv1(self.ts_sum_div_cs1(div(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),div(get('buy_big_order_up_fx'),get('buy_big_order_up_cnt_fx'))))


class alpha_vboth_fx_1642:
    time_consume=5
    def __init__(self):
        self.ts_corr1 = ts_corr(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_corr1(div(div(get('sell_small_order_down_fx'),get('sell_small_order_down_cnt_fx')),get('amount_trade_fx')),get('amount_trade_fx'))


class alpha_vboth_fx_1645:
    time_consume=8
    def __init__(self):
        self.ts_tcv1 = ts_tcv(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_tcv1(self.ts_sum_div_cs1(div(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),div(get('buy_big_order_up_fx'),get('buy_big_order_up_cnt_fx'))))


class alpha_vboth_fx_1666:
    time_consume=42
    def __init__(self):
        self.ts_range1 = ts_range(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_range1(self.ts_sum_div_cs1(get('sell_small_order_up_fx'),get('amount_trade_fx')))


class alpha_vboth_fx_1670:
    time_consume=8
    def __init__(self):
        self.ts_tcv1 = ts_tcv(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_tcv1(self.ts_sum_div_cs1(div(add(get('buy_small_order_up_fx'),get('buy_big_order_up_fx')),add(get('buy_small_order_up_cnt_fx'),get('buy_big_order_up_cnt_fx'))),div(add(get('buy_small_order_down_fx'),get('buy_big_order_down_fx')),add(get('buy_small_order_down_cnt_fx'),get('buy_big_order_down_cnt_fx')))))


class alpha_vboth_fx_1674:
    time_consume=5
    def __init__(self):
        self.ts_std1 = ts_std(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_std1(div_cs(get('sell_small_order_down_fx'),get('amount_trade_fx')))


class alpha_vboth_fx_1679:
    time_consume=42
    def __init__(self):
        self.ts_range1 = ts_range(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_range1(self.ts_sum_div_cs1(get('buy_small_order_act_up_fx'),get('amount_trade_fx')))


class alpha_vboth_fx_1688:
    time_consume=42
    def __init__(self):
        self.ts_range1 = ts_range(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_range1(self.ts_sum_div1(get('buy_small_order_act_up_fx'),get('amount_trade_fx')))


class alpha_vboth_fx_1689:
    time_consume=5
    def __init__(self):
        self.ts_std1 = ts_std(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_std1(div_cs(get('buy_small_order_act_tot_fx'),add(get('sell_small_order_act_tot_fx'),get('sell_big_order_act_tot_fx'))))


class alpha_vboth_fx_1691:
    time_consume=8
    def __init__(self):
        self.ts_cv_abs1 = ts_cv_abs(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_cv_abs1(self.ts_sum_div_cs1(div(get('small_actS_equi_fx'),get('small_actS_cnt_equi_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_1692:
    time_consume=6
    def __init__(self):
        self.ts_cv_abs1 = ts_cv_abs(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_cv_abs1(div_cs(get('buy_small_order_up_fx'),get('amount_trade_fx')))


class alpha_vboth_fx_1703:
    time_consume=8
    def __init__(self):
        self.ts_tcv1 = ts_tcv(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_tcv1(self.ts_sum_div_cs1(div(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),div(get('sell_big_order_up_fx'),get('sell_big_order_up_cnt_fx'))))


class alpha_vboth_fx_1704:
    time_consume=9
    def __init__(self):
        self.ts_std1 = ts_std(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div2 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_std1(div_cs(self.ts_sum_div1(get('sell_small_order_down_fx'),get('sell_small_order_down_cnt_fx')),self.ts_sum_div2(add(get('sell_small_order_up_fx'),get('sell_big_order_up_fx'),get('sell_small_order_down_fx'),get('sell_big_order_down_fx')),add(get('sell_small_order_up_cnt_fx'),get('sell_big_order_up_cnt_fx'),get('sell_small_order_down_cnt_fx'),get('sell_big_order_down_cnt_fx')))))


class alpha_vboth_fx_1716:
    time_consume=5
    def __init__(self):
        self.ts_std1 = ts_std(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_std1(div_cs(get('sell_small_order_act_tot_fx'),get('amount_trade_fx')))


class alpha_vboth_fx_1722:
    time_consume=22
    def __init__(self):
        self.ts_min1 = ts_min(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_sum_div_cs1(div(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),div(get('buy_big_order_down_fx'),get('buy_big_order_down_cnt_fx'))))


class alpha_vboth_fx_1723:
    time_consume=8
    def __init__(self):
        self.ts_tcv1 = ts_tcv(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_tcv1(self.ts_sum_div_cs1(div(get('sell_small_order_down_fx'),get('sell_small_order_down_cnt_fx')),div(get('sell_big_order_up_fx'),get('sell_big_order_up_cnt_fx'))))


class alpha_vboth_fx_1734:
    time_consume=22
    def __init__(self):
        self.ts_min1 = ts_min(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_sum_div1(div(get('sell_small_order_down_fx'),get('sell_small_order_down_cnt_fx')),div(get('sell_big_order_down_fx'),get('sell_big_order_down_cnt_fx'))))


class alpha_vboth_fx_1736:
    time_consume=22
    def __init__(self):
        self.ts_min1 = ts_min(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_sum_div1(div(get('sell_small_order_down_fx'),get('sell_small_order_down_cnt_fx')),div(get('sell_big_order_down_fx'),get('sell_big_order_down_cnt_fx'))))


class alpha_vboth_fx_1760:
    time_consume=6
    def __init__(self):
        self.ts_tcv1 = ts_tcv(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_tcv1(div_cs(div(get('sell_small_order_up_fx'),get('sell_small_order_up_cnt_fx')),div(add(get('sell_small_order_up_fx'),get('sell_big_order_up_fx')),add(get('sell_small_order_up_cnt_fx'),get('sell_big_order_up_cnt_fx')))))


class alpha_vboth_fx_1763:
    time_consume=5
    def __init__(self):
        self.ts_std1 = ts_std(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_std1(div_cs(get('small_actB_fx'),add(get('small_actS_fx'),get('big_actS_fx'))))


class alpha_vboth_fx_1777:
    time_consume=8
    def __init__(self):
        self.ts_tcv1 = ts_tcv(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_tcv1(self.ts_sum_div_cs1(get('buy_small_order_act_up_fx'),get('amount_trade_fx')))


class alpha_vboth_fx_1818:
    time_consume=6
    def __init__(self):
        self.ts_cv_abs1 = ts_cv_abs(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_cv_abs1(div_cs(get('buy_small_order_act_tot_fx'),add(get('sell_small_order_tot_fx'),get('sell_big_order_tot_fx'))))


class alpha_vboth_fx_1824:
    time_consume=5
    def __init__(self):
        self.ts_std1 = ts_std(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_std1(div_cs(get('buy_small_order_up_fx'),get('amount_trade_fx')))


class alpha_vboth_fx_1826:
    time_consume=21
    def __init__(self):
        self.ts_max1 = ts_max(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_avg1 = ts_avg(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_max1(self.ts_avg1(div_cs(get('buy_small_order_act_tot_fx'),get('amount_trade_fx'))))


class alpha_vboth_fx_1827:
    time_consume=21
    def __init__(self):
        self.ts_max1 = ts_max(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_avg1 = ts_avg(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_max1(self.ts_avg1(div_cs(get('buy_small_order_act_tot_fx'),add(get('buy_small_order_tot_fx'),get('buy_big_order_tot_fx')))))


class alpha_vboth_fx_1832:
    time_consume=8
    def __init__(self):
        self.ts_tcv1 = ts_tcv(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_tcv1(self.ts_sum_div_cs1(div(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),div(add(get('sell_small_order_up_fx'),get('sell_big_order_up_fx')),add(get('sell_small_order_up_cnt_fx'),get('sell_big_order_up_cnt_fx')))))


class alpha_vboth_fx_1833:
    time_consume=3
    def __init__(self):
        self.ts_avg1 = ts_avg(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_avg1(self.ts_sum_div_cs1(div(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),div(get('buy_big_order_up_fx'),get('buy_big_order_up_cnt_fx'))))


class alpha_vboth_fx_1854:
    time_consume=8
    def __init__(self):
        self.ts_cv_abs1 = ts_cv_abs(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_cv_abs1(div_cs(self.ts_sum_div1(get('small_actS_down_fx'),get('small_actS_cnt_down_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_1859:
    time_consume=21
    def __init__(self):
        self.ts_min1 = ts_min(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_avg1 = ts_avg(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_avg1(div_cs(div(get('sell_small_order_down_fx'),get('sell_small_order_down_cnt_fx')),div(add(get('sell_small_order_up_fx'),get('sell_big_order_up_fx')),add(get('sell_small_order_up_cnt_fx'),get('sell_big_order_up_cnt_fx'))))))


class alpha_vboth_fx_1864:
    time_consume=6
    def __init__(self):
        self.ts_tcv1 = ts_tcv(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_tcv1(div_cs(div(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),div(add(get('buy_small_order_up_fx'),get('buy_big_order_up_fx')),add(get('buy_small_order_up_cnt_fx'),get('buy_big_order_up_cnt_fx')))))


class alpha_vboth_fx_1868:
    time_consume=41
    def __init__(self):
        self.ts_range1 = ts_range(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_avg1 = ts_avg(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_range1(self.ts_avg1(div(div(get('small_actB_equi_fx'),get('small_actB_cnt_equi_fx')),get('amount_trade_fx'))))


class alpha_vboth_fx_1869:
    time_consume=8
    def __init__(self):
        self.ts_cv_abs1 = ts_cv_abs(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_cv_abs1(div_cs(self.ts_sum_div1(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_1872:
    time_consume=8
    def __init__(self):
        self.ts_tcv1 = ts_tcv(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_tcv1(self.ts_sum_div_cs1(div(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),div(get('sell_big_order_up_fx'),get('sell_big_order_up_cnt_fx'))))


class alpha_vboth_fx_1874:
    time_consume=11
    def __init__(self):
        self.ts_std1 = ts_std(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div2 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div3 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_std1(self.ts_sum_div1(self.ts_sum_div2(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),self.ts_sum_div3(get('buy_big_order_up_fx'),get('buy_big_order_up_cnt_fx'))))


class alpha_vboth_fx_1876:
    time_consume=8
    def __init__(self):
        self.ts_tcv1 = ts_tcv(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_tcv1(self.ts_sum_div_cs1(div(get('sell_small_order_down_fx'),get('sell_small_order_down_cnt_fx')),div(get('sell_big_order_down_fx'),get('sell_big_order_down_cnt_fx'))))


class alpha_vboth_fx_1881:
    time_consume=8
    def __init__(self):
        self.ts_tcv1 = ts_tcv(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_tcv1(self.ts_sum_div_cs1(div(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),div(get('sell_big_order_down_fx'),get('sell_big_order_down_cnt_fx'))))


class alpha_vboth_fx_1887:
    time_consume=23
    def __init__(self):
        self.ts_max1 = ts_max(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_avg1 = ts_avg(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_max1(self.ts_avg1(div(self.ts_sum_div1(get('sell_small_order_up_fx'),get('sell_small_order_up_cnt_fx')),get('amount_trade_fx'))))


class alpha_vboth_fx_1903:
    time_consume=22
    def __init__(self):
        self.ts_min1 = ts_min(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_sum_div_cs1(get('sell_small_order_act_tot_fx'),get('buy_big_order_tot_fx')))


class alpha_vboth_fx_1904:
    time_consume=8
    def __init__(self):
        self.ts_tcv1 = ts_tcv(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_tcv1(self.ts_sum_div_cs1(div(add(get('sell_small_order_down_fx'),get('sell_big_order_down_fx')),add(get('sell_small_order_down_cnt_fx'),get('sell_big_order_down_cnt_fx'))),div(add(get('sell_small_order_up_fx'),get('sell_big_order_up_fx')),add(get('sell_small_order_up_cnt_fx'),get('sell_big_order_up_cnt_fx')))))


class alpha_vboth_fx_1908:
    time_consume=23
    def __init__(self):
        self.ts_max1 = ts_max(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_avg1 = ts_avg(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_max1(self.ts_avg1(div_cs(self.ts_sum_div1(get('buy_small_order_down_fx'),get('buy_small_order_down_cnt_fx')),get('amount_trade_fx'))))


class alpha_vboth_fx_1913:
    time_consume=43
    def __init__(self):
        self.ts_range1 = ts_range(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_avg1 = ts_avg(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_range1(self.ts_avg1(div(self.ts_sum_div1(get('sell_big_order_down_fx'),get('sell_big_order_down_cnt_fx')),get('amount_trade_fx'))))


class alpha_vboth_fx_1922:
    time_consume=43
    def __init__(self):
        self.ts_range1 = ts_range(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_avg1 = ts_avg(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_range1(self.ts_avg1(div_cs(self.ts_sum_div1(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),get('amount_trade_fx'))))


class alpha_vboth_fx_1923:
    time_consume=7
    def __init__(self):
        self.ts_std1 = ts_std(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_std1(div(self.ts_sum_div1(get('small_actS_down_fx'),get('small_actS_cnt_down_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_1926:
    time_consume=6
    def __init__(self):
        self.ts_cv_abs1 = ts_cv_abs(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_cv_abs1(div(get('sell_small_order_act_tot_fx'),add(get('buy_small_order_tot_fx'),get('buy_big_order_tot_fx'))))


class alpha_vboth_fx_1930:
    time_consume=23
    def __init__(self):
        self.ts_max1 = ts_max(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_avg1 = ts_avg(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_max1(self.ts_avg1(div_cs(self.ts_sum_div1(get('sell_small_order_up_fx'),get('sell_small_order_up_cnt_fx')),get('amount_trade_fx'))))


class alpha_vboth_fx_1939:
    time_consume=3
    def __init__(self):
        self.ts_avg1 = ts_avg(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_avg1(div_cs(self.ts_sum_div1(get('buy_small_order_down_fx'),get('buy_small_order_down_cnt_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_1940:
    time_consume=23
    def __init__(self):
        self.ts_min1 = ts_min(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_avg1 = ts_avg(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_avg1(div(self.ts_sum_div1(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),get('amount_trade_fx'))))


class alpha_vboth_fx_1960:
    time_consume=8
    def __init__(self):
        self.ts_tcv1 = ts_tcv(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_tcv1(self.ts_sum_div_cs1(div(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),div(get('buy_big_order_down_fx'),get('buy_big_order_down_cnt_fx'))))


class alpha_vboth_fx_1963:
    time_consume=41
    def __init__(self):
        self.ts_range1 = ts_range(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_avg1 = ts_avg(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_range1(self.ts_avg1(div(div(get('sell_small_order_down_fx'),get('sell_small_order_down_cnt_fx')),get('amount_trade_fx'))))


class alpha_vboth_fx_1964:
    time_consume=41
    def __init__(self):
        self.ts_range1 = ts_range(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_avg1 = ts_avg(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_range1(self.ts_avg1(div(div(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),get('amount_trade_fx'))))


class alpha_vboth_fx_1965:
    time_consume=7
    def __init__(self):
        self.ts_std1 = ts_std(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_std1(div_cs(self.ts_sum_div1(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_1971:
    time_consume=7
    def __init__(self):
        self.ts_std1 = ts_std(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_std1(div_cs(self.ts_sum_div1(get('sell_small_order_down_fx'),get('sell_small_order_down_cnt_fx')),get('amount_trade_fx')))


class alpha_vboth_fx_1973:
    time_consume=21
    def __init__(self):
        self.ts_min1 = ts_min(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_avg1 = ts_avg(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_avg1(div_cs(div(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),div(get('sell_small_order_up_fx'),get('sell_small_order_up_cnt_fx')))))


class alpha_vboth_fx_1980:
    time_consume=12
    def __init__(self):
        self.ts_tcv1 = ts_tcv(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div2 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_tcv1(self.ts_sum_div_cs1(self.ts_sum_div1(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),self.ts_sum_div2(get('buy_big_order_up_fx'),get('buy_big_order_up_cnt_fx'))))


class alpha_vboth_fx_1993:
    time_consume=10
    def __init__(self):
        self.ts_tcv1 = ts_tcv(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div2 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_tcv1(div_cs(self.ts_sum_div1(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),self.ts_sum_div2(get('buy_big_order_up_fx'),get('buy_big_order_up_cnt_fx'))))


class alpha_vboth_fx_1994:
    time_consume=43
    def __init__(self):
        self.ts_range1 = ts_range(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_avg1 = ts_avg(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_range1(self.ts_avg1(div(self.ts_sum_div1(get('buy_small_order_down_fx'),get('buy_small_order_down_cnt_fx')),get('amount_trade_fx'))))


class alpha_vboth_fx_1995:
    time_consume=21
    def __init__(self):
        self.ts_min1 = ts_min(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_avg1 = ts_avg(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_avg1(div_cs(div(get('sell_small_order_up_fx'),get('sell_small_order_up_cnt_fx')),div(add(get('sell_small_order_up_fx'),get('sell_big_order_up_fx')),add(get('sell_small_order_up_cnt_fx'),get('sell_big_order_up_cnt_fx'))))))


class alpha_vboth_fx_1996:
    time_consume=21
    def __init__(self):
        self.ts_min1 = ts_min(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_avg1 = ts_avg(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_avg1(div_cs(div(get('sell_small_order_up_fx'),get('sell_small_order_up_cnt_fx')),div(add(get('sell_small_order_up_fx'),get('sell_big_order_up_fx')),add(get('sell_small_order_up_cnt_fx'),get('sell_big_order_up_cnt_fx'))))))


class alpha_vboth_fx_2001:
    time_consume=6
    def __init__(self):
        self.ts_tcv1 = ts_tcv(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_tcv1(div_cs(get('buy_small_order_tot_fx'),add(get('buy_small_order_tot_fx'),get('buy_big_order_tot_fx'))))


class alpha_vboth_fx_2002:
    time_consume=12
    def __init__(self):
        self.ts_tcv1 = ts_tcv(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div2 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_tcv1(self.ts_sum_div_cs1(self.ts_sum_div1(get('buy_small_order_down_fx'),get('buy_small_order_down_cnt_fx')),self.ts_sum_div2(get('buy_big_order_up_fx'),get('buy_big_order_up_cnt_fx'))))


class alpha_vboth_fx_2003:
    time_consume=21
    def __init__(self):
        self.ts_min1 = ts_min(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_avg1 = ts_avg(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_avg1(div_cs(div(get('buy_small_order_down_fx'),get('buy_small_order_down_cnt_fx')),div(add(get('buy_small_order_down_fx'),get('buy_big_order_down_fx')),add(get('buy_small_order_down_cnt_fx'),get('buy_big_order_down_cnt_fx'))))))


class alpha_vboth_fx_2010:
    time_consume=21
    def __init__(self):
        self.ts_min1 = ts_min(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_avg1 = ts_avg(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_avg1(div_cs(div(get('sell_small_order_up_fx'),get('sell_small_order_up_cnt_fx')),div(add(get('sell_small_order_up_fx'),get('sell_big_order_up_fx')),add(get('sell_small_order_up_cnt_fx'),get('sell_big_order_up_cnt_fx'))))))


class alpha_vboth_fx_2015:
    time_consume=21
    def __init__(self):
        self.ts_min1 = ts_min(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_avg1 = ts_avg(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_avg1(div_cs(div(get('sell_small_order_down_fx'),get('sell_small_order_down_cnt_fx')),div(get('sell_small_order_up_fx'),get('sell_small_order_up_cnt_fx')))))


class alpha_vboth_fx_2024:
    time_consume=21
    def __init__(self):
        self.ts_min1 = ts_min(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_avg1 = ts_avg(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_avg1(div_cs(div(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),div(add(get('buy_small_order_up_fx'),get('buy_big_order_up_fx')),add(get('buy_small_order_up_cnt_fx'),get('buy_big_order_up_cnt_fx'))))))


class alpha_vboth_fx_2025:
    time_consume=43
    def __init__(self):
        self.ts_range1 = ts_range(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_avg1 = ts_avg(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_range1(self.ts_avg1(div(self.ts_sum_div1(get('small_actS_up_fx'),get('small_actS_cnt_up_fx')),get('amount_trade_fx'))))


class alpha_vboth_fx_2026:
    time_consume=43
    def __init__(self):
        self.ts_range1 = ts_range(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_avg1 = ts_avg(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_range1(self.ts_avg1(div(self.ts_sum_div1(get('small_actB_down_fx'),get('small_actB_cnt_down_fx')),get('amount_trade_fx'))))


class alpha_vboth_fx_2028:
    time_consume=3
    def __init__(self):
        self.ts_up_ratio1 = ts_up_ratio(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_up_ratio1(div_cs(div(get('buy_big_order_down_fx'),get('buy_big_order_down_cnt_fx')),div(add(get('buy_small_order_up_fx'),get('buy_big_order_up_fx'),get('buy_small_order_down_fx'),get('buy_big_order_down_fx')),add(get('buy_small_order_up_cnt_fx'),get('buy_big_order_up_cnt_fx'),get('buy_small_order_down_cnt_fx'),get('buy_big_order_down_cnt_fx')))))


class alpha_vboth_fx_2030:
    time_consume=3
    def __init__(self):
        self.ts_up_ratio1 = ts_up_ratio(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_up_ratio1(div_cs(div(get('buy_small_order_down_fx'),get('buy_small_order_down_cnt_fx')),div(add(get('buy_small_order_up_fx'),get('buy_big_order_up_fx'),get('buy_small_order_down_fx'),get('buy_big_order_down_fx')),add(get('buy_small_order_up_cnt_fx'),get('buy_big_order_up_cnt_fx'),get('buy_small_order_down_cnt_fx'),get('buy_big_order_down_cnt_fx')))))


class alpha_vboth_fx_2040:
    time_consume=10
    def __init__(self):
        self.ts_tcv1 = ts_tcv(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div2 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_tcv1(div_cs(self.ts_sum_div1(get('buy_small_order_down_fx'),get('buy_small_order_down_cnt_fx')),self.ts_sum_div2(get('buy_big_order_up_fx'),get('buy_big_order_up_cnt_fx'))))


class alpha_vboth_fx_2043:
    time_consume=8
    def __init__(self):
        self.ts_cv_abs1 = ts_cv_abs(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div_cs1 = ts_sum_div_cs(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_cv_abs1(self.ts_sum_div_cs1(get('buy_small_order_act_tot_fx'),get('amount_trade_fx')))


class alpha_vboth_fx_2044:
    time_consume=1
    def __init__(self):
        self.ts_avg1 = ts_avg(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_avg1(div_cs(div(get('buy_small_order_down_fx'),get('buy_small_order_down_cnt_fx')),div(add(get('buy_small_order_down_fx'),get('buy_big_order_down_fx')),add(get('buy_small_order_down_cnt_fx'),get('buy_big_order_down_cnt_fx')))))


class alpha_vboth_fx_2045:
    time_consume=21
    def __init__(self):
        self.ts_min1 = ts_min(237,size=DC.stock_nums,dtype=np.float32)
        self.ts_avg1 = ts_avg(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_min1(self.ts_avg1(div_cs(div(get('buy_small_order_down_fx'),get('buy_small_order_down_cnt_fx')),div(add(get('buy_small_order_down_fx'),get('buy_big_order_down_fx')),add(get('buy_small_order_down_cnt_fx'),get('buy_big_order_down_cnt_fx'))))))


class alpha_vboth_fx_2046:
    time_consume=41
    def __init__(self):
        self.ts_range1 = ts_range(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_avg1 = ts_avg(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_range1(self.ts_avg1(div(div(get('small_actB_equi_fx'),get('small_actB_cnt_equi_fx')),get('amount_trade_fx'))))


class alpha_vboth_fx_2050:
    time_consume=41
    def __init__(self):
        self.ts_range1 = ts_range(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_avg1 = ts_avg(237,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_range1(self.ts_avg1(div_cs(div(get('buy_small_order_up_fx'),get('buy_small_order_up_cnt_fx')),get('amount_trade_fx'))))


class alpha_vboth_fx_2057:
    time_consume=25
    def __init__(self):
        self.ts_max1 = ts_max(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_avg1 = ts_avg(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div1 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)
        self.ts_sum_div2 = ts_sum_div(30,size=DC.stock_nums,dtype=np.float32)

    def update(self):
        return self.ts_max1(self.ts_avg1(div_cs(self.ts_sum_div1(get('buy_small_order_down_fx'),get('buy_small_order_down_cnt_fx')),self.ts_sum_div2(get('buy_big_order_up_fx'),get('buy_big_order_up_cnt_fx')))))

