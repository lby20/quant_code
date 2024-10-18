"""
@author: caiyucheng
@contact: caiyucheng23@163.com
@date: 2024-07-02
"""
# from sn_platform.config import C
# from sn_platform.data_utils.data_storage import FactorStorageByLocal
# from sn_platform.data_utils.sn_factor import CrossSectionalFactorPool
# from sn_platform.data_utils.sn_data import D, logger

from mcqueen_quant.cs_factor.cs_factor_base import CSFactorBase
from mcqueen_quant.cs_factor.cs_factor_base import logger
from data_bucket import data_api as dapi
import pandas as pd
import numpy as np
from copy import deepcopy
from . import factor_expression as fe
import shutil
# import factor_expression as fe

from multiprocessing import (
    Queue, Process, Value, Lock
)
import multiprocessing
from . import cyc_operator
# import cyc_operator


from collections import defaultdict
import os
import pickle as pkl
import datetime


dev_features = [
    'big_actB_fx', 
    'big_actB_cnt_fx', 
    'big_actB_vol_fx', 
    'small_actB_fx', 
    'small_actB_cnt_fx', 
    'small_actB_vol_fx', 
    'big_actS_fx', 
    'big_actS_cnt_fx', 
    'big_actS_vol_fx', 
    'small_actS_fx', 
    'small_actS_cnt_fx', 
    'small_actS_vol_fx', 
    'big_actB_up_fx', 
    'big_actB_cnt_up_fx', 
    'big_actB_vol_up_fx', 
    'small_actB_up_fx', 
    'small_actB_cnt_up_fx', 
    'small_actB_vol_up_fx', 
    'big_actS_up_fx', 
    'big_actS_cnt_up_fx', 
    'big_actS_vol_up_fx', 
    'small_actS_up_fx', 
    'small_actS_cnt_up_fx', 
    'small_actS_vol_up_fx', 
    'big_actB_down_fx', 
    'big_actB_cnt_down_fx', 
    'big_actB_vol_down_fx', 
    'small_actB_down_fx', 
    'small_actB_cnt_down_fx', 
    'small_actB_vol_down_fx', 
    'big_actS_down_fx', 
    'big_actS_cnt_down_fx', 
    'big_actS_vol_down_fx', 
    'small_actS_down_fx', 
    'small_actS_cnt_down_fx', 
    'small_actS_vol_down_fx', 
    'big_actB_equi_fx', 
    'big_actB_cnt_equi_fx', 
    'big_actB_vol_equi_fx', 
    'small_actB_equi_fx', 
    'small_actB_cnt_equi_fx', 
    'small_actB_vol_equi_fx', 
    'big_actS_equi_fx', 
    'big_actS_cnt_equi_fx', 
    'big_actS_vol_equi_fx', 
    'small_actS_equi_fx', 
    'small_actS_cnt_equi_fx', 
    'small_actS_vol_equi_fx', 
    'buy_big_order_tot_fx', 
    'buy_big_order_act_tot_fx', 
    'buy_big_order_cnt_fx', 
    'buy_big_order_act_cnt_fx', 
    'buy_big_order_tot_vol_fx', 
    'buy_big_order_act_vol_fx', 
    'buy_small_order_tot_fx', 
    'buy_small_order_act_tot_fx', 
    'buy_small_order_cnt_fx', 
    'buy_small_order_act_cnt_fx', 
    'buy_small_order_tot_vol_fx', 
    'buy_small_order_act_vol_fx', 
    'sell_big_order_tot_fx', 
    'sell_big_order_act_tot_fx', 
    'sell_big_order_cnt_fx', 
    'sell_big_order_act_cnt_fx', 
    'sell_big_order_tot_vol_fx', 
    'sell_big_order_act_vol_fx', 
    'sell_small_order_tot_fx', 
    'sell_small_order_act_tot_fx', 
    'sell_small_order_cnt_fx', 
    'sell_small_order_act_cnt_fx', 
    'sell_small_order_tot_vol_fx', 
    'sell_small_order_act_vol_fx', 
    'buy_big_order_up_fx', 
    'buy_big_order_act_up_fx', 
    'buy_big_order_up_cnt_fx', 
    'buy_big_order_up_vol_fx', 
    'buy_big_order_act_up_vol_fx', 
    'buy_small_order_up_fx', 
    'buy_small_order_act_up_fx', 
    'buy_small_order_up_cnt_fx', 
    'buy_small_order_up_vol_fx', 
    'buy_small_order_act_up_vol_fx', 
    'sell_big_order_up_fx', 
    'sell_big_order_act_up_fx', 
    'sell_big_order_up_cnt_fx', 
    'sell_big_order_up_vol_fx', 
    'sell_big_order_act_up_vol_fx', 
    'sell_small_order_up_fx', 
    'sell_small_order_act_up_fx', 
    'sell_small_order_up_cnt_fx', 
    'sell_small_order_up_vol_fx', 
    'sell_small_order_act_up_vol_fx', 
    'buy_big_order_down_fx', 
    'buy_big_order_act_down_fx', 
    'buy_big_order_down_cnt_fx', 
    'buy_big_order_down_vol_fx', 
    'buy_big_order_act_down_vol_fx', 
    'buy_small_order_down_fx', 
    'buy_small_order_act_down_fx', 
    'buy_small_order_down_cnt_fx', 
    'buy_small_order_down_vol_fx', 
    'buy_small_order_act_down_vol_fx', 
    'sell_big_order_down_fx', 
    'sell_big_order_act_down_fx', 
    'sell_big_order_down_cnt_fx', 
    'sell_big_order_down_vol_fx', 
    'sell_big_order_act_down_vol_fx', 
    'sell_small_order_down_fx', 
    'sell_small_order_act_down_fx', 
    'sell_small_order_down_cnt_fx', 
    'sell_small_order_down_vol_fx', 
    'sell_small_order_act_down_vol_fx', 
    'buy_big_order_spreadup_fx', 
    'buy_big_order_act_spreadup_fx', 
    'buy_big_order_spreadup_cnt_fx', 
    'buy_big_order_spreadup_vol_fx', 
    'buy_big_order_act_spreadup_vol_fx', 
    'buy_small_order_spreadup_fx', 
    'buy_small_order_act_spreadup_fx', 
    'buy_small_order_spreadup_cnt_fx', 
    'buy_small_order_spreadup_vol_fx', 
    'buy_small_order_act_spreadup_vol_fx', 
    'sell_big_order_spreadup_fx', 
    'sell_big_order_act_spreadup_fx', 
    'sell_big_order_spreadup_cnt_fx', 
    'sell_big_order_spreadup_vol_fx', 
    'sell_big_order_act_spreadup_vol_fx', 
    'sell_small_order_spreadup_fx', 
    'sell_small_order_act_spreadup_fx', 
    'sell_small_order_spreadup_cnt_fx', 
    'sell_small_order_spreadup_vol_fx', 
    'sell_small_order_act_spreadup_vol_fx', 
    'buy_big_order_spreaddown_fx', 
    'buy_big_order_act_spreaddown_fx', 
    'buy_big_order_spreaddown_cnt_fx', 
    'buy_big_order_spreaddown_vol_fx', 
    'buy_big_order_act_spreaddown_vol_fx', 
    'buy_small_order_spreaddown_fx', 
    'buy_small_order_act_spreaddown_fx', 
    'buy_small_order_spreaddown_cnt_fx', 
    'buy_small_order_spreaddown_vol_fx', 
    'buy_small_order_act_spreaddown_vol_fx', 
    'sell_big_order_spreaddown_fx', 
    'sell_big_order_act_spreaddown_fx', 
    'sell_big_order_spreaddown_cnt_fx', 
    'sell_big_order_spreaddown_vol_fx', 
    'sell_big_order_act_spreaddown_vol_fx', 
    'sell_small_order_spreaddown_fx', 
    'sell_small_order_act_spreaddown_fx', 
    'sell_small_order_spreaddown_cnt_fx', 
    'sell_small_order_spreaddown_vol_fx', 
    'sell_small_order_act_spreaddown_vol_fx', 
    'buy_big_order_sellagg_fx', 
    'buy_big_order_act_sellagg_fx', 
    'buy_big_order_sellagg_cnt_fx', 
    'buy_big_order_sellagg_vol_fx', 
    'buy_big_order_act_sellagg_vol_fx', 
    'buy_small_order_sellagg_fx', 
    'buy_small_order_act_sellagg_fx', 
    'buy_small_order_sellagg_cnt_fx', 
    'buy_small_order_sellagg_vol_fx', 
    'buy_small_order_act_sellagg_vol_fx', 
    'sell_big_order_sellagg_fx', 
    'sell_big_order_act_sellagg_fx', 
    'sell_big_order_sellagg_cnt_fx', 
    'sell_big_order_sellagg_vol_fx', 
    'sell_big_order_act_sellagg_vol_fx', 
    'sell_small_order_sellagg_fx', 
    'sell_small_order_act_sellagg_fx', 
    'sell_small_order_sellagg_cnt_fx', 
    'sell_small_order_sellagg_vol_fx', 
    'sell_small_order_act_sellagg_vol_fx', 
    'buy_big_order_buyagg_fx', 
    'buy_big_order_act_buyagg_fx', 
    'buy_big_order_buyagg_cnt_fx', 
    'buy_big_order_buyagg_vol_fx', 
    'buy_big_order_act_buyagg_vol_fx', 
    'buy_small_order_buyagg_fx', 
    'buy_small_order_act_buyagg_fx', 
    'buy_small_order_buyagg_cnt_fx', 
    'buy_small_order_buyagg_vol_fx', 
    'buy_small_order_act_buyagg_vol_fx', 
    'sell_big_order_buyagg_fx', 
    'sell_big_order_act_buyagg_fx', 
    'sell_big_order_buyagg_cnt_fx', 
    'sell_big_order_buyagg_vol_fx', 
    'sell_big_order_act_buyagg_vol_fx', 
    'sell_small_order_buyagg_fx', 
    'sell_small_order_act_buyagg_fx', 
    'sell_small_order_buyagg_cnt_fx', 
    'sell_small_order_buyagg_vol_fx', 
    'sell_small_order_act_buyagg_vol_fx', 
    'buy_big_order_sellspa_fx', 
    'buy_big_order_act_sellspa_fx', 
    'buy_big_order_sellspa_cnt_fx', 
    'buy_big_order_sellspa_vol_fx', 
    'buy_big_order_act_sellspa_vol_fx', 
    'buy_small_order_sellspa_fx', 
    'buy_small_order_act_sellspa_fx', 
    'buy_small_order_sellspa_cnt_fx', 
    'buy_small_order_sellspa_vol_fx', 
    'buy_small_order_act_sellspa_vol_fx', 
    'sell_big_order_sellspa_fx', 
    'sell_big_order_act_sellspa_fx', 
    'sell_big_order_sellspa_cnt_fx', 
    'sell_big_order_sellspa_vol_fx', 
    'sell_big_order_act_sellspa_vol_fx', 
    'sell_small_order_sellspa_fx', 
    'sell_small_order_act_sellspa_fx', 
    'sell_small_order_sellspa_cnt_fx', 
    'sell_small_order_sellspa_vol_fx', 
    'sell_small_order_act_sellspa_vol_fx', 
    'buy_big_order_buyspa_fx', 
    'buy_big_order_act_buyspa_fx', 
    'buy_big_order_buyspa_cnt_fx', 
    'buy_big_order_buyspa_vol_fx', 
    'buy_big_order_act_buyspa_vol_fx', 
    'buy_small_order_buyspa_fx', 
    'buy_small_order_act_buyspa_fx', 
    'buy_small_order_buyspa_cnt_fx', 
    'buy_small_order_buyspa_vol_fx', 
    'buy_small_order_act_buyspa_vol_fx', 
    'sell_big_order_buyspa_fx', 
    'sell_big_order_act_buyspa_fx', 
    'sell_big_order_buyspa_cnt_fx', 
    'sell_big_order_buyspa_vol_fx', 
    'sell_big_order_act_buyspa_vol_fx', 
    'sell_small_order_buyspa_fx', 
    'sell_small_order_act_buyspa_fx', 
    'sell_small_order_buyspa_cnt_fx', 
    'sell_small_order_buyspa_vol_fx', 
    'sell_small_order_act_buyspa_vol_fx', 
    'buy_small_order_rt_fx', 
    'buy_small_order_rt_act_fx', 
    'buy_big_order_rt_fx', 
    'buy_big_order_act_rt_fx', 
    'sell_small_order_rt_fx', 
    'sell_small_order_act_rt_fx', 
    'sell_big_order_rt_fx', 
    'sell_big_order_act_rt_fx',
]
# custom_file_path = D.get_custom_info()['root_url']
# today = D.get_trading_dates()[-1]
# yesterday = D.get_trading_dates()[-2]
# factor_ready = np.ndarray
# cross_sectional_file_path = D.get_cyc_info('root_url')
# calc_pool_name = 'FxALPHAV1'
custom_file_path = None
today = None
factor_ready = np.ndarray
yesterday = None
cross_sectional_file_path = None
calc_pool_name = None

def calculating_factor(
        factor_names:list, 
        signal_ready:multiprocessing.Value,    # signal_ready.value +1 后开始计算,下一次signal_ready.value == count + 1时才会继续计算
        pre_update_cnt:int,    # 不存在中间状态时需要更新的历史数据的截面数量
        symbols:list,          # 今日股票代码
        process_id:int,        # 当前进程编号 
        factor_idx:list,       # 单个因子的坐标
        new_index,  
        old_index,
        ):
    global factor_ready
    count = 0
    sub_alpha_data = pd.DataFrame(index=symbols, columns=factor_names, dtype=np.float32)
    # 初始化计算因子的类
    # 假如存在read_date则加载该日的因子类,若无则初始化当日的因子类
    if new_index is not None:
        new_size = len(symbols)
        factor_class = {}
        for name in factor_names:
            try:
                factor_class[name] = pkl.load(open(os.path.join(custom_file_path, str(yesterday), name + '.pkl'), 'rb'))
            except Exception as e:
                print(e)
                print(name)

        error_op = []
        for name in factor_names:
            for ts_obj_name in factor_class[name].__dict__.keys():
                if ts_obj_name.startswith('ts'):
                    try:
                        getattr(factor_class[name], ts_obj_name).adjust(new_size=new_size, old_index=old_index, new_index=new_index)
                    except:
                        error_op.append(ts_obj_name)
                        logger.error(name)
        logger.info(f"{process_id}:loading complete")
        if len(error_op) > 0:
            logger.error(list(set(error_op)))
        # raise Exception("11")
    else:
        factor_class = {name: getattr(fe, name)() for name in factor_names}
        rest_count = pre_update_cnt
        while True:
            if signal_ready.value == count:
                # t0 = datetime.datetime.now()
                for name in factor_class.keys():
                    factor_class[name].update()
                count += 1
                rest_count -= 1
                factor_ready[process_id] = 1
            if rest_count == 0:
                break

    # 当日更新,即每个截面都需要更新
    while True:
        # 收到更新信号,开始逐个更新该进程中的因子
        if signal_ready.value == count:
            for name, alpha_obj in factor_class.items():
                # 计算完的当前截面的因子放入进程间通信容器
                sub_alpha_data[name] = alpha_obj.update()
            # 计算完毕后 因子计数器+1,等待下一次更新信号
            count += 1
            fe.DC.factor_array[:, factor_idx] = sub_alpha_data.values
            factor_ready[process_id] = 1
        if signal_ready.value == - 2:
            #TODO
            if not os.path.exists(os.path.join(custom_file_path, str(today))):
                os.makedirs(os.path.join(custom_file_path, str(today)))
            save_path = os.path.join(custom_file_path, str(today))
            for name, obj in factor_class.items():
                pkl.dump(obj, open(os.path.join(save_path, name + '.pkl'), 'wb'))
            logger.info(f"{process_id}号进程因子类储存完毕")
            factor_ready[process_id] = - 1
            break


def partition_dict(d, n):
    # 将字典的项目按值的降序排序
    items = sorted(d.items(), key=lambda item: item[1], reverse=True)
    # 初始化 n 个分组
    groups = defaultdict(list)
    group_sums = [0] * n
    for key, value in items:
        # 找到当前和最小的组，分配当前项到该组
        min_group_index = np.argmin(group_sums)
        groups[min_group_index].append((key, value))
        group_sums[min_group_index] += value
    return [[y[0] for y in x] for x in groups.values()]


class FactorEngine:
    def __init__(self, symbols, stock_nums, basic_factor_names, factor_names, dtypes, pre_update_cnt, n_jobs, last_symbols=None):
        """
        1. DC类初始化共享内存
        2. 开启n个子进程,每个进程计算若干个因子
        3. 每个子进程中等待DC的完成基础因子接收信号,接收完后开始更新最终因子
        4. 最终因子计算完后,对最终因子矩阵进行填值,所有子进程计算一轮后,通知主进程开始下一轮
        5. 在on_end函数中关闭所有进程与共享内存
        """
        # 进程数 这里还是固定的好
        self.n_jobs = min(n_jobs, len(factor_names))
        # 基础特征 + 基础因子合成的特征
        # DC(DataContainer)根据提供的字段名和字段数量,创建相应的共享内存矩阵
        fe.DC.set_params(
            stock_nums=stock_nums, 
            field_names=basic_factor_names, 
            dtypes=dtypes,
            factor_list=factor_names,
            n_jobs=self.n_jobs,
            )
        # 初始化最终因子DataFrame
        self.alpha_data = pd.DataFrame(index=symbols, columns=factor_names, dtype=np.float32)
        # 修改算符类中的股票数量
        cyc_operator.SIZE = stock_nums
        # 计算历史因子时所需迭代的次数
        self.pre_update_cnt = pre_update_cnt
        # 因子更新次数,初始化为 -1,之后每次收到全部基础因子都+1,子进程中可访问但不修改
        self.signal_ready = Value('i', -1)  
        # alpha因子名(会初始化同名类)
        self.factor_names = factor_names
        # 创建多进程,每个子进程会分配一定的因子计算任务
        self.sub_process = {}
        # # 单个截面中,子进程计算完成数目,更新一次后+1,若epoch_end == n_jobs,表示单截面所有因子计算完毕
        # 将因子按窗口大小分成几组，确保每组的窗口数的均值近似相等(后续改为按因子计算时间规划分组)
        win_map = {name: getattr(fe, name).time_consume for name in factor_names}
        split_group = partition_dict(win_map, self.n_jobs)
        # 因子计算完毕信号 
        global factor_ready
        factor_ready = multiprocessing.Array('i', len(split_group))
        factor_ready = np.frombuffer(factor_ready.get_obj(), dtype='int32').reshape(len(split_group))
        # TODO 按照新旧股票池获取昨日相对于今日的股票池的坐标
        if last_symbols is not None:
            new_index = []
            old_index = []
            for s in last_symbols:
                if s in symbols:
                    new_index.append(symbols.index(s))
                    old_index.append(last_symbols.index(s))
        else:
            new_index = None
            old_index = None
        # 结束信号
        self.end_queue = Queue()
        # 按照上述因子分组开启多进程计算任务
        for idx, group in enumerate(split_group):
            print(len(group))
            factor_idx = [factor_names.index(i) for i in group]
            self.sub_process[idx] = Process(
                target=calculating_factor,
                args=(
                    list(group),       # 该进程计算的因子列表
                    self.signal_ready, # 主进程收到的截面更新完成的信号
                    self.pre_update_cnt,    # 预先计算的截面数量(当没有中间状态文件时,盘前启动需要用到)
                    symbols,           # 当日股票池(在不同日期计算时,当日的universe与昨日的不一样,因此因子类需要将其算符成员做调整)
                    idx,               # 进程编号,用于在监视进程中识别该截面是否计算完成
                    factor_idx,        # 在该进程中计算的最终因子在共享内存表中的坐标(填值用)
                    new_index,         # 昨日票池在今日票池中的坐标
                    old_index,         # 存在于今日票池的股票在昨日票池中的坐标
                    ),
            )
            self.sub_process[idx].start()

    def run(self, data):
        global factor_ready
        # 更新基础因子
        fe.DC.update_shared_memory(data)
        # 更新完基础因子后,更新信号+1,子进程开始计算因子
        self.signal_ready.value += 1
        wait = True
        while wait:
            # 监视进程显示factor_finish = 1,开始给因子赋值
            while factor_ready.min() == 1:
                self.alpha_data.loc[:,:] = fe.DC.factor_array
                factor_ready[:] = 0
                wait = False
                # 结束循环
                break
        return self.alpha_data

    def wait_process_end(self):
        # 给子进程发出信号,计算完毕后储存因子类实例
        self.signal_ready.value = - 2
        # 当收到结束信号次数 = 进程数 时,表明所有因子类实例储存完毕
        wait = True
        while wait:
            while factor_ready.max() == - 1:
                wait = False
                break
        # 关闭所有多进程
        process_id = list(self.sub_process.keys())
        for id in process_id:
            self.sub_process[id].terminate()
            self.sub_process[id].join()
            self.sub_process[id].close()


class FxALPHAV1(CSFactorBase):
    
    # save_time_cut = ['0930','0932','0933','0934','0935','0945','0950','1000','1005','1035','1045','1105','1130','1305','1335','1405','1435','1445','1455','1457']
    def static_init():
        pass

    def get_factor_names():
        # return ['alpha_vboth_fx_0','alpha_vboth_fx_1']
        return [
    'alpha_vboth_fx_0',
    'alpha_vboth_fx_1',
    'alpha_vboth_fx_2',
    'alpha_vboth_fx_3',
    'alpha_vboth_fx_4',
    'alpha_vboth_fx_5',
    'alpha_vboth_fx_8',
    'alpha_vboth_fx_9',
    'alpha_vboth_fx_10',
    'alpha_vboth_fx_11',
    'alpha_vboth_fx_12',
    'alpha_vboth_fx_13',
    'alpha_vboth_fx_15',
    'alpha_vboth_fx_24',
    'alpha_vboth_fx_28',
    'alpha_vboth_fx_29',
    'alpha_vboth_fx_36',
    'alpha_vboth_fx_39',
    'alpha_vboth_fx_40',
    'alpha_vboth_fx_41',
    'alpha_vboth_fx_45',
    'alpha_vboth_fx_47',
    'alpha_vboth_fx_50',
    'alpha_vboth_fx_64',
    'alpha_vboth_fx_70',
    'alpha_vboth_fx_71',
    'alpha_vboth_fx_72',
    'alpha_vboth_fx_73',
    'alpha_vboth_fx_84',
    'alpha_vboth_fx_85',
    'alpha_vboth_fx_88',
    'alpha_vboth_fx_89',
    'alpha_vboth_fx_92',
    'alpha_vboth_fx_100',
    'alpha_vboth_fx_104',
    'alpha_vboth_fx_105',
    'alpha_vboth_fx_111',
    'alpha_vboth_fx_112',
    'alpha_vboth_fx_114',
    'alpha_vboth_fx_119',
    'alpha_vboth_fx_122',
    'alpha_vboth_fx_132',
    'alpha_vboth_fx_133',
    'alpha_vboth_fx_134',
    'alpha_vboth_fx_141',
    'alpha_vboth_fx_154',
    'alpha_vboth_fx_161',
    'alpha_vboth_fx_163',
    'alpha_vboth_fx_164',
    'alpha_vboth_fx_165',
    'alpha_vboth_fx_168',
    'alpha_vboth_fx_170',
    'alpha_vboth_fx_175',
    'alpha_vboth_fx_184',
    'alpha_vboth_fx_186',
    'alpha_vboth_fx_190',
    'alpha_vboth_fx_193',
    'alpha_vboth_fx_195',
    'alpha_vboth_fx_196',
    'alpha_vboth_fx_198',
    'alpha_vboth_fx_207',
    'alpha_vboth_fx_211',
    'alpha_vboth_fx_212',
    'alpha_vboth_fx_220',
    'alpha_vboth_fx_230',
    'alpha_vboth_fx_233',
    'alpha_vboth_fx_243',
    'alpha_vboth_fx_250',
    'alpha_vboth_fx_255',
    'alpha_vboth_fx_259',
    'alpha_vboth_fx_263',
    'alpha_vboth_fx_264',
    'alpha_vboth_fx_265',
    'alpha_vboth_fx_267',
    'alpha_vboth_fx_275',
    'alpha_vboth_fx_277',
    'alpha_vboth_fx_282',
    'alpha_vboth_fx_287',
    'alpha_vboth_fx_292',
    'alpha_vboth_fx_301',
    'alpha_vboth_fx_302',
    'alpha_vboth_fx_303',
    'alpha_vboth_fx_304',
    'alpha_vboth_fx_306',
    'alpha_vboth_fx_312',
    'alpha_vboth_fx_313',
    'alpha_vboth_fx_314',
    'alpha_vboth_fx_316',
    'alpha_vboth_fx_321',
    'alpha_vboth_fx_324',
    'alpha_vboth_fx_325',
    'alpha_vboth_fx_326',
    'alpha_vboth_fx_329',
    'alpha_vboth_fx_332',
    'alpha_vboth_fx_334',
    'alpha_vboth_fx_337',
    'alpha_vboth_fx_341',
    'alpha_vboth_fx_352',
    'alpha_vboth_fx_355',
    'alpha_vboth_fx_356',
    'alpha_vboth_fx_359',
    'alpha_vboth_fx_363',
    'alpha_vboth_fx_367',
    'alpha_vboth_fx_369',
    'alpha_vboth_fx_371',
    'alpha_vboth_fx_375',
    'alpha_vboth_fx_380',
    'alpha_vboth_fx_381',
    'alpha_vboth_fx_389',
    'alpha_vboth_fx_390',
    'alpha_vboth_fx_397',
    'alpha_vboth_fx_400',
    'alpha_vboth_fx_402',
    'alpha_vboth_fx_407',
    'alpha_vboth_fx_412',
    'alpha_vboth_fx_417',
    'alpha_vboth_fx_422',
    'alpha_vboth_fx_428',
    'alpha_vboth_fx_438',
    'alpha_vboth_fx_439',
    'alpha_vboth_fx_440',
    'alpha_vboth_fx_441',
    'alpha_vboth_fx_448',
    'alpha_vboth_fx_452',
    'alpha_vboth_fx_453',
    'alpha_vboth_fx_458',
    'alpha_vboth_fx_459',
    'alpha_vboth_fx_460',
    'alpha_vboth_fx_462',
    'alpha_vboth_fx_469',
    'alpha_vboth_fx_480',
    'alpha_vboth_fx_484',
    'alpha_vboth_fx_486',
    'alpha_vboth_fx_500',
    'alpha_vboth_fx_501',
    'alpha_vboth_fx_506',
    'alpha_vboth_fx_512',
    'alpha_vboth_fx_513',
    'alpha_vboth_fx_517',
    'alpha_vboth_fx_521',
    'alpha_vboth_fx_552',
    'alpha_vboth_fx_553',
    'alpha_vboth_fx_555',
    'alpha_vboth_fx_559',
    'alpha_vboth_fx_560',
    'alpha_vboth_fx_566',
    'alpha_vboth_fx_567',
    'alpha_vboth_fx_569',
    'alpha_vboth_fx_572',
    'alpha_vboth_fx_573',
    'alpha_vboth_fx_574',
    'alpha_vboth_fx_575',
    'alpha_vboth_fx_582',
    'alpha_vboth_fx_584',
    'alpha_vboth_fx_587',
    'alpha_vboth_fx_593',
    'alpha_vboth_fx_595',
    'alpha_vboth_fx_598',
    'alpha_vboth_fx_607',
    'alpha_vboth_fx_610',
    'alpha_vboth_fx_612',
    'alpha_vboth_fx_614',
    'alpha_vboth_fx_617',
    'alpha_vboth_fx_620',
    'alpha_vboth_fx_624',
    'alpha_vboth_fx_628',
    'alpha_vboth_fx_630',
    'alpha_vboth_fx_632',
    'alpha_vboth_fx_635',
    'alpha_vboth_fx_638',
    'alpha_vboth_fx_642',
    'alpha_vboth_fx_644',
    'alpha_vboth_fx_654',
    'alpha_vboth_fx_659',
    'alpha_vboth_fx_661',
    'alpha_vboth_fx_665',
    'alpha_vboth_fx_673',
    'alpha_vboth_fx_689',
    'alpha_vboth_fx_693',
    'alpha_vboth_fx_696',
    'alpha_vboth_fx_700',
    'alpha_vboth_fx_711',
    'alpha_vboth_fx_712',
    'alpha_vboth_fx_714',
    'alpha_vboth_fx_723',
    'alpha_vboth_fx_733',
    'alpha_vboth_fx_734',
    'alpha_vboth_fx_744',
    'alpha_vboth_fx_746',
    'alpha_vboth_fx_754',
    'alpha_vboth_fx_757',
    'alpha_vboth_fx_758',
    'alpha_vboth_fx_764',
    'alpha_vboth_fx_766',
    'alpha_vboth_fx_770',
    'alpha_vboth_fx_771',
    'alpha_vboth_fx_784',
    'alpha_vboth_fx_787',
    'alpha_vboth_fx_790',
    'alpha_vboth_fx_801',
    'alpha_vboth_fx_809',
    'alpha_vboth_fx_815',
    'alpha_vboth_fx_816',
    'alpha_vboth_fx_820',
    'alpha_vboth_fx_830',
    'alpha_vboth_fx_837',
    'alpha_vboth_fx_838',
    'alpha_vboth_fx_858',
    'alpha_vboth_fx_865',
    'alpha_vboth_fx_873',
    'alpha_vboth_fx_884',
    'alpha_vboth_fx_896',
    'alpha_vboth_fx_898',
    'alpha_vboth_fx_899',
    'alpha_vboth_fx_901',
    'alpha_vboth_fx_902',
    'alpha_vboth_fx_903',
    'alpha_vboth_fx_910',
    'alpha_vboth_fx_917',
    'alpha_vboth_fx_921',
    'alpha_vboth_fx_926',
    'alpha_vboth_fx_937',
    'alpha_vboth_fx_941',
    'alpha_vboth_fx_944',
    'alpha_vboth_fx_949',
    'alpha_vboth_fx_959',
    'alpha_vboth_fx_975',
    'alpha_vboth_fx_977',
    'alpha_vboth_fx_984',
    'alpha_vboth_fx_986',
    'alpha_vboth_fx_990',
    'alpha_vboth_fx_996',
    'alpha_vboth_fx_1003',
    'alpha_vboth_fx_1020',
    'alpha_vboth_fx_1022',
    'alpha_vboth_fx_1023',
    'alpha_vboth_fx_1024',
    'alpha_vboth_fx_1034',
    'alpha_vboth_fx_1051',
    'alpha_vboth_fx_1059',
    'alpha_vboth_fx_1064',
    'alpha_vboth_fx_1066',
    'alpha_vboth_fx_1074',
    'alpha_vboth_fx_1076',
    'alpha_vboth_fx_1078',
    'alpha_vboth_fx_1089',
    'alpha_vboth_fx_1101',
    'alpha_vboth_fx_1102',
    'alpha_vboth_fx_1106',
    'alpha_vboth_fx_1114',
    'alpha_vboth_fx_1115',
    'alpha_vboth_fx_1118',
    'alpha_vboth_fx_1124',
    'alpha_vboth_fx_1144',
    'alpha_vboth_fx_1145',
    'alpha_vboth_fx_1158',
    'alpha_vboth_fx_1171',
    'alpha_vboth_fx_1179',
    'alpha_vboth_fx_1185',
    'alpha_vboth_fx_1188',
    'alpha_vboth_fx_1190',
    'alpha_vboth_fx_1209',
    'alpha_vboth_fx_1213',
    'alpha_vboth_fx_1214',
    'alpha_vboth_fx_1218',
    'alpha_vboth_fx_1219',
    'alpha_vboth_fx_1220',
    'alpha_vboth_fx_1222',
    'alpha_vboth_fx_1232',
    'alpha_vboth_fx_1234',
    'alpha_vboth_fx_1235',
    'alpha_vboth_fx_1242',
    'alpha_vboth_fx_1254',
    'alpha_vboth_fx_1261',
    'alpha_vboth_fx_1274',
    'alpha_vboth_fx_1277',
    'alpha_vboth_fx_1282',
    'alpha_vboth_fx_1293',
    'alpha_vboth_fx_1302',
    'alpha_vboth_fx_1309',
    'alpha_vboth_fx_1311',
    'alpha_vboth_fx_1319',
    'alpha_vboth_fx_1324',
    'alpha_vboth_fx_1326',
    'alpha_vboth_fx_1331',
    'alpha_vboth_fx_1336',
    'alpha_vboth_fx_1350',
    'alpha_vboth_fx_1351',
    'alpha_vboth_fx_1352',
    'alpha_vboth_fx_1364',
    'alpha_vboth_fx_1366',
    'alpha_vboth_fx_1370',
    'alpha_vboth_fx_1373',
    'alpha_vboth_fx_1377',
    'alpha_vboth_fx_1381',
    'alpha_vboth_fx_1383',
    'alpha_vboth_fx_1389',
    'alpha_vboth_fx_1403',
    'alpha_vboth_fx_1407',
    'alpha_vboth_fx_1410',
    'alpha_vboth_fx_1415',
    'alpha_vboth_fx_1435',
    'alpha_vboth_fx_1438',
    'alpha_vboth_fx_1440',
    'alpha_vboth_fx_1441',
    'alpha_vboth_fx_1451',
    'alpha_vboth_fx_1452',
    'alpha_vboth_fx_1455',
    'alpha_vboth_fx_1458',
    'alpha_vboth_fx_1466',
    'alpha_vboth_fx_1467',
    'alpha_vboth_fx_1491',
    'alpha_vboth_fx_1505',
    'alpha_vboth_fx_1515',
    'alpha_vboth_fx_1519',
    'alpha_vboth_fx_1521',
    'alpha_vboth_fx_1525',
    'alpha_vboth_fx_1528',
    'alpha_vboth_fx_1552',
    'alpha_vboth_fx_1561',
    'alpha_vboth_fx_1562',
    'alpha_vboth_fx_1566',
    'alpha_vboth_fx_1567',
    'alpha_vboth_fx_1569',
    'alpha_vboth_fx_1574',
    'alpha_vboth_fx_1587',
    'alpha_vboth_fx_1590',
    'alpha_vboth_fx_1592',
    'alpha_vboth_fx_1600',
    'alpha_vboth_fx_1603',
    'alpha_vboth_fx_1606',
    'alpha_vboth_fx_1607',
    'alpha_vboth_fx_1608',
    'alpha_vboth_fx_1610',
    'alpha_vboth_fx_1624',
    'alpha_vboth_fx_1631',
    'alpha_vboth_fx_1641',
    'alpha_vboth_fx_1642',
    'alpha_vboth_fx_1645',
    'alpha_vboth_fx_1666',
    'alpha_vboth_fx_1670',
    'alpha_vboth_fx_1674',
    'alpha_vboth_fx_1679',
    'alpha_vboth_fx_1688',
    'alpha_vboth_fx_1689',
    'alpha_vboth_fx_1691',
    'alpha_vboth_fx_1692',
    'alpha_vboth_fx_1703',
    'alpha_vboth_fx_1704',
    'alpha_vboth_fx_1716',
    'alpha_vboth_fx_1722',
    'alpha_vboth_fx_1723',
    'alpha_vboth_fx_1734',
    'alpha_vboth_fx_1736',
    'alpha_vboth_fx_1760',
    'alpha_vboth_fx_1763',
    'alpha_vboth_fx_1777',
    'alpha_vboth_fx_1818',
    'alpha_vboth_fx_1824',
    'alpha_vboth_fx_1826',
    'alpha_vboth_fx_1827',
    'alpha_vboth_fx_1832',
    'alpha_vboth_fx_1833',
    'alpha_vboth_fx_1854',
    'alpha_vboth_fx_1859',
    'alpha_vboth_fx_1864',
    'alpha_vboth_fx_1868',
    'alpha_vboth_fx_1869',
    'alpha_vboth_fx_1872',
    'alpha_vboth_fx_1874',
    'alpha_vboth_fx_1876',
    'alpha_vboth_fx_1881',
    'alpha_vboth_fx_1887',
    'alpha_vboth_fx_1903',
    'alpha_vboth_fx_1904',
    'alpha_vboth_fx_1908',
    'alpha_vboth_fx_1913',
    'alpha_vboth_fx_1922',
    'alpha_vboth_fx_1923',
    'alpha_vboth_fx_1926',
    'alpha_vboth_fx_1930',
    'alpha_vboth_fx_1939',
    'alpha_vboth_fx_1940',
    'alpha_vboth_fx_1960',
    'alpha_vboth_fx_1963',
    'alpha_vboth_fx_1964',
    'alpha_vboth_fx_1965',
    'alpha_vboth_fx_1971',
    'alpha_vboth_fx_1973',
    'alpha_vboth_fx_1980',
    'alpha_vboth_fx_1993',
    'alpha_vboth_fx_1994',
    'alpha_vboth_fx_1995',
    'alpha_vboth_fx_1996',
    'alpha_vboth_fx_2001',
    'alpha_vboth_fx_2002',
    'alpha_vboth_fx_2003',
    'alpha_vboth_fx_2010',
    'alpha_vboth_fx_2015',
    'alpha_vboth_fx_2024',
    'alpha_vboth_fx_2025',
    'alpha_vboth_fx_2026',
    'alpha_vboth_fx_2028',
    'alpha_vboth_fx_2030',
    'alpha_vboth_fx_2040',
    'alpha_vboth_fx_2043',
    'alpha_vboth_fx_2044',
    'alpha_vboth_fx_2045',
    'alpha_vboth_fx_2046',
    'alpha_vboth_fx_2050',
    'alpha_vboth_fx_2057',
    ]
    def get_sub_factor_pools():
        return ["FxTickFactorPool","FxTradeFactorPool"]

    def __init__(self, symbols) -> None:
        super(FxALPHAV1, self).__init__(symbols)
        # 若无中间状态数据,将读取过去pre_date_cnt日基础特征去计算中间状态数据,并在当日回放结束时储存
        self.pre_date_cnt = self.get_cur_pool_cfg()["pre_date_count"]
        global custom_file_path
        global today
        global yesterday
        global cross_sectional_file_path
        global calc_pool_name

        custom_file_path = self.get_cur_pool_cfg()['custom_url']
        today = self.get_trading_dates()[-1]
        yesterday = self.get_trading_dates()[-2]
        # cross_sectional_file_path = self.get_cur_pool_cfg()['cur_root_dir']
        cross_sectional_file_path = self.get_cur_pool_cfg()['mc_factor_db_dir']

        calc_pool_name = self.get_cur_pool_cfg()['pool_name']


        # 不要对这个东西排序！！
        # symbol_array = D.get_all_symbols()  
        self.current_timestamp = 93000000
        symbol_array = self.get_symbols()  
        self.symbol_array = symbol_array
        # 储存当日票池 千万别改变顺序
        if not os.path.exists(os.path.join(custom_file_path, str(today))):
            os.makedirs(os.path.join(custom_file_path, str(today)))
        pkl.dump(self.symbol_array, open(os.path.join(custom_file_path, str(today), 'symbol.pkl'), 'wb'))
        if not os.path.exists(os.path.join(custom_file_path, str(yesterday), 'symbol.pkl')): # TODO 这里得改成判断所有因子是否存在
            self.last_symbol_array = None
        else:
            self.last_symbol_array = pkl.load(open(os.path.join(custom_file_path, str(yesterday), 'symbol.pkl'), 'rb'))
        
        #TODO 
        # time_cut = deepcopy(D.get_time_cuts())
        # time_cut = deepcopy(self.get_date_time_cuts())
        time_cut = deepcopy(self.get_cs_date_time_cuts())
        # time_cut = [_ for _ in time_cut if str(_//1000)[-6:-2] in FxALPHAV1.save_time_cut]
        
        index = pd.MultiIndex.from_product([time_cut,
                                            symbol_array,
                                            ],
                                           names=['date_time_cut', 'symbol'])

        # index = pd.MultiIndex.from_product([D.get_time_cuts(),
        #                                     symbol_array,
        #                                     ],
        #                                    names=['date_time_cut', 'symbol'])
        # self.n_jobs = D.get_cyc_info('n_jobs')
        self.n_jobs = self.get_cur_pool_cfg()['n_jobs']
        # 初始化最终因子表 index:(date_time_cut,symbol)
        # self.factor_data = pd.DataFrame(index=index, columns=self.calc_factor_names, dtype=np.float64)
        self.factor_data = pd.DataFrame(index=index, columns=type(self).get_factor_names(), dtype=np.float32)
        # 记录当前的因子
        self.timestamp = None  
        # 获取订阅所有因子池的信息
        # self.baisc_factor_names = D.get_sub_basic_factor_names()
        # self.baisc_factor_names = type(self).get_sub_factor_pools()
        self.baisc_factor_names = {}
        for i in range(len(self.get_cur_pool_cfg()['sub_pool_key'])):
            self.baisc_factor_names[self.get_cur_pool_cfg()['sub_pool_key'][i]] = self.get_cur_pool_cfg()['sub_pool_value'][i]

        # 记录收到了多少个人的基础因子
        self.feature_ready = 0
        # 当self.feature_ready == self.pool_num时,才开始计算因子
        self.pool_num = len(self.baisc_factor_names)
        # 记录每个人的特征池的字段名
        self.factor_names_map = {}
        # 初始化基础特征表
        self.data = pd.DataFrame(index=symbol_array, dtype=np.float32).rename_axis('symbol')
        # for pool_name, pool_info in self.baisc_factor_names.items():
        #     self.factor_names_map[pool_name] = [x for x in pool_info['factor_names'] if x != 'is_ok']
        #     self.data.loc[:, self.factor_names_map[pool_name]] = np.nan
        for pool_name in self.baisc_factor_names.keys():
            self.factor_names_map[pool_name] = self.baisc_factor_names[pool_name]
            self.data.loc[:, self.factor_names_map[pool_name]] = np.nan

    # 盘前初始化
    def factor_init(self):
        n = self.pre_date_cnt
        pre_update_rows = int(n * 237)
        # 如果没有昨日信息,那么意味着当天要加载过去信息并算出因子
        if self.last_symbol_array is None:
            # region 加载过去n日数据
            # pre_date = D.get_trading_dates()[-n-1:-1]
            # cap_date = D.get_trading_dates()[-n-2:-2]
            # cap_data = D.get_stock_capital_series(start_date=cap_date[0], end_date=cap_date[-1], fields=['circulating_market_cap'])
            pre_date = self.get_trading_dates()[-n-1:-1]
            cap_date = self.get_trading_dates()[-n-2:-2]
            # cap_data = dapi.get_stock_capital_series(start_date=cap_date[0], end_date=cap_date[-1], fields=['circulating_market_cap'])
            
            # cap_data.columns = [x + '_cyc' for x in cap_data.columns]
            # cap_data.reset_index(inplace=True)
            #TODO
            cap_data = []
            for cap_day in cap_date:
                cap_tmp = pd.read_csv(os.path.join(self.get_cur_pool_cfg()['cap_path'],str(cap_day)[:4],str(cap_day)[4:6],str(cap_day)[6:],'derivative_indicator.csv'))[['TradingDate','SecurityID','CirculatingMarketValue']]
                cap_tmp.columns = ['timestamp','ticker','circulating_market_cap']
                cap_tmp['ticker'] = cap_tmp['ticker'].apply(lambda x:('000000' + str(x))[-6:])
                cap_data.append(cap_tmp)
            cap_data = pd.concat(cap_data,axis = 0)
            
            cap_data.rename(columns={"timestamp": 'date'}, inplace=True)
            cap_data['date'] = cap_data['date'].map(dict(zip(cap_date,pre_date)))
            cap_data['date'] = cap_data['date'].map(str)
            pre_data = []
            logger.info(f"No previous day factor instance file found, therefore loading the past {n} days of data to calculate historical factors")
            # for name, fields in self.factor_names_map.items():
            #     pre_data.append(D.qry_his_basic_factors(
            #         pool=name,
            #         trading_dates=pre_date, 
            #         symbols=self.symbol_array, 
            #         factor_names=fields,
            #         ))
            # for name, fields in self.factor_names_map.items():
            #     pre_data.append(self.load_his_factors(
            #         pool=name,
            #         trading_dates=pre_date, 
            #         symbols=self.symbol_array, 
            #         factor_names=fields,
            #         ))
            
            for prd in pre_date:
                pre_data_sample = []
                for pname in self.factor_names_map.keys():
                    df_tmp = pd.read_parquet(os.path.join(self.get_cur_pool_cfg()['sub_pool_dir'],str(prd)[:4],str(prd)[4:6],str(prd)[6:],'{}.par'.format(pname)))
                    pre_data_sample.append(df_tmp)
                pre_data.append(pd.concat(pre_data_sample, axis = 1))
            # pre_data = pd.concat(pre_data, axis = 1)
            pre_data = pd.concat(pre_data, axis = 0)

            pre_data['date'] = pre_data.index.get_level_values('date_time_cut').astype(str).str.slice(0,8)
            pre_data['ticker'] = pre_data.index.get_level_values('symbol')
            pre_data = pd.merge(pre_data.reset_index(), cap_data, how='left', on=['date', 'ticker'])
            pre_data.drop(['date', 'ticker'], axis=1, inplace=True)
            pre_data.set_index(['date_time_cut', 'symbol'], inplace=True)
            logger.info(f"Loading complete")
            # endregion

            # region 计算衍生基础特征
            pre_data['big_actB_fx'] = (pre_data.actB_4_amt_fx + pre_data.actB_3_amt_fx)
            pre_data['big_actB_cnt_fx'] = (pre_data.actB_4_cnt_fx + pre_data.actB_3_cnt_fx)
            pre_data['big_actB_vol_fx'] = (pre_data.actB_4_vol_fx + pre_data.actB_3_vol_fx)
            pre_data['small_actB_fx'] = (pre_data.actB_1_amt_fx + pre_data.actB_2_amt_fx)
            pre_data['small_actB_cnt_fx'] = (pre_data.actB_1_cnt_fx + pre_data.actB_2_cnt_fx)
            pre_data['small_actB_vol_fx'] = (pre_data.actB_1_vol_fx + pre_data.actB_2_vol_fx)
            pre_data['big_actS_fx'] = (pre_data.actS_4_amt_fx + pre_data.actS_3_amt_fx)
            pre_data['big_actS_cnt_fx'] = (pre_data.actS_4_cnt_fx + pre_data.actS_3_cnt_fx)
            pre_data['big_actS_vol_fx'] = (pre_data.actS_4_vol_fx + pre_data.actS_3_vol_fx)
            pre_data['small_actS_fx'] = (pre_data.actS_1_amt_fx + pre_data.actS_2_amt_fx)
            pre_data['small_actS_cnt_fx'] = (pre_data.actS_1_cnt_fx + pre_data.actS_2_cnt_fx)
            pre_data['small_actS_vol_fx'] = (pre_data.actS_1_vol_fx + pre_data.actS_2_vol_fx)
            pre_data['big_actB_up_fx'] = (pre_data.actB_4_amt_up_fx + pre_data.actB_3_amt_up_fx)
            pre_data['big_actB_cnt_up_fx'] = (pre_data.actB_4_cnt_up_fx + pre_data.actB_3_cnt_up_fx)
            pre_data['big_actB_vol_up_fx'] = (pre_data.actB_4_vol_up_fx + pre_data.actB_3_vol_up_fx)
            pre_data['small_actB_up_fx'] = (pre_data.actB_1_amt_up_fx + pre_data.actB_2_amt_up_fx)
            pre_data['small_actB_cnt_up_fx'] = (pre_data.actB_1_cnt_up_fx + pre_data.actB_2_cnt_up_fx)
            pre_data['small_actB_vol_up_fx'] = (pre_data.actB_1_vol_up_fx + pre_data.actB_2_vol_up_fx)
            pre_data['big_actS_up_fx'] = (pre_data.actS_4_amt_up_fx + pre_data.actS_3_amt_up_fx)
            pre_data['big_actS_cnt_up_fx'] = (pre_data.actS_4_cnt_up_fx + pre_data.actS_3_cnt_up_fx)
            pre_data['big_actS_vol_up_fx'] = (pre_data.actS_4_vol_up_fx + pre_data.actS_3_vol_up_fx)
            pre_data['small_actS_up_fx'] = (pre_data.actS_1_amt_up_fx + pre_data.actS_2_amt_up_fx)
            pre_data['small_actS_cnt_up_fx'] = (pre_data.actS_1_cnt_up_fx + pre_data.actS_2_cnt_up_fx)
            pre_data['small_actS_vol_up_fx'] = (pre_data.actS_1_vol_up_fx + pre_data.actS_2_vol_up_fx)
            pre_data['big_actB_down_fx'] = (pre_data.actB_4_amt_down_fx + pre_data.actB_3_amt_down_fx)
            pre_data['big_actB_cnt_down_fx'] = (pre_data.actB_4_cnt_down_fx + pre_data.actB_3_cnt_down_fx)
            pre_data['big_actB_vol_down_fx'] = (pre_data.actB_4_vol_down_fx + pre_data.actB_3_vol_down_fx)
            pre_data['small_actB_down_fx'] = (pre_data.actB_1_amt_down_fx + pre_data.actB_2_amt_down_fx)
            pre_data['small_actB_cnt_down_fx'] = (pre_data.actB_1_cnt_down_fx + pre_data.actB_2_cnt_down_fx)
            pre_data['small_actB_vol_down_fx'] = (pre_data.actB_1_vol_down_fx + pre_data.actB_2_vol_down_fx)
            pre_data['big_actS_down_fx'] = (pre_data.actS_4_amt_down_fx + pre_data.actS_3_amt_down_fx)
            pre_data['big_actS_cnt_down_fx'] = (pre_data.actS_4_cnt_down_fx + pre_data.actS_3_cnt_down_fx)
            pre_data['big_actS_vol_down_fx'] = (pre_data.actS_4_vol_down_fx + pre_data.actS_3_vol_down_fx)
            pre_data['small_actS_down_fx'] = (pre_data.actS_1_amt_down_fx + pre_data.actS_2_amt_down_fx)
            pre_data['small_actS_cnt_down_fx'] = (pre_data.actS_1_cnt_down_fx + pre_data.actS_2_cnt_down_fx)
            pre_data['small_actS_vol_down_fx'] = (pre_data.actS_1_vol_down_fx + pre_data.actS_2_vol_down_fx)
            pre_data['big_actB_equi_fx'] = (pre_data.actB_4_amt_equi_fx + pre_data.actB_3_amt_equi_fx)
            pre_data['big_actB_cnt_equi_fx'] = (pre_data.actB_4_cnt_equi_fx + pre_data.actB_3_cnt_equi_fx)
            pre_data['big_actB_vol_equi_fx'] = (pre_data.actB_4_vol_equi_fx + pre_data.actB_3_vol_equi_fx)
            pre_data['small_actB_equi_fx'] = (pre_data.actB_1_amt_equi_fx + pre_data.actB_2_amt_equi_fx)
            pre_data['small_actB_cnt_equi_fx'] = (pre_data.actB_1_cnt_equi_fx + pre_data.actB_2_cnt_equi_fx)
            pre_data['small_actB_vol_equi_fx'] = (pre_data.actB_1_vol_equi_fx + pre_data.actB_2_vol_equi_fx)
            pre_data['big_actS_equi_fx'] = (pre_data.actS_4_amt_equi_fx + pre_data.actS_3_amt_equi_fx)
            pre_data['big_actS_cnt_equi_fx'] = (pre_data.actS_4_cnt_equi_fx + pre_data.actS_3_cnt_equi_fx)
            pre_data['big_actS_vol_equi_fx'] = (pre_data.actS_4_vol_equi_fx + pre_data.actS_3_vol_equi_fx)
            pre_data['small_actS_equi_fx'] = (pre_data.actS_1_amt_equi_fx + pre_data.actS_2_amt_equi_fx)
            pre_data['small_actS_cnt_equi_fx'] = (pre_data.actS_1_cnt_equi_fx + pre_data.actS_2_cnt_equi_fx)
            pre_data['small_actS_vol_equi_fx'] = (pre_data.actS_1_vol_equi_fx + pre_data.actS_2_vol_equi_fx)
            pre_data['buy_big_order_tot_fx'] = (pre_data.buy_order_3_fx + pre_data.buy_order_4_fx)
            pre_data['buy_big_order_act_tot_fx'] = (pre_data.buy_order_act_3_fx + pre_data.buy_order_act_4_fx)
            pre_data['buy_big_order_cnt_fx'] = (pre_data.buy_order_cnt_3_fx + pre_data.buy_order_cnt_4_fx)
            pre_data['buy_big_order_act_cnt_fx'] = (pre_data.buy_order_act_cnt_3_fx + pre_data.buy_order_act_cnt_4_fx)
            pre_data['buy_big_order_tot_vol_fx'] = (pre_data.buy_order_vol_3_fx + pre_data.buy_order_vol_4_fx)
            pre_data['buy_big_order_act_vol_fx'] = (pre_data.buy_order_act_vol_3_fx + pre_data.buy_order_act_vol_4_fx)
            pre_data['buy_small_order_tot_fx'] = (pre_data.buy_order_1_fx + pre_data.buy_order_2_fx)
            pre_data['buy_small_order_act_tot_fx'] = (pre_data.buy_order_act_1_fx + pre_data.buy_order_act_2_fx)
            pre_data['buy_small_order_cnt_fx'] = (pre_data.buy_order_cnt_1_fx + pre_data.buy_order_cnt_2_fx)
            pre_data['buy_small_order_act_cnt_fx'] = (pre_data.buy_order_act_cnt_1_fx + pre_data.buy_order_act_cnt_2_fx)
            pre_data['buy_small_order_tot_vol_fx'] = (pre_data.buy_order_vol_1_fx + pre_data.buy_order_vol_2_fx)
            pre_data['buy_small_order_act_vol_fx'] = (pre_data.buy_order_act_vol_1_fx + pre_data.buy_order_act_vol_2_fx)
            pre_data['sell_big_order_tot_fx'] = (pre_data.sell_order_3_fx + pre_data.sell_order_4_fx)
            pre_data['sell_big_order_act_tot_fx'] = (pre_data.sell_order_act_3_fx + pre_data.sell_order_act_4_fx)
            pre_data['sell_big_order_cnt_fx'] = (pre_data.sell_order_cnt_3_fx + pre_data.sell_order_cnt_4_fx)
            pre_data['sell_big_order_act_cnt_fx'] = (pre_data.sell_order_act_cnt_3_fx + pre_data.sell_order_act_cnt_4_fx)
            pre_data['sell_big_order_tot_vol_fx'] = (pre_data.sell_order_vol_3_fx + pre_data.sell_order_vol_4_fx)
            pre_data['sell_big_order_act_vol_fx'] = (pre_data.sell_order_act_vol_3_fx + pre_data.sell_order_act_vol_4_fx)
            pre_data['sell_small_order_tot_fx'] = (pre_data.sell_order_1_fx + pre_data.sell_order_2_fx)
            pre_data['sell_small_order_act_tot_fx'] = (pre_data.sell_order_act_1_fx + pre_data.sell_order_act_2_fx)
            pre_data['sell_small_order_cnt_fx'] = (pre_data.sell_order_cnt_1_fx + pre_data.sell_order_cnt_2_fx)
            pre_data['sell_small_order_act_cnt_fx'] = (pre_data.sell_order_act_cnt_1_fx + pre_data.sell_order_act_cnt_2_fx)
            pre_data['sell_small_order_tot_vol_fx'] = (pre_data.sell_order_vol_1_fx + pre_data.sell_order_vol_2_fx)
            pre_data['sell_small_order_act_vol_fx'] = (pre_data.sell_order_act_vol_1_fx + pre_data.sell_order_act_vol_2_fx)
            pre_data['buy_big_order_up_fx'] = (pre_data.buy_order_up_3_fx + pre_data.buy_order_up_4_fx)
            pre_data['buy_big_order_act_up_fx'] = (pre_data.buy_order_up_act_3_fx + pre_data.buy_order_up_act_4_fx)
            pre_data['buy_big_order_up_cnt_fx'] = (pre_data.buy_order_up_cnt_3_fx + pre_data.buy_order_up_cnt_4_fx)
            pre_data['buy_big_order_up_vol_fx'] = (pre_data.buy_order_up_vol_3_fx + pre_data.buy_order_up_vol_4_fx)
            pre_data['buy_big_order_act_up_vol_fx'] = (pre_data.buy_order_up_act_vol_3_fx + pre_data.buy_order_up_act_vol_4_fx)
            pre_data['buy_small_order_up_fx'] = (pre_data.buy_order_up_1_fx + pre_data.buy_order_up_2_fx)
            pre_data['buy_small_order_act_up_fx'] = (pre_data.buy_order_up_act_1_fx + pre_data.buy_order_up_act_2_fx)
            pre_data['buy_small_order_up_cnt_fx'] = (pre_data.buy_order_up_cnt_1_fx + pre_data.buy_order_up_cnt_2_fx)
            pre_data['buy_small_order_up_vol_fx'] = (pre_data.buy_order_up_vol_1_fx + pre_data.buy_order_up_vol_2_fx)
            pre_data['buy_small_order_act_up_vol_fx'] = (pre_data.buy_order_up_act_vol_1_fx + pre_data.buy_order_up_act_vol_2_fx)
            pre_data['sell_big_order_up_fx'] = (pre_data.sell_order_up_3_fx + pre_data.sell_order_up_4_fx)
            pre_data['sell_big_order_act_up_fx'] = (pre_data.sell_order_up_act_3_fx + pre_data.sell_order_up_act_4_fx)
            pre_data['sell_big_order_up_cnt_fx'] = (pre_data.sell_order_up_cnt_3_fx + pre_data.sell_order_up_cnt_4_fx)
            pre_data['sell_big_order_up_vol_fx'] = (pre_data.sell_order_up_vol_3_fx + pre_data.sell_order_up_vol_4_fx)
            pre_data['sell_big_order_act_up_vol_fx'] = (pre_data.sell_order_up_act_vol_3_fx + pre_data.sell_order_up_act_vol_4_fx)
            pre_data['sell_small_order_up_fx'] = (pre_data.sell_order_up_1_fx + pre_data.sell_order_up_2_fx)
            pre_data['sell_small_order_act_up_fx'] = (pre_data.sell_order_up_act_1_fx + pre_data.sell_order_up_act_2_fx)
            pre_data['sell_small_order_up_cnt_fx'] = (pre_data.sell_order_up_cnt_1_fx + pre_data.sell_order_up_cnt_2_fx)
            pre_data['sell_small_order_up_vol_fx'] = (pre_data.sell_order_up_vol_1_fx + pre_data.sell_order_up_vol_2_fx)
            pre_data['sell_small_order_act_up_vol_fx'] = (pre_data.sell_order_up_act_vol_1_fx + pre_data.sell_order_up_act_vol_2_fx)
            pre_data['buy_big_order_down_fx'] = (pre_data.buy_order_down_3_fx + pre_data.buy_order_down_4_fx)
            pre_data['buy_big_order_act_down_fx'] = (pre_data.buy_order_down_act_3_fx + pre_data.buy_order_down_act_4_fx)
            pre_data['buy_big_order_down_cnt_fx'] = (pre_data.buy_order_down_cnt_3_fx + pre_data.buy_order_down_cnt_4_fx)
            pre_data['buy_big_order_down_vol_fx'] = (pre_data.buy_order_down_vol_3_fx + pre_data.buy_order_down_vol_4_fx)
            pre_data['buy_big_order_act_down_vol_fx'] = (pre_data.buy_order_down_act_vol_3_fx + pre_data.buy_order_down_act_vol_4_fx)
            pre_data['buy_small_order_down_fx'] = (pre_data.buy_order_down_1_fx + pre_data.buy_order_down_2_fx)
            pre_data['buy_small_order_act_down_fx'] = (pre_data.buy_order_down_act_1_fx + pre_data.buy_order_down_act_2_fx)
            pre_data['buy_small_order_down_cnt_fx'] = (pre_data.buy_order_down_cnt_1_fx + pre_data.buy_order_down_cnt_2_fx)
            pre_data['buy_small_order_down_vol_fx'] = (pre_data.buy_order_down_vol_1_fx + pre_data.buy_order_down_vol_2_fx)
            pre_data['buy_small_order_act_down_vol_fx'] = (pre_data.buy_order_down_act_vol_1_fx + pre_data.buy_order_down_act_vol_2_fx)
            pre_data['sell_big_order_down_fx'] = (pre_data.sell_order_down_3_fx + pre_data.sell_order_down_4_fx)
            pre_data['sell_big_order_act_down_fx'] = (pre_data.sell_order_down_act_3_fx + pre_data.sell_order_down_act_4_fx)
            pre_data['sell_big_order_down_cnt_fx'] = (pre_data.sell_order_down_cnt_3_fx + pre_data.sell_order_down_cnt_4_fx)
            pre_data['sell_big_order_down_vol_fx'] = (pre_data.sell_order_down_vol_3_fx + pre_data.sell_order_down_vol_4_fx)
            pre_data['sell_big_order_act_down_vol_fx'] = (pre_data.sell_order_down_act_vol_3_fx + pre_data.sell_order_down_act_vol_4_fx)
            pre_data['sell_small_order_down_fx'] = (pre_data.sell_order_down_1_fx + pre_data.sell_order_down_2_fx)
            pre_data['sell_small_order_act_down_fx'] = (pre_data.sell_order_down_act_1_fx + pre_data.sell_order_down_act_2_fx)
            pre_data['sell_small_order_down_cnt_fx'] = (pre_data.sell_order_down_cnt_1_fx + pre_data.sell_order_down_cnt_2_fx)
            pre_data['sell_small_order_down_vol_fx'] = (pre_data.sell_order_down_vol_1_fx + pre_data.sell_order_down_vol_2_fx)
            pre_data['sell_small_order_act_down_vol_fx'] = (pre_data.sell_order_down_act_vol_1_fx + pre_data.sell_order_down_act_vol_2_fx)
            pre_data['buy_big_order_spreadup_fx'] = (pre_data.buy_order_spreadup_3_fx + pre_data.buy_order_spreadup_4_fx)
            pre_data['buy_big_order_act_spreadup_fx'] = (pre_data.buy_order_spreadup_act_3_fx + pre_data.buy_order_spreadup_act_4_fx)
            pre_data['buy_big_order_spreadup_cnt_fx'] = (pre_data.buy_order_spreadup_cnt_3_fx + pre_data.buy_order_spreadup_cnt_4_fx)
            pre_data['buy_big_order_spreadup_vol_fx'] = (pre_data.buy_order_spreadup_vol_3_fx + pre_data.buy_order_spreadup_vol_4_fx)
            pre_data['buy_big_order_act_spreadup_vol_fx'] = (pre_data.buy_order_spreadup_act_vol_3_fx + pre_data.buy_order_spreadup_act_vol_4_fx)
            pre_data['buy_small_order_spreadup_fx'] = (pre_data.buy_order_spreadup_1_fx + pre_data.buy_order_spreadup_2_fx)
            pre_data['buy_small_order_act_spreadup_fx'] = (pre_data.buy_order_spreadup_act_1_fx + pre_data.buy_order_spreadup_act_2_fx)
            pre_data['buy_small_order_spreadup_cnt_fx'] = (pre_data.buy_order_spreadup_cnt_1_fx + pre_data.buy_order_spreadup_cnt_2_fx)
            pre_data['buy_small_order_spreadup_vol_fx'] = (pre_data.buy_order_spreadup_vol_1_fx + pre_data.buy_order_spreadup_vol_2_fx)
            pre_data['buy_small_order_act_spreadup_vol_fx'] = (pre_data.buy_order_spreadup_act_vol_1_fx + pre_data.buy_order_spreadup_act_vol_2_fx)
            pre_data['sell_big_order_spreadup_fx'] = (pre_data.sell_order_spreadup_3_fx + pre_data.sell_order_spreadup_4_fx)
            pre_data['sell_big_order_act_spreadup_fx'] = (pre_data.sell_order_spreadup_act_3_fx + pre_data.sell_order_spreadup_act_4_fx)
            pre_data['sell_big_order_spreadup_cnt_fx'] = (pre_data.sell_order_spreadup_cnt_3_fx + pre_data.sell_order_spreadup_cnt_4_fx)
            pre_data['sell_big_order_spreadup_vol_fx'] = (pre_data.sell_order_spreadup_vol_3_fx + pre_data.sell_order_spreadup_vol_4_fx)
            pre_data['sell_big_order_act_spreadup_vol_fx'] = (pre_data.sell_order_spreadup_act_vol_3_fx + pre_data.sell_order_spreadup_act_vol_4_fx)
            pre_data['sell_small_order_spreadup_fx'] = (pre_data.sell_order_spreadup_1_fx + pre_data.sell_order_spreadup_2_fx)
            pre_data['sell_small_order_act_spreadup_fx'] = (pre_data.sell_order_spreadup_act_1_fx + pre_data.sell_order_spreadup_act_2_fx)
            pre_data['sell_small_order_spreadup_cnt_fx'] = (pre_data.sell_order_spreadup_cnt_1_fx + pre_data.sell_order_spreadup_cnt_2_fx)
            pre_data['sell_small_order_spreadup_vol_fx'] = (pre_data.sell_order_spreadup_vol_1_fx + pre_data.sell_order_spreadup_vol_2_fx)
            pre_data['sell_small_order_act_spreadup_vol_fx'] = (pre_data.sell_order_spreadup_act_vol_1_fx + pre_data.sell_order_spreadup_act_vol_2_fx)
            pre_data['buy_big_order_spreaddown_fx'] = (pre_data.buy_order_spreaddown_3_fx + pre_data.buy_order_spreaddown_4_fx)
            pre_data['buy_big_order_act_spreaddown_fx'] = (pre_data.buy_order_spreaddown_act_3_fx + pre_data.buy_order_spreaddown_act_4_fx)
            pre_data['buy_big_order_spreaddown_cnt_fx'] = (pre_data.buy_order_spreaddown_cnt_3_fx + pre_data.buy_order_spreaddown_cnt_4_fx)
            pre_data['buy_big_order_spreaddown_vol_fx'] = (pre_data.buy_order_spreaddown_vol_3_fx + pre_data.buy_order_spreaddown_vol_4_fx)
            pre_data['buy_big_order_act_spreaddown_vol_fx'] = (pre_data.buy_order_spreaddown_vol_act_3_fx + pre_data.buy_order_spreaddown_vol_act_4_fx)
            pre_data['buy_small_order_spreaddown_fx'] = (pre_data.buy_order_spreaddown_1_fx + pre_data.buy_order_spreaddown_2_fx)
            pre_data['buy_small_order_act_spreaddown_fx'] = (pre_data.buy_order_spreaddown_act_1_fx + pre_data.buy_order_spreaddown_act_2_fx)
            pre_data['buy_small_order_spreaddown_cnt_fx'] = (pre_data.buy_order_spreaddown_cnt_1_fx + pre_data.buy_order_spreaddown_cnt_2_fx)
            pre_data['buy_small_order_spreaddown_vol_fx'] = (pre_data.buy_order_spreaddown_vol_1_fx + pre_data.buy_order_spreaddown_vol_2_fx)
            pre_data['buy_small_order_act_spreaddown_vol_fx'] = (pre_data.buy_order_spreaddown_vol_act_1_fx + pre_data.buy_order_spreaddown_vol_act_2_fx)
            pre_data['sell_big_order_spreaddown_fx'] = (pre_data.sell_order_spreaddown_3_fx + pre_data.sell_order_spreaddown_4_fx)
            pre_data['sell_big_order_act_spreaddown_fx'] = (pre_data.sell_order_spreaddown_act_3_fx + pre_data.sell_order_spreaddown_act_4_fx)
            pre_data['sell_big_order_spreaddown_cnt_fx'] = (pre_data.sell_order_spreaddown_cnt_3_fx + pre_data.sell_order_spreaddown_cnt_4_fx)
            pre_data['sell_big_order_spreaddown_vol_fx'] = (pre_data.sell_order_spreaddown_vol_3_fx + pre_data.sell_order_spreaddown_vol_4_fx)
            pre_data['sell_big_order_act_spreaddown_vol_fx'] = (pre_data.sell_order_spreaddown_vol_act_3_fx + pre_data.sell_order_spreaddown_vol_act_4_fx)
            pre_data['sell_small_order_spreaddown_fx'] = (pre_data.sell_order_spreaddown_1_fx + pre_data.sell_order_spreaddown_2_fx)
            pre_data['sell_small_order_act_spreaddown_fx'] = (pre_data.sell_order_spreaddown_act_1_fx + pre_data.sell_order_spreaddown_act_2_fx)
            pre_data['sell_small_order_spreaddown_cnt_fx'] = (pre_data.sell_order_spreaddown_cnt_1_fx + pre_data.sell_order_spreaddown_cnt_2_fx)
            pre_data['sell_small_order_spreaddown_vol_fx'] = (pre_data.sell_order_spreaddown_vol_1_fx + pre_data.sell_order_spreaddown_vol_2_fx)
            pre_data['sell_small_order_act_spreaddown_vol_fx'] = (pre_data.sell_order_spreaddown_vol_act_1_fx + pre_data.sell_order_spreaddown_vol_act_2_fx)
            pre_data['buy_big_order_sellagg_fx'] = (pre_data.buy_order_sellagg_3_fx + pre_data.buy_order_sellagg_4_fx)
            pre_data['buy_big_order_act_sellagg_fx'] = (pre_data.buy_order_sellagg_act_3_fx + pre_data.buy_order_sellagg_act_4_fx)
            pre_data['buy_big_order_sellagg_cnt_fx'] = (pre_data.buy_order_sellagg_cnt_3_fx + pre_data.buy_order_sellagg_cnt_4_fx)
            pre_data['buy_big_order_sellagg_vol_fx'] = (pre_data.buy_order_sellagg_vol_3_fx + pre_data.buy_order_sellagg_vol_4_fx)
            pre_data['buy_big_order_act_sellagg_vol_fx'] = (pre_data.buy_order_sellagg_act_vol_3_fx + pre_data.buy_order_sellagg_act_vol_4_fx)
            pre_data['buy_small_order_sellagg_fx'] = (pre_data.buy_order_sellagg_1_fx + pre_data.buy_order_sellagg_2_fx)
            pre_data['buy_small_order_act_sellagg_fx'] = (pre_data.buy_order_sellagg_act_1_fx + pre_data.buy_order_sellagg_act_2_fx)
            pre_data['buy_small_order_sellagg_cnt_fx'] = (pre_data.buy_order_sellagg_cnt_1_fx + pre_data.buy_order_sellagg_cnt_2_fx)
            pre_data['buy_small_order_sellagg_vol_fx'] = (pre_data.buy_order_sellagg_vol_1_fx + pre_data.buy_order_sellagg_vol_2_fx)
            pre_data['buy_small_order_act_sellagg_vol_fx'] = (pre_data.buy_order_sellagg_act_vol_1_fx + pre_data.buy_order_sellagg_act_vol_2_fx)
            pre_data['sell_big_order_sellagg_fx'] = (pre_data.sell_order_sellagg_3_fx + pre_data.sell_order_sellagg_4_fx)
            pre_data['sell_big_order_act_sellagg_fx'] = (pre_data.sell_order_sellagg_act_3_fx + pre_data.sell_order_sellagg_act_4_fx)
            pre_data['sell_big_order_sellagg_cnt_fx'] = (pre_data.sell_order_sellagg_cnt_3_fx + pre_data.sell_order_sellagg_cnt_4_fx)
            pre_data['sell_big_order_sellagg_vol_fx'] = (pre_data.sell_order_sellagg_vol_3_fx + pre_data.sell_order_sellagg_vol_4_fx)
            pre_data['sell_big_order_act_sellagg_vol_fx'] = (pre_data.sell_order_sellagg_act_vol_3_fx + pre_data.sell_order_sellagg_act_vol_4_fx)
            pre_data['sell_small_order_sellagg_fx'] = (pre_data.sell_order_sellagg_1_fx + pre_data.sell_order_sellagg_2_fx)
            pre_data['sell_small_order_act_sellagg_fx'] = (pre_data.sell_order_sellagg_act_1_fx + pre_data.sell_order_sellagg_act_2_fx)
            pre_data['sell_small_order_sellagg_cnt_fx'] = (pre_data.sell_order_sellagg_cnt_1_fx + pre_data.sell_order_sellagg_cnt_2_fx)
            pre_data['sell_small_order_sellagg_vol_fx'] = (pre_data.sell_order_sellagg_vol_1_fx + pre_data.sell_order_sellagg_vol_2_fx)
            pre_data['sell_small_order_act_sellagg_vol_fx'] = (pre_data.sell_order_sellagg_act_vol_1_fx + pre_data.sell_order_sellagg_act_vol_2_fx)
            pre_data['buy_big_order_buyagg_fx'] = (pre_data.buy_order_buyagg_3_fx + pre_data.buy_order_buyagg_4_fx)
            pre_data['buy_big_order_act_buyagg_fx'] = (pre_data.buy_order_buyagg_act_3_fx + pre_data.buy_order_buyagg_act_4_fx)
            pre_data['buy_big_order_buyagg_cnt_fx'] = (pre_data.buy_order_buyagg_cnt_3_fx + pre_data.buy_order_buyagg_cnt_4_fx)
            pre_data['buy_big_order_buyagg_vol_fx'] = (pre_data.buy_order_buyagg_vol_3_fx + pre_data.buy_order_buyagg_vol_4_fx)
            pre_data['buy_big_order_act_buyagg_vol_fx'] = (pre_data.buy_order_buyagg_act_vol_3_fx + pre_data.buy_order_buyagg_act_vol_4_fx)
            pre_data['buy_small_order_buyagg_fx'] = (pre_data.buy_order_buyagg_1_fx + pre_data.buy_order_buyagg_2_fx)
            pre_data['buy_small_order_act_buyagg_fx'] = (pre_data.buy_order_buyagg_act_1_fx + pre_data.buy_order_buyagg_act_2_fx)
            pre_data['buy_small_order_buyagg_cnt_fx'] = (pre_data.buy_order_buyagg_cnt_1_fx + pre_data.buy_order_buyagg_cnt_2_fx)
            pre_data['buy_small_order_buyagg_vol_fx'] = (pre_data.buy_order_buyagg_vol_1_fx + pre_data.buy_order_buyagg_vol_2_fx)
            pre_data['buy_small_order_act_buyagg_vol_fx'] = (pre_data.buy_order_buyagg_act_vol_1_fx + pre_data.buy_order_buyagg_act_vol_2_fx)
            pre_data['sell_big_order_buyagg_fx'] = (pre_data.sell_order_buyagg_3_fx + pre_data.sell_order_buyagg_4_fx)
            pre_data['sell_big_order_act_buyagg_fx'] = (pre_data.sell_order_buyagg_act_3_fx + pre_data.sell_order_buyagg_act_4_fx)
            pre_data['sell_big_order_buyagg_cnt_fx'] = (pre_data.sell_order_buyagg_cnt_3_fx + pre_data.sell_order_buyagg_cnt_4_fx)
            pre_data['sell_big_order_buyagg_vol_fx'] = (pre_data.sell_order_buyagg_vol_3_fx + pre_data.sell_order_buyagg_vol_4_fx)
            pre_data['sell_big_order_act_buyagg_vol_fx'] = (pre_data.sell_order_buyagg_act_vol_3_fx + pre_data.sell_order_buyagg_act_vol_4_fx)
            pre_data['sell_small_order_buyagg_fx'] = (pre_data.sell_order_buyagg_1_fx + pre_data.sell_order_buyagg_2_fx)
            pre_data['sell_small_order_act_buyagg_fx'] = (pre_data.sell_order_buyagg_act_1_fx + pre_data.sell_order_buyagg_act_2_fx)
            pre_data['sell_small_order_buyagg_cnt_fx'] = (pre_data.sell_order_buyagg_cnt_1_fx + pre_data.sell_order_buyagg_cnt_2_fx)
            pre_data['sell_small_order_buyagg_vol_fx'] = (pre_data.sell_order_buyagg_vol_1_fx + pre_data.sell_order_buyagg_vol_2_fx)
            pre_data['sell_small_order_act_buyagg_vol_fx'] = (pre_data.sell_order_buyagg_act_vol_1_fx + pre_data.sell_order_buyagg_act_vol_2_fx)
            pre_data['buy_big_order_sellspa_fx'] = (pre_data.buy_order_sellspa_3_fx + pre_data.buy_order_sellspa_4_fx)
            pre_data['buy_big_order_act_sellspa_fx'] = (pre_data.buy_order_sellspa_act_3_fx + pre_data.buy_order_sellspa_act_4_fx)
            pre_data['buy_big_order_sellspa_cnt_fx'] = (pre_data.buy_order_sellspa_cnt_3_fx + pre_data.buy_order_sellspa_cnt_4_fx)
            pre_data['buy_big_order_sellspa_vol_fx'] = (pre_data.buy_order_sellspa_vol_3_fx + pre_data.buy_order_sellspa_vol_4_fx)
            pre_data['buy_big_order_act_sellspa_vol_fx'] = (pre_data.buy_order_sellspa_act_vol_3_fx + pre_data.buy_order_sellspa_act_vol_4_fx)
            pre_data['buy_small_order_sellspa_fx'] = (pre_data.buy_order_sellspa_1_fx + pre_data.buy_order_sellspa_2_fx)
            pre_data['buy_small_order_act_sellspa_fx'] = (pre_data.buy_order_sellspa_act_1_fx + pre_data.buy_order_sellspa_act_2_fx)
            pre_data['buy_small_order_sellspa_cnt_fx'] = (pre_data.buy_order_sellspa_cnt_1_fx + pre_data.buy_order_sellspa_cnt_2_fx)
            pre_data['buy_small_order_sellspa_vol_fx'] = (pre_data.buy_order_sellspa_vol_1_fx + pre_data.buy_order_sellspa_vol_2_fx)
            pre_data['buy_small_order_act_sellspa_vol_fx'] = (pre_data.buy_order_sellspa_act_vol_1_fx + pre_data.buy_order_sellspa_act_vol_2_fx)
            pre_data['sell_big_order_sellspa_fx'] = (pre_data.sell_order_sellspa_3_fx + pre_data.sell_order_sellspa_4_fx)
            pre_data['sell_big_order_act_sellspa_fx'] = (pre_data.sell_order_sellspa_act_3_fx + pre_data.sell_order_sellspa_act_4_fx)
            pre_data['sell_big_order_sellspa_cnt_fx'] = (pre_data.sell_order_sellspa_cnt_3_fx + pre_data.sell_order_sellspa_cnt_4_fx)
            pre_data['sell_big_order_sellspa_vol_fx'] = (pre_data.sell_order_sellspa_vol_3_fx + pre_data.sell_order_sellspa_vol_4_fx)
            pre_data['sell_big_order_act_sellspa_vol_fx'] = (pre_data.sell_order_sellspa_act_vol_3_fx + pre_data.sell_order_sellspa_act_vol_4_fx)
            pre_data['sell_small_order_sellspa_fx'] = (pre_data.sell_order_sellspa_1_fx + pre_data.sell_order_sellspa_2_fx)
            pre_data['sell_small_order_act_sellspa_fx'] = (pre_data.sell_order_sellspa_act_1_fx + pre_data.sell_order_sellspa_act_2_fx)
            pre_data['sell_small_order_sellspa_cnt_fx'] = (pre_data.sell_order_sellspa_cnt_1_fx + pre_data.sell_order_sellspa_cnt_2_fx)
            pre_data['sell_small_order_sellspa_vol_fx'] = (pre_data.sell_order_sellspa_vol_1_fx + pre_data.sell_order_sellspa_vol_2_fx)
            pre_data['sell_small_order_act_sellspa_vol_fx'] = (pre_data.sell_order_sellspa_act_vol_1_fx + pre_data.sell_order_sellspa_act_vol_2_fx)
            pre_data['buy_big_order_buyspa_fx'] = (pre_data.buy_order_buyspa_3_fx + pre_data.buy_order_buyspa_4_fx)
            pre_data['buy_big_order_act_buyspa_fx'] = (pre_data.buy_order_buyspa_act_3_fx + pre_data.buy_order_buyspa_act_4_fx)
            pre_data['buy_big_order_buyspa_cnt_fx'] = (pre_data.buy_order_buyspa_cnt_3_fx + pre_data.buy_order_buyspa_cnt_4_fx)
            pre_data['buy_big_order_buyspa_vol_fx'] = (pre_data.buy_order_buyspa_vol_3_fx + pre_data.buy_order_buyspa_vol_4_fx)
            pre_data['buy_big_order_act_buyspa_vol_fx'] = (pre_data.buy_order_buyspa_act_vol_3_fx + pre_data.buy_order_buyspa_act_vol_4_fx)
            pre_data['buy_small_order_buyspa_fx'] = (pre_data.buy_order_buyspa_1_fx + pre_data.buy_order_buyspa_2_fx)
            pre_data['buy_small_order_act_buyspa_fx'] = (pre_data.buy_order_buyspa_act_1_fx + pre_data.buy_order_buyspa_act_2_fx)
            pre_data['buy_small_order_buyspa_cnt_fx'] = (pre_data.buy_order_buyspa_cnt_1_fx + pre_data.buy_order_buyspa_cnt_2_fx)
            pre_data['buy_small_order_buyspa_vol_fx'] = (pre_data.buy_order_buyspa_vol_1_fx + pre_data.buy_order_buyspa_vol_2_fx)
            pre_data['buy_small_order_act_buyspa_vol_fx'] = (pre_data.buy_order_buyspa_act_vol_1_fx + pre_data.buy_order_buyspa_act_vol_2_fx)
            pre_data['sell_big_order_buyspa_fx'] = (pre_data.sell_order_buyspa_3_fx + pre_data.sell_order_buyspa_4_fx)
            pre_data['sell_big_order_act_buyspa_fx'] = (pre_data.sell_order_buyspa_act_3_fx + pre_data.sell_order_buyspa_act_4_fx)
            pre_data['sell_big_order_buyspa_cnt_fx'] = (pre_data.sell_order_buyspa_cnt_3_fx + pre_data.sell_order_buyspa_cnt_4_fx)
            pre_data['sell_big_order_buyspa_vol_fx'] = (pre_data.sell_order_buyspa_vol_3_fx + pre_data.sell_order_buyspa_vol_4_fx)
            pre_data['sell_big_order_act_buyspa_vol_fx'] = (pre_data.sell_order_buyspa_act_vol_3_fx + pre_data.sell_order_buyspa_act_vol_4_fx)
            pre_data['sell_small_order_buyspa_fx'] = (pre_data.sell_order_buyspa_1_fx + pre_data.sell_order_buyspa_2_fx)
            pre_data['sell_small_order_act_buyspa_fx'] = (pre_data.sell_order_buyspa_act_1_fx + pre_data.sell_order_buyspa_act_2_fx)
            pre_data['sell_small_order_buyspa_cnt_fx'] = (pre_data.sell_order_buyspa_cnt_1_fx + pre_data.sell_order_buyspa_cnt_2_fx)
            pre_data['sell_small_order_buyspa_vol_fx'] = (pre_data.sell_order_buyspa_vol_1_fx + pre_data.sell_order_buyspa_vol_2_fx)
            pre_data['sell_small_order_act_buyspa_vol_fx'] = (pre_data.sell_order_buyspa_act_vol_1_fx + pre_data.sell_order_buyspa_act_vol_2_fx)
            pre_data['buy_small_order_rt_fx'] = (pre_data.buy_order_rt_1_fx + pre_data.buy_order_rt_2_fx)
            pre_data['buy_small_order_rt_act_fx'] = (pre_data.buy_order_rt_act_1_fx + pre_data.buy_order_rt_act_2_fx)
            pre_data['buy_big_order_rt_fx'] = (pre_data.buy_order_rt_3_fx + pre_data.buy_order_rt_4_fx)
            pre_data['buy_big_order_act_rt_fx'] = (pre_data.buy_order_rt_act_3_fx + pre_data.buy_order_rt_act_4_fx)
            pre_data['sell_small_order_rt_fx'] = (pre_data.sell_order_rt_1_fx + pre_data.sell_order_rt_2_fx)
            pre_data['sell_small_order_act_rt_fx'] = (pre_data.sell_order_rt_act_1_fx + pre_data.sell_order_rt_act_2_fx)
            pre_data['sell_big_order_rt_fx'] = (pre_data.sell_order_rt_3_fx + pre_data.sell_order_rt_4_fx)
            pre_data['sell_big_order_act_rt_fx'] = (pre_data.sell_order_rt_act_3_fx + pre_data.sell_order_rt_act_4_fx)
            
            # region 对齐股票数量
            _temp_time_cut = pre_data.index.get_level_values('date_time_cut').unique()
            tot_index = pd.MultiIndex.from_product(
                [_temp_time_cut, list(self.symbol_array)], 
                names=['date_time_cut', 'symbol']
                )
            del_index = set(pre_data.index).difference(set(tot_index))
            add_index = set(tot_index).difference(set(pre_data.index))
            if len(del_index) > 0:
                pre_data.drop(list(del_index), axis=0, inplace=True)
            if len(add_index) > 0:
                _add_data = pd.DataFrame(index=pd.Index(add_index), columns=pre_data.columns).rename_axis(pre_data.index.names)
                pre_data = pd.concat([pre_data, _add_data], axis=0, join='outer')
            pre_data.sort_index(level=['date_time_cut', 'symbol'], ascending=[True, True], inplace=True)
            # TODO 检查这里是否能完全按照self.symbol_array排序
            pre_data = pre_data.loc[(slice(None),self.symbol_array),:]
            # endregion
        
            # region 初始化因子计算类
            basic_fator_fields = pre_data.columns
            self.alpha_engine = FactorEngine(
                stock_nums=len(self.symbol_array),      # 股票数量
                basic_factor_names=basic_fator_fields,  # 基础因子名
                factor_names=type(self).get_factor_names(),    # 最终因子名
                dtypes=pre_data.dtypes.tolist(),        # 基础因子类型
                symbols=self.symbol_array,              # 当日股票池
                # n_jobs=D.get_cyc_info('n_jobs'),        # 多进程数量
                n_jobs=self.get_cur_pool_cfg()['n_jobs'],
                # n_jobs = 1,
                pre_update_cnt=pre_update_rows,         # 预先更新的样本行数
                last_symbols=self.last_symbol_array,    # 昨日股票池
                )
            # 计算当日之前的因子
            # TODO 这里得注意股票排序是否正确
            for date_time_cut, sub_df in pre_data.groupby('date_time_cut'):
                # 这里自己滤掉了非连续竞价截面
                dt = str(date_time_cut)[8:12]
                if dt in ['1300', '0930', '1458', '1459', '1500']: continue
                # logger.info(f"calculating of cross-sectional factors at {date_time_cut}")
                # 这里有返回值,不过对自己来说没啥用
                self.alpha_engine.run(data=sub_df[basic_fator_fields])
                logger.info(f"calculate finished at {date_time_cut}")
            # endregion

        else:
            # region 初始化因子计算类 且不计算历史因子
            basic_fator_fields = []
            for _, fields in self.factor_names_map.items():
                basic_fator_fields.extend(fields)
            basic_fator_fields.extend(dev_features)
            basic_fator_fields.append('circulating_market_cap')
            dtypes = [np.float32 for _ in basic_fator_fields]
            self.alpha_engine = FactorEngine(
                stock_nums=len(self.symbol_array),      # 股票数量
                basic_factor_names=basic_fator_fields,  # 基础因子名
                factor_names=type(self).get_factor_names(),    # 最终因子名
                dtypes=dtypes,                          # 基础因子类型
                symbols=self.symbol_array,              # 当日股票池
                # n_jobs=D.get_cyc_info('n_jobs'),        # 多进程数量
                n_jobs=self.get_cur_pool_cfg()['n_jobs'],
                pre_update_cnt=pre_update_rows,         # 预先更新的样本行数
                last_symbols=self.last_symbol_array,    # 昨日股票池
                )
            # endregion
        # region 历史数据计算完后 读取今日的eod数据
        # cap_data = D.get_stock_capital_series(start_date=yesterday, end_date=yesterday)
        # cap_data = dapi.get_stock_capital_series(start_date=yesterday, end_date=yesterday)
        # cap_data.reset_index(level='timestamp', drop=True, inplace=True)
        # cap_data.rename_axis('symbol', inplace=True)
        # cap_data = cap_data[['circulating_market_cap']]
        # cap_data.columns = ['circulating_market_cap']
        #TODO
        cap_data = pd.read_csv(os.path.join(self.get_cur_pool_cfg()['cap_path'],str(yesterday)[:4],str(yesterday)[4:6],str(yesterday)[6:],'derivative_indicator.csv'))[['SecurityID','CirculatingMarketValue']]
        cap_data.columns = ['symbol','circulating_market_cap']
        cap_data['symbol'] = cap_data['symbol'].apply(lambda x:('000000' + str(x))[-6:])
        cap_data = cap_data.set_index(['symbol'])
        
        
        # add_s = set(D.get_all_symbols()).difference(set(cap_data.index))
        add_s = set(self.get_symbols()).difference(set(cap_data.index))
        if add_s:
            add_d = pd.DataFrame(index=pd.Index(add_s), columns=['circulating_market_cap'])
            cap_data = pd.concat([cap_data, add_d]).rename_axis('symbol')
        # del_s = set(cap_data.index).difference(set(D.get_all_symbols()))
        del_s = set(cap_data.index).difference(set(self.get_symbols()))
        if del_s:
            cap_data.drop(pd.Index(del_s), axis=0, inplace=True)
        # cap_data = cap_data.loc[D.get_all_symbols(), :]
        cap_data = cap_data.loc[self.get_symbols(), :]
        # endregion
       
        # region 更新当日市值信息
        fe.DC.update_shared_memory(cap_data)
        # endregion

    def get_factors(self):
        return self.current_timestamp, self.factor_data

    def on_end(self):
        """当天交易时间结束,会通知所有因子
        """
        # 当前因子视图转为非共享内存 防止内存泄漏
        self.res = deepcopy(self.factor_data)
        # 因子计算引擎 等待所有子进程储存最终因子类
        self.alpha_engine.wait_process_end()
        # 关闭共享内存
        fe.DC.close()
        #save:
        # date_str = str(D.get_trading_dates()[-1])
        date_str = str(self.get_trading_dates()[-1])
        year = str(date_str[0:4])
        month = str(date_str[4:6])
        day = str(date_str[6:])
        # if not os.path.exists(os.path.join(cross_sectional_file_path,year,month,day)):
        #     os.makedirs(os.path.join(cross_sectional_file_path,year,month,day))
        # self.res.to_parquet(os.path.join(cross_sectional_file_path,year,month,day,f'{calc_pool_name}.par'))
        # self.res.to_parquet(os.path.join(cross_sectional_file_path,year,month,day,f'{calc_pool_name}.par'))
        

        #delete pkl:
        filelist = os.listdir(custom_file_path)
        # delete_list = [_ for _ in filelist if int(_) < D.get_trading_dates()[-5]]
        delete_list = [_ for _ in filelist if int(_) < self.get_trading_dates()[-5]]
        if len(delete_list) > 0:
            for _ in delete_list:
                shutil.rmtree(os.path.join(custom_file_path,_))
            logger.info('delete_custom_file===!!!!')

        # ======================
        logger.info('end===!!!!')

    def on_rtn_factors(self, date_time_cut: int,
                              pool_name: str, data: pd.DataFrame):
        # 获取当前时间戳
        self.current_timestamp = date_time_cut
        self.timestamp = str(date_time_cut)[8:12]
        if self.timestamp in ['1300', '0930', '1458', '1459', '1500']: return
        self.feature_ready += 1
        # 因子大表根据收到的特征池更新对应坐标的值
        self.data.loc[:,self.factor_names_map[pool_name]] = \
            data.loc[:,self.factor_names_map[pool_name]].reset_index(level='date_time_cut', drop=True)
        # 当收齐了所有人的基础特征,开始计算因子
        if self.feature_ready == self.pool_num:
            # 计算衍生因子
            self.cal_dev()
            # 获取返回值并填入大表
            temp_data = self.alpha_engine.run(data=self.data)
            # print(temp_data)
            #TODO 只存几个截面
            # if self.timestamp in FxALPHAV1.save_time_cut:
            if date_time_cut in self.get_cs_date_time_cuts():
                # self.factor_data.loc[(date_time_cut, slice(None)), :] = temp_data.values
                self.factor_data.loc[(date_time_cut, slice(None)), :] = np.float32(temp_data.values)
            #TODO
            self.feature_ready = 0
        logger.debug(f"========{date_time_cut}==={pool_name}")

    def cal_dev(self):
        self.data['big_actB_fx'] = (self.data.actB_4_amt_fx + self.data.actB_3_amt_fx)
        self.data['big_actB_cnt_fx'] = (self.data.actB_4_cnt_fx + self.data.actB_3_cnt_fx)
        self.data['big_actB_vol_fx'] = (self.data.actB_4_vol_fx + self.data.actB_3_vol_fx)
        self.data['small_actB_fx'] = (self.data.actB_1_amt_fx + self.data.actB_2_amt_fx)
        self.data['small_actB_cnt_fx'] = (self.data.actB_1_cnt_fx + self.data.actB_2_cnt_fx)
        self.data['small_actB_vol_fx'] = (self.data.actB_1_vol_fx + self.data.actB_2_vol_fx)
        self.data['big_actS_fx'] = (self.data.actS_4_amt_fx + self.data.actS_3_amt_fx)
        self.data['big_actS_cnt_fx'] = (self.data.actS_4_cnt_fx + self.data.actS_3_cnt_fx)
        self.data['big_actS_vol_fx'] = (self.data.actS_4_vol_fx + self.data.actS_3_vol_fx)
        self.data['small_actS_fx'] = (self.data.actS_1_amt_fx + self.data.actS_2_amt_fx)
        self.data['small_actS_cnt_fx'] = (self.data.actS_1_cnt_fx + self.data.actS_2_cnt_fx)
        self.data['small_actS_vol_fx'] = (self.data.actS_1_vol_fx + self.data.actS_2_vol_fx)
        self.data['big_actB_up_fx'] = (self.data.actB_4_amt_up_fx + self.data.actB_3_amt_up_fx)
        self.data['big_actB_cnt_up_fx'] = (self.data.actB_4_cnt_up_fx + self.data.actB_3_cnt_up_fx)
        self.data['big_actB_vol_up_fx'] = (self.data.actB_4_vol_up_fx + self.data.actB_3_vol_up_fx)
        self.data['small_actB_up_fx'] = (self.data.actB_1_amt_up_fx + self.data.actB_2_amt_up_fx)
        self.data['small_actB_cnt_up_fx'] = (self.data.actB_1_cnt_up_fx + self.data.actB_2_cnt_up_fx)
        self.data['small_actB_vol_up_fx'] = (self.data.actB_1_vol_up_fx + self.data.actB_2_vol_up_fx)
        self.data['big_actS_up_fx'] = (self.data.actS_4_amt_up_fx + self.data.actS_3_amt_up_fx)
        self.data['big_actS_cnt_up_fx'] = (self.data.actS_4_cnt_up_fx + self.data.actS_3_cnt_up_fx)
        self.data['big_actS_vol_up_fx'] = (self.data.actS_4_vol_up_fx + self.data.actS_3_vol_up_fx)
        self.data['small_actS_up_fx'] = (self.data.actS_1_amt_up_fx + self.data.actS_2_amt_up_fx)
        self.data['small_actS_cnt_up_fx'] = (self.data.actS_1_cnt_up_fx + self.data.actS_2_cnt_up_fx)
        self.data['small_actS_vol_up_fx'] = (self.data.actS_1_vol_up_fx + self.data.actS_2_vol_up_fx)
        self.data['big_actB_down_fx'] = (self.data.actB_4_amt_down_fx + self.data.actB_3_amt_down_fx)
        self.data['big_actB_cnt_down_fx'] = (self.data.actB_4_cnt_down_fx + self.data.actB_3_cnt_down_fx)
        self.data['big_actB_vol_down_fx'] = (self.data.actB_4_vol_down_fx + self.data.actB_3_vol_down_fx)
        self.data['small_actB_down_fx'] = (self.data.actB_1_amt_down_fx + self.data.actB_2_amt_down_fx)
        self.data['small_actB_cnt_down_fx'] = (self.data.actB_1_cnt_down_fx + self.data.actB_2_cnt_down_fx)
        self.data['small_actB_vol_down_fx'] = (self.data.actB_1_vol_down_fx + self.data.actB_2_vol_down_fx)
        self.data['big_actS_down_fx'] = (self.data.actS_4_amt_down_fx + self.data.actS_3_amt_down_fx)
        self.data['big_actS_cnt_down_fx'] = (self.data.actS_4_cnt_down_fx + self.data.actS_3_cnt_down_fx)
        self.data['big_actS_vol_down_fx'] = (self.data.actS_4_vol_down_fx + self.data.actS_3_vol_down_fx)
        self.data['small_actS_down_fx'] = (self.data.actS_1_amt_down_fx + self.data.actS_2_amt_down_fx)
        self.data['small_actS_cnt_down_fx'] = (self.data.actS_1_cnt_down_fx + self.data.actS_2_cnt_down_fx)
        self.data['small_actS_vol_down_fx'] = (self.data.actS_1_vol_down_fx + self.data.actS_2_vol_down_fx)
        self.data['big_actB_equi_fx'] = (self.data.actB_4_amt_equi_fx + self.data.actB_3_amt_equi_fx)
        self.data['big_actB_cnt_equi_fx'] = (self.data.actB_4_cnt_equi_fx + self.data.actB_3_cnt_equi_fx)
        self.data['big_actB_vol_equi_fx'] = (self.data.actB_4_vol_equi_fx + self.data.actB_3_vol_equi_fx)
        self.data['small_actB_equi_fx'] = (self.data.actB_1_amt_equi_fx + self.data.actB_2_amt_equi_fx)
        self.data['small_actB_cnt_equi_fx'] = (self.data.actB_1_cnt_equi_fx + self.data.actB_2_cnt_equi_fx)
        self.data['small_actB_vol_equi_fx'] = (self.data.actB_1_vol_equi_fx + self.data.actB_2_vol_equi_fx)
        self.data['big_actS_equi_fx'] = (self.data.actS_4_amt_equi_fx + self.data.actS_3_amt_equi_fx)
        self.data['big_actS_cnt_equi_fx'] = (self.data.actS_4_cnt_equi_fx + self.data.actS_3_cnt_equi_fx)
        self.data['big_actS_vol_equi_fx'] = (self.data.actS_4_vol_equi_fx + self.data.actS_3_vol_equi_fx)
        self.data['small_actS_equi_fx'] = (self.data.actS_1_amt_equi_fx + self.data.actS_2_amt_equi_fx)
        self.data['small_actS_cnt_equi_fx'] = (self.data.actS_1_cnt_equi_fx + self.data.actS_2_cnt_equi_fx)
        self.data['small_actS_vol_equi_fx'] = (self.data.actS_1_vol_equi_fx + self.data.actS_2_vol_equi_fx)
        self.data['buy_big_order_tot_fx'] = (self.data.buy_order_3_fx + self.data.buy_order_4_fx)
        self.data['buy_big_order_act_tot_fx'] = (self.data.buy_order_act_3_fx + self.data.buy_order_act_4_fx)
        self.data['buy_big_order_cnt_fx'] = (self.data.buy_order_cnt_3_fx + self.data.buy_order_cnt_4_fx)
        self.data['buy_big_order_act_cnt_fx'] = (self.data.buy_order_act_cnt_3_fx + self.data.buy_order_act_cnt_4_fx)
        self.data['buy_big_order_tot_vol_fx'] = (self.data.buy_order_vol_3_fx + self.data.buy_order_vol_4_fx)
        self.data['buy_big_order_act_vol_fx'] = (self.data.buy_order_act_vol_3_fx + self.data.buy_order_act_vol_4_fx)
        self.data['buy_small_order_tot_fx'] = (self.data.buy_order_1_fx + self.data.buy_order_2_fx)
        self.data['buy_small_order_act_tot_fx'] = (self.data.buy_order_act_1_fx + self.data.buy_order_act_2_fx)
        self.data['buy_small_order_cnt_fx'] = (self.data.buy_order_cnt_1_fx + self.data.buy_order_cnt_2_fx)
        self.data['buy_small_order_act_cnt_fx'] = (self.data.buy_order_act_cnt_1_fx + self.data.buy_order_act_cnt_2_fx)
        self.data['buy_small_order_tot_vol_fx'] = (self.data.buy_order_vol_1_fx + self.data.buy_order_vol_2_fx)
        self.data['buy_small_order_act_vol_fx'] = (self.data.buy_order_act_vol_1_fx + self.data.buy_order_act_vol_2_fx)
        self.data['sell_big_order_tot_fx'] = (self.data.sell_order_3_fx + self.data.sell_order_4_fx)
        self.data['sell_big_order_act_tot_fx'] = (self.data.sell_order_act_3_fx + self.data.sell_order_act_4_fx)
        self.data['sell_big_order_cnt_fx'] = (self.data.sell_order_cnt_3_fx + self.data.sell_order_cnt_4_fx)
        self.data['sell_big_order_act_cnt_fx'] = (self.data.sell_order_act_cnt_3_fx + self.data.sell_order_act_cnt_4_fx)
        self.data['sell_big_order_tot_vol_fx'] = (self.data.sell_order_vol_3_fx + self.data.sell_order_vol_4_fx)
        self.data['sell_big_order_act_vol_fx'] = (self.data.sell_order_act_vol_3_fx + self.data.sell_order_act_vol_4_fx)
        self.data['sell_small_order_tot_fx'] = (self.data.sell_order_1_fx + self.data.sell_order_2_fx)
        self.data['sell_small_order_act_tot_fx'] = (self.data.sell_order_act_1_fx + self.data.sell_order_act_2_fx)
        self.data['sell_small_order_cnt_fx'] = (self.data.sell_order_cnt_1_fx + self.data.sell_order_cnt_2_fx)
        self.data['sell_small_order_act_cnt_fx'] = (self.data.sell_order_act_cnt_1_fx + self.data.sell_order_act_cnt_2_fx)
        self.data['sell_small_order_tot_vol_fx'] = (self.data.sell_order_vol_1_fx + self.data.sell_order_vol_2_fx)
        self.data['sell_small_order_act_vol_fx'] = (self.data.sell_order_act_vol_1_fx + self.data.sell_order_act_vol_2_fx)
        self.data['buy_big_order_up_fx'] = (self.data.buy_order_up_3_fx + self.data.buy_order_up_4_fx)
        self.data['buy_big_order_act_up_fx'] = (self.data.buy_order_up_act_3_fx + self.data.buy_order_up_act_4_fx)
        self.data['buy_big_order_up_cnt_fx'] = (self.data.buy_order_up_cnt_3_fx + self.data.buy_order_up_cnt_4_fx)
        self.data['buy_big_order_up_vol_fx'] = (self.data.buy_order_up_vol_3_fx + self.data.buy_order_up_vol_4_fx)
        self.data['buy_big_order_act_up_vol_fx'] = (self.data.buy_order_up_act_vol_3_fx + self.data.buy_order_up_act_vol_4_fx)
        self.data['buy_small_order_up_fx'] = (self.data.buy_order_up_1_fx + self.data.buy_order_up_2_fx)
        self.data['buy_small_order_act_up_fx'] = (self.data.buy_order_up_act_1_fx + self.data.buy_order_up_act_2_fx)
        self.data['buy_small_order_up_cnt_fx'] = (self.data.buy_order_up_cnt_1_fx + self.data.buy_order_up_cnt_2_fx)
        self.data['buy_small_order_up_vol_fx'] = (self.data.buy_order_up_vol_1_fx + self.data.buy_order_up_vol_2_fx)
        self.data['buy_small_order_act_up_vol_fx'] = (self.data.buy_order_up_act_vol_1_fx + self.data.buy_order_up_act_vol_2_fx)
        self.data['sell_big_order_up_fx'] = (self.data.sell_order_up_3_fx + self.data.sell_order_up_4_fx)
        self.data['sell_big_order_act_up_fx'] = (self.data.sell_order_up_act_3_fx + self.data.sell_order_up_act_4_fx)
        self.data['sell_big_order_up_cnt_fx'] = (self.data.sell_order_up_cnt_3_fx + self.data.sell_order_up_cnt_4_fx)
        self.data['sell_big_order_up_vol_fx'] = (self.data.sell_order_up_vol_3_fx + self.data.sell_order_up_vol_4_fx)
        self.data['sell_big_order_act_up_vol_fx'] = (self.data.sell_order_up_act_vol_3_fx + self.data.sell_order_up_act_vol_4_fx)
        self.data['sell_small_order_up_fx'] = (self.data.sell_order_up_1_fx + self.data.sell_order_up_2_fx)
        self.data['sell_small_order_act_up_fx'] = (self.data.sell_order_up_act_1_fx + self.data.sell_order_up_act_2_fx)
        self.data['sell_small_order_up_cnt_fx'] = (self.data.sell_order_up_cnt_1_fx + self.data.sell_order_up_cnt_2_fx)
        self.data['sell_small_order_up_vol_fx'] = (self.data.sell_order_up_vol_1_fx + self.data.sell_order_up_vol_2_fx)
        self.data['sell_small_order_act_up_vol_fx'] = (self.data.sell_order_up_act_vol_1_fx + self.data.sell_order_up_act_vol_2_fx)
        self.data['buy_big_order_down_fx'] = (self.data.buy_order_down_3_fx + self.data.buy_order_down_4_fx)
        self.data['buy_big_order_act_down_fx'] = (self.data.buy_order_down_act_3_fx + self.data.buy_order_down_act_4_fx)
        self.data['buy_big_order_down_cnt_fx'] = (self.data.buy_order_down_cnt_3_fx + self.data.buy_order_down_cnt_4_fx)
        self.data['buy_big_order_down_vol_fx'] = (self.data.buy_order_down_vol_3_fx + self.data.buy_order_down_vol_4_fx)
        self.data['buy_big_order_act_down_vol_fx'] = (self.data.buy_order_down_act_vol_3_fx + self.data.buy_order_down_act_vol_4_fx)
        self.data['buy_small_order_down_fx'] = (self.data.buy_order_down_1_fx + self.data.buy_order_down_2_fx)
        self.data['buy_small_order_act_down_fx'] = (self.data.buy_order_down_act_1_fx + self.data.buy_order_down_act_2_fx)
        self.data['buy_small_order_down_cnt_fx'] = (self.data.buy_order_down_cnt_1_fx + self.data.buy_order_down_cnt_2_fx)
        self.data['buy_small_order_down_vol_fx'] = (self.data.buy_order_down_vol_1_fx + self.data.buy_order_down_vol_2_fx)
        self.data['buy_small_order_act_down_vol_fx'] = (self.data.buy_order_down_act_vol_1_fx + self.data.buy_order_down_act_vol_2_fx)
        self.data['sell_big_order_down_fx'] = (self.data.sell_order_down_3_fx + self.data.sell_order_down_4_fx)
        self.data['sell_big_order_act_down_fx'] = (self.data.sell_order_down_act_3_fx + self.data.sell_order_down_act_4_fx)
        self.data['sell_big_order_down_cnt_fx'] = (self.data.sell_order_down_cnt_3_fx + self.data.sell_order_down_cnt_4_fx)
        self.data['sell_big_order_down_vol_fx'] = (self.data.sell_order_down_vol_3_fx + self.data.sell_order_down_vol_4_fx)
        self.data['sell_big_order_act_down_vol_fx'] = (self.data.sell_order_down_act_vol_3_fx + self.data.sell_order_down_act_vol_4_fx)
        self.data['sell_small_order_down_fx'] = (self.data.sell_order_down_1_fx + self.data.sell_order_down_2_fx)
        self.data['sell_small_order_act_down_fx'] = (self.data.sell_order_down_act_1_fx + self.data.sell_order_down_act_2_fx)
        self.data['sell_small_order_down_cnt_fx'] = (self.data.sell_order_down_cnt_1_fx + self.data.sell_order_down_cnt_2_fx)
        self.data['sell_small_order_down_vol_fx'] = (self.data.sell_order_down_vol_1_fx + self.data.sell_order_down_vol_2_fx)
        self.data['sell_small_order_act_down_vol_fx'] = (self.data.sell_order_down_act_vol_1_fx + self.data.sell_order_down_act_vol_2_fx)
        self.data['buy_big_order_spreadup_fx'] = (self.data.buy_order_spreadup_3_fx + self.data.buy_order_spreadup_4_fx)
        self.data['buy_big_order_act_spreadup_fx'] = (self.data.buy_order_spreadup_act_3_fx + self.data.buy_order_spreadup_act_4_fx)
        self.data['buy_big_order_spreadup_cnt_fx'] = (self.data.buy_order_spreadup_cnt_3_fx + self.data.buy_order_spreadup_cnt_4_fx)
        self.data['buy_big_order_spreadup_vol_fx'] = (self.data.buy_order_spreadup_vol_3_fx + self.data.buy_order_spreadup_vol_4_fx)
        self.data['buy_big_order_act_spreadup_vol_fx'] = (self.data.buy_order_spreadup_act_vol_3_fx + self.data.buy_order_spreadup_act_vol_4_fx)
        self.data['buy_small_order_spreadup_fx'] = (self.data.buy_order_spreadup_1_fx + self.data.buy_order_spreadup_2_fx)
        self.data['buy_small_order_act_spreadup_fx'] = (self.data.buy_order_spreadup_act_1_fx + self.data.buy_order_spreadup_act_2_fx)
        self.data['buy_small_order_spreadup_cnt_fx'] = (self.data.buy_order_spreadup_cnt_1_fx + self.data.buy_order_spreadup_cnt_2_fx)
        self.data['buy_small_order_spreadup_vol_fx'] = (self.data.buy_order_spreadup_vol_1_fx + self.data.buy_order_spreadup_vol_2_fx)
        self.data['buy_small_order_act_spreadup_vol_fx'] = (self.data.buy_order_spreadup_act_vol_1_fx + self.data.buy_order_spreadup_act_vol_2_fx)
        self.data['sell_big_order_spreadup_fx'] = (self.data.sell_order_spreadup_3_fx + self.data.sell_order_spreadup_4_fx)
        self.data['sell_big_order_act_spreadup_fx'] = (self.data.sell_order_spreadup_act_3_fx + self.data.sell_order_spreadup_act_4_fx)
        self.data['sell_big_order_spreadup_cnt_fx'] = (self.data.sell_order_spreadup_cnt_3_fx + self.data.sell_order_spreadup_cnt_4_fx)
        self.data['sell_big_order_spreadup_vol_fx'] = (self.data.sell_order_spreadup_vol_3_fx + self.data.sell_order_spreadup_vol_4_fx)
        self.data['sell_big_order_act_spreadup_vol_fx'] = (self.data.sell_order_spreadup_act_vol_3_fx + self.data.sell_order_spreadup_act_vol_4_fx)
        self.data['sell_small_order_spreadup_fx'] = (self.data.sell_order_spreadup_1_fx + self.data.sell_order_spreadup_2_fx)
        self.data['sell_small_order_act_spreadup_fx'] = (self.data.sell_order_spreadup_act_1_fx + self.data.sell_order_spreadup_act_2_fx)
        self.data['sell_small_order_spreadup_cnt_fx'] = (self.data.sell_order_spreadup_cnt_1_fx + self.data.sell_order_spreadup_cnt_2_fx)
        self.data['sell_small_order_spreadup_vol_fx'] = (self.data.sell_order_spreadup_vol_1_fx + self.data.sell_order_spreadup_vol_2_fx)
        self.data['sell_small_order_act_spreadup_vol_fx'] = (self.data.sell_order_spreadup_act_vol_1_fx + self.data.sell_order_spreadup_act_vol_2_fx)
        self.data['buy_big_order_spreaddown_fx'] = (self.data.buy_order_spreaddown_3_fx + self.data.buy_order_spreaddown_4_fx)
        self.data['buy_big_order_act_spreaddown_fx'] = (self.data.buy_order_spreaddown_act_3_fx + self.data.buy_order_spreaddown_act_4_fx)
        self.data['buy_big_order_spreaddown_cnt_fx'] = (self.data.buy_order_spreaddown_cnt_3_fx + self.data.buy_order_spreaddown_cnt_4_fx)
        self.data['buy_big_order_spreaddown_vol_fx'] = (self.data.buy_order_spreaddown_vol_3_fx + self.data.buy_order_spreaddown_vol_4_fx)
        self.data['buy_big_order_act_spreaddown_vol_fx'] = (self.data.buy_order_spreaddown_vol_act_3_fx + self.data.buy_order_spreaddown_vol_act_4_fx)
        self.data['buy_small_order_spreaddown_fx'] = (self.data.buy_order_spreaddown_1_fx + self.data.buy_order_spreaddown_2_fx)
        self.data['buy_small_order_act_spreaddown_fx'] = (self.data.buy_order_spreaddown_act_1_fx + self.data.buy_order_spreaddown_act_2_fx)
        self.data['buy_small_order_spreaddown_cnt_fx'] = (self.data.buy_order_spreaddown_cnt_1_fx + self.data.buy_order_spreaddown_cnt_2_fx)
        self.data['buy_small_order_spreaddown_vol_fx'] = (self.data.buy_order_spreaddown_vol_1_fx + self.data.buy_order_spreaddown_vol_2_fx)
        self.data['buy_small_order_act_spreaddown_vol_fx'] = (self.data.buy_order_spreaddown_vol_act_1_fx + self.data.buy_order_spreaddown_vol_act_2_fx)
        self.data['sell_big_order_spreaddown_fx'] = (self.data.sell_order_spreaddown_3_fx + self.data.sell_order_spreaddown_4_fx)
        self.data['sell_big_order_act_spreaddown_fx'] = (self.data.sell_order_spreaddown_act_3_fx + self.data.sell_order_spreaddown_act_4_fx)
        self.data['sell_big_order_spreaddown_cnt_fx'] = (self.data.sell_order_spreaddown_cnt_3_fx + self.data.sell_order_spreaddown_cnt_4_fx)
        self.data['sell_big_order_spreaddown_vol_fx'] = (self.data.sell_order_spreaddown_vol_3_fx + self.data.sell_order_spreaddown_vol_4_fx)
        self.data['sell_big_order_act_spreaddown_vol_fx'] = (self.data.sell_order_spreaddown_vol_act_3_fx + self.data.sell_order_spreaddown_vol_act_4_fx)
        self.data['sell_small_order_spreaddown_fx'] = (self.data.sell_order_spreaddown_1_fx + self.data.sell_order_spreaddown_2_fx)
        self.data['sell_small_order_act_spreaddown_fx'] = (self.data.sell_order_spreaddown_act_1_fx + self.data.sell_order_spreaddown_act_2_fx)
        self.data['sell_small_order_spreaddown_cnt_fx'] = (self.data.sell_order_spreaddown_cnt_1_fx + self.data.sell_order_spreaddown_cnt_2_fx)
        self.data['sell_small_order_spreaddown_vol_fx'] = (self.data.sell_order_spreaddown_vol_1_fx + self.data.sell_order_spreaddown_vol_2_fx)
        self.data['sell_small_order_act_spreaddown_vol_fx'] = (self.data.sell_order_spreaddown_vol_act_1_fx + self.data.sell_order_spreaddown_vol_act_2_fx)
        self.data['buy_big_order_sellagg_fx'] = (self.data.buy_order_sellagg_3_fx + self.data.buy_order_sellagg_4_fx)
        self.data['buy_big_order_act_sellagg_fx'] = (self.data.buy_order_sellagg_act_3_fx + self.data.buy_order_sellagg_act_4_fx)
        self.data['buy_big_order_sellagg_cnt_fx'] = (self.data.buy_order_sellagg_cnt_3_fx + self.data.buy_order_sellagg_cnt_4_fx)
        self.data['buy_big_order_sellagg_vol_fx'] = (self.data.buy_order_sellagg_vol_3_fx + self.data.buy_order_sellagg_vol_4_fx)
        self.data['buy_big_order_act_sellagg_vol_fx'] = (self.data.buy_order_sellagg_act_vol_3_fx + self.data.buy_order_sellagg_act_vol_4_fx)
        self.data['buy_small_order_sellagg_fx'] = (self.data.buy_order_sellagg_1_fx + self.data.buy_order_sellagg_2_fx)
        self.data['buy_small_order_act_sellagg_fx'] = (self.data.buy_order_sellagg_act_1_fx + self.data.buy_order_sellagg_act_2_fx)
        self.data['buy_small_order_sellagg_cnt_fx'] = (self.data.buy_order_sellagg_cnt_1_fx + self.data.buy_order_sellagg_cnt_2_fx)
        self.data['buy_small_order_sellagg_vol_fx'] = (self.data.buy_order_sellagg_vol_1_fx + self.data.buy_order_sellagg_vol_2_fx)
        self.data['buy_small_order_act_sellagg_vol_fx'] = (self.data.buy_order_sellagg_act_vol_1_fx + self.data.buy_order_sellagg_act_vol_2_fx)
        self.data['sell_big_order_sellagg_fx'] = (self.data.sell_order_sellagg_3_fx + self.data.sell_order_sellagg_4_fx)
        self.data['sell_big_order_act_sellagg_fx'] = (self.data.sell_order_sellagg_act_3_fx + self.data.sell_order_sellagg_act_4_fx)
        self.data['sell_big_order_sellagg_cnt_fx'] = (self.data.sell_order_sellagg_cnt_3_fx + self.data.sell_order_sellagg_cnt_4_fx)
        self.data['sell_big_order_sellagg_vol_fx'] = (self.data.sell_order_sellagg_vol_3_fx + self.data.sell_order_sellagg_vol_4_fx)
        self.data['sell_big_order_act_sellagg_vol_fx'] = (self.data.sell_order_sellagg_act_vol_3_fx + self.data.sell_order_sellagg_act_vol_4_fx)
        self.data['sell_small_order_sellagg_fx'] = (self.data.sell_order_sellagg_1_fx + self.data.sell_order_sellagg_2_fx)
        self.data['sell_small_order_act_sellagg_fx'] = (self.data.sell_order_sellagg_act_1_fx + self.data.sell_order_sellagg_act_2_fx)
        self.data['sell_small_order_sellagg_cnt_fx'] = (self.data.sell_order_sellagg_cnt_1_fx + self.data.sell_order_sellagg_cnt_2_fx)
        self.data['sell_small_order_sellagg_vol_fx'] = (self.data.sell_order_sellagg_vol_1_fx + self.data.sell_order_sellagg_vol_2_fx)
        self.data['sell_small_order_act_sellagg_vol_fx'] = (self.data.sell_order_sellagg_act_vol_1_fx + self.data.sell_order_sellagg_act_vol_2_fx)
        self.data['buy_big_order_buyagg_fx'] = (self.data.buy_order_buyagg_3_fx + self.data.buy_order_buyagg_4_fx)
        self.data['buy_big_order_act_buyagg_fx'] = (self.data.buy_order_buyagg_act_3_fx + self.data.buy_order_buyagg_act_4_fx)
        self.data['buy_big_order_buyagg_cnt_fx'] = (self.data.buy_order_buyagg_cnt_3_fx + self.data.buy_order_buyagg_cnt_4_fx)
        self.data['buy_big_order_buyagg_vol_fx'] = (self.data.buy_order_buyagg_vol_3_fx + self.data.buy_order_buyagg_vol_4_fx)
        self.data['buy_big_order_act_buyagg_vol_fx'] = (self.data.buy_order_buyagg_act_vol_3_fx + self.data.buy_order_buyagg_act_vol_4_fx)
        self.data['buy_small_order_buyagg_fx'] = (self.data.buy_order_buyagg_1_fx + self.data.buy_order_buyagg_2_fx)
        self.data['buy_small_order_act_buyagg_fx'] = (self.data.buy_order_buyagg_act_1_fx + self.data.buy_order_buyagg_act_2_fx)
        self.data['buy_small_order_buyagg_cnt_fx'] = (self.data.buy_order_buyagg_cnt_1_fx + self.data.buy_order_buyagg_cnt_2_fx)
        self.data['buy_small_order_buyagg_vol_fx'] = (self.data.buy_order_buyagg_vol_1_fx + self.data.buy_order_buyagg_vol_2_fx)
        self.data['buy_small_order_act_buyagg_vol_fx'] = (self.data.buy_order_buyagg_act_vol_1_fx + self.data.buy_order_buyagg_act_vol_2_fx)
        self.data['sell_big_order_buyagg_fx'] = (self.data.sell_order_buyagg_3_fx + self.data.sell_order_buyagg_4_fx)
        self.data['sell_big_order_act_buyagg_fx'] = (self.data.sell_order_buyagg_act_3_fx + self.data.sell_order_buyagg_act_4_fx)
        self.data['sell_big_order_buyagg_cnt_fx'] = (self.data.sell_order_buyagg_cnt_3_fx + self.data.sell_order_buyagg_cnt_4_fx)
        self.data['sell_big_order_buyagg_vol_fx'] = (self.data.sell_order_buyagg_vol_3_fx + self.data.sell_order_buyagg_vol_4_fx)
        self.data['sell_big_order_act_buyagg_vol_fx'] = (self.data.sell_order_buyagg_act_vol_3_fx + self.data.sell_order_buyagg_act_vol_4_fx)
        self.data['sell_small_order_buyagg_fx'] = (self.data.sell_order_buyagg_1_fx + self.data.sell_order_buyagg_2_fx)
        self.data['sell_small_order_act_buyagg_fx'] = (self.data.sell_order_buyagg_act_1_fx + self.data.sell_order_buyagg_act_2_fx)
        self.data['sell_small_order_buyagg_cnt_fx'] = (self.data.sell_order_buyagg_cnt_1_fx + self.data.sell_order_buyagg_cnt_2_fx)
        self.data['sell_small_order_buyagg_vol_fx'] = (self.data.sell_order_buyagg_vol_1_fx + self.data.sell_order_buyagg_vol_2_fx)
        self.data['sell_small_order_act_buyagg_vol_fx'] = (self.data.sell_order_buyagg_act_vol_1_fx + self.data.sell_order_buyagg_act_vol_2_fx)
        self.data['buy_big_order_sellspa_fx'] = (self.data.buy_order_sellspa_3_fx + self.data.buy_order_sellspa_4_fx)
        self.data['buy_big_order_act_sellspa_fx'] = (self.data.buy_order_sellspa_act_3_fx + self.data.buy_order_sellspa_act_4_fx)
        self.data['buy_big_order_sellspa_cnt_fx'] = (self.data.buy_order_sellspa_cnt_3_fx + self.data.buy_order_sellspa_cnt_4_fx)
        self.data['buy_big_order_sellspa_vol_fx'] = (self.data.buy_order_sellspa_vol_3_fx + self.data.buy_order_sellspa_vol_4_fx)
        self.data['buy_big_order_act_sellspa_vol_fx'] = (self.data.buy_order_sellspa_act_vol_3_fx + self.data.buy_order_sellspa_act_vol_4_fx)
        self.data['buy_small_order_sellspa_fx'] = (self.data.buy_order_sellspa_1_fx + self.data.buy_order_sellspa_2_fx)
        self.data['buy_small_order_act_sellspa_fx'] = (self.data.buy_order_sellspa_act_1_fx + self.data.buy_order_sellspa_act_2_fx)
        self.data['buy_small_order_sellspa_cnt_fx'] = (self.data.buy_order_sellspa_cnt_1_fx + self.data.buy_order_sellspa_cnt_2_fx)
        self.data['buy_small_order_sellspa_vol_fx'] = (self.data.buy_order_sellspa_vol_1_fx + self.data.buy_order_sellspa_vol_2_fx)
        self.data['buy_small_order_act_sellspa_vol_fx'] = (self.data.buy_order_sellspa_act_vol_1_fx + self.data.buy_order_sellspa_act_vol_2_fx)
        self.data['sell_big_order_sellspa_fx'] = (self.data.sell_order_sellspa_3_fx + self.data.sell_order_sellspa_4_fx)
        self.data['sell_big_order_act_sellspa_fx'] = (self.data.sell_order_sellspa_act_3_fx + self.data.sell_order_sellspa_act_4_fx)
        self.data['sell_big_order_sellspa_cnt_fx'] = (self.data.sell_order_sellspa_cnt_3_fx + self.data.sell_order_sellspa_cnt_4_fx)
        self.data['sell_big_order_sellspa_vol_fx'] = (self.data.sell_order_sellspa_vol_3_fx + self.data.sell_order_sellspa_vol_4_fx)
        self.data['sell_big_order_act_sellspa_vol_fx'] = (self.data.sell_order_sellspa_act_vol_3_fx + self.data.sell_order_sellspa_act_vol_4_fx)
        self.data['sell_small_order_sellspa_fx'] = (self.data.sell_order_sellspa_1_fx + self.data.sell_order_sellspa_2_fx)
        self.data['sell_small_order_act_sellspa_fx'] = (self.data.sell_order_sellspa_act_1_fx + self.data.sell_order_sellspa_act_2_fx)
        self.data['sell_small_order_sellspa_cnt_fx'] = (self.data.sell_order_sellspa_cnt_1_fx + self.data.sell_order_sellspa_cnt_2_fx)
        self.data['sell_small_order_sellspa_vol_fx'] = (self.data.sell_order_sellspa_vol_1_fx + self.data.sell_order_sellspa_vol_2_fx)
        self.data['sell_small_order_act_sellspa_vol_fx'] = (self.data.sell_order_sellspa_act_vol_1_fx + self.data.sell_order_sellspa_act_vol_2_fx)
        self.data['buy_big_order_buyspa_fx'] = (self.data.buy_order_buyspa_3_fx + self.data.buy_order_buyspa_4_fx)
        self.data['buy_big_order_act_buyspa_fx'] = (self.data.buy_order_buyspa_act_3_fx + self.data.buy_order_buyspa_act_4_fx)
        self.data['buy_big_order_buyspa_cnt_fx'] = (self.data.buy_order_buyspa_cnt_3_fx + self.data.buy_order_buyspa_cnt_4_fx)
        self.data['buy_big_order_buyspa_vol_fx'] = (self.data.buy_order_buyspa_vol_3_fx + self.data.buy_order_buyspa_vol_4_fx)
        self.data['buy_big_order_act_buyspa_vol_fx'] = (self.data.buy_order_buyspa_act_vol_3_fx + self.data.buy_order_buyspa_act_vol_4_fx)
        self.data['buy_small_order_buyspa_fx'] = (self.data.buy_order_buyspa_1_fx + self.data.buy_order_buyspa_2_fx)
        self.data['buy_small_order_act_buyspa_fx'] = (self.data.buy_order_buyspa_act_1_fx + self.data.buy_order_buyspa_act_2_fx)
        self.data['buy_small_order_buyspa_cnt_fx'] = (self.data.buy_order_buyspa_cnt_1_fx + self.data.buy_order_buyspa_cnt_2_fx)
        self.data['buy_small_order_buyspa_vol_fx'] = (self.data.buy_order_buyspa_vol_1_fx + self.data.buy_order_buyspa_vol_2_fx)
        self.data['buy_small_order_act_buyspa_vol_fx'] = (self.data.buy_order_buyspa_act_vol_1_fx + self.data.buy_order_buyspa_act_vol_2_fx)
        self.data['sell_big_order_buyspa_fx'] = (self.data.sell_order_buyspa_3_fx + self.data.sell_order_buyspa_4_fx)
        self.data['sell_big_order_act_buyspa_fx'] = (self.data.sell_order_buyspa_act_3_fx + self.data.sell_order_buyspa_act_4_fx)
        self.data['sell_big_order_buyspa_cnt_fx'] = (self.data.sell_order_buyspa_cnt_3_fx + self.data.sell_order_buyspa_cnt_4_fx)
        self.data['sell_big_order_buyspa_vol_fx'] = (self.data.sell_order_buyspa_vol_3_fx + self.data.sell_order_buyspa_vol_4_fx)
        self.data['sell_big_order_act_buyspa_vol_fx'] = (self.data.sell_order_buyspa_act_vol_3_fx + self.data.sell_order_buyspa_act_vol_4_fx)
        self.data['sell_small_order_buyspa_fx'] = (self.data.sell_order_buyspa_1_fx + self.data.sell_order_buyspa_2_fx)
        self.data['sell_small_order_act_buyspa_fx'] = (self.data.sell_order_buyspa_act_1_fx + self.data.sell_order_buyspa_act_2_fx)
        self.data['sell_small_order_buyspa_cnt_fx'] = (self.data.sell_order_buyspa_cnt_1_fx + self.data.sell_order_buyspa_cnt_2_fx)
        self.data['sell_small_order_buyspa_vol_fx'] = (self.data.sell_order_buyspa_vol_1_fx + self.data.sell_order_buyspa_vol_2_fx)
        self.data['sell_small_order_act_buyspa_vol_fx'] = (self.data.sell_order_buyspa_act_vol_1_fx + self.data.sell_order_buyspa_act_vol_2_fx)
        self.data['buy_small_order_rt_fx'] = (self.data.buy_order_rt_1_fx + self.data.buy_order_rt_2_fx)
        self.data['buy_small_order_rt_act_fx'] = (self.data.buy_order_rt_act_1_fx + self.data.buy_order_rt_act_2_fx)
        self.data['buy_big_order_rt_fx'] = (self.data.buy_order_rt_3_fx + self.data.buy_order_rt_4_fx)
        self.data['buy_big_order_act_rt_fx'] = (self.data.buy_order_rt_act_3_fx + self.data.buy_order_rt_act_4_fx)
        self.data['sell_small_order_rt_fx'] = (self.data.sell_order_rt_1_fx + self.data.sell_order_rt_2_fx)
        self.data['sell_small_order_act_rt_fx'] = (self.data.sell_order_rt_act_1_fx + self.data.sell_order_rt_act_2_fx)
        self.data['sell_big_order_rt_fx'] = (self.data.sell_order_rt_3_fx + self.data.sell_order_rt_4_fx)
        self.data['sell_big_order_act_rt_fx'] = (self.data.sell_order_rt_act_3_fx + self.data.sell_order_rt_act_4_fx)



