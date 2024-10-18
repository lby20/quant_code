"""
@author: caiyucheng
@contact: caiyucheng23@163.com
@date: 2024-07-02
"""
from sn_platform.data_utils.sn_factor import CrossSectionalFactorPool
from sn_platform.data_utils.sn_data import D, logger
import pandas as pd
import numpy as np
from copy import deepcopy
from . import factor_expression as fe
from multiprocessing import (
    Queue, Process, Value, Lock
)
import multiprocessing
from . import cyc_operator
from collections import defaultdict
import os
import pickle as pkl
import datetime


dev_features = [
    'BidOrderSmallVol_cyc',
    'BidOrderSmallAmt_cyc',
    'BidOrderSmallCnt_cyc',
    'BidOrderLargeVol_cyc',
    'BidOrderLargeAmt_cyc',
    'BidOrderLargeCnt_cyc',
    'BidTradeSmallVol_cyc',
    'BidTradeSmallAmt_cyc',
    'BidTradeSmallCnt_cyc',
    'BidTradeLargeVol_cyc',
    'BidTradeLargeAmt_cyc',
    'BidTradeLargeCnt_cyc',
    'BidCancelSmallVol_cyc',
    'BidCancelSmallAmt_cyc',
    'BidCancelSmallCnt_cyc',
    'BidCancelLargeVol_cyc',
    'BidCancelLargeAmt_cyc',
    'BidCancelLargeCnt_cyc',
    'BidOrderActVol_cyc',
    'BidOrderActAmt_cyc',
    'BidOrderActCnt_cyc',
    'BidOrderCutVol_cyc',
    'BidOrderCutAmt_cyc',
    'BidOrderCutCnt_cyc',
    'BidOrderPasVol_cyc',
    'BidOrderPasAmt_cyc',
    'BidOrderPasCnt_cyc',
    'BidOrderNegVol_cyc',
    'BidOrderNegAmt_cyc',
    'BidOrderNegCnt_cyc',
    'BidTradeActVol_cyc',
    'BidTradeActAmt_cyc',
    'BidTradeActCnt_cyc',
    'BidTradeCutVol_cyc',
    'BidTradeCutAmt_cyc',
    'BidTradeCutCnt_cyc',
    'BidTradePasVol_cyc',
    'BidTradePasAmt_cyc',
    'BidTradePasCnt_cyc',
    'BidTradeNegVol_cyc',
    'BidTradeNegAmt_cyc',
    'BidTradeNegCnt_cyc',
    'BidCancelActVol_cyc',
    'BidCancelActAmt_cyc',
    'BidCancelActCnt_cyc',
    'BidCancelCutVol_cyc',
    'BidCancelCutAmt_cyc',
    'BidCancelCutCnt_cyc',
    'BidCancelPasVol_cyc',
    'BidCancelPasAmt_cyc',
    'BidCancelPasCnt_cyc',
    'BidCancelNegVol_cyc',
    'BidCancelNegAmt_cyc',
    'BidCancelNegCnt_cyc',
    'AskOrderSmallVol_cyc',
    'AskOrderSmallAmt_cyc',
    'AskOrderSmallCnt_cyc',
    'AskOrderLargeVol_cyc',
    'AskOrderLargeAmt_cyc',
    'AskOrderLargeCnt_cyc',
    'AskTradeSmallVol_cyc',
    'AskTradeSmallAmt_cyc',
    'AskTradeSmallCnt_cyc',
    'AskTradeLargeVol_cyc',
    'AskTradeLargeAmt_cyc',
    'AskTradeLargeCnt_cyc',
    'AskCancelSmallVol_cyc',
    'AskCancelSmallAmt_cyc',
    'AskCancelSmallCnt_cyc',
    'AskCancelLargeVol_cyc',
    'AskCancelLargeAmt_cyc',
    'AskCancelLargeCnt_cyc',
    'AskOrderActVol_cyc',
    'AskOrderActAmt_cyc',
    'AskOrderActCnt_cyc',
    'AskOrderCutVol_cyc',
    'AskOrderCutAmt_cyc',
    'AskOrderCutCnt_cyc',
    'AskOrderPasVol_cyc',
    'AskOrderPasAmt_cyc',
    'AskOrderPasCnt_cyc',
    'AskOrderNegVol_cyc',
    'AskOrderNegAmt_cyc',
    'AskOrderNegCnt_cyc',
    'AskTradeActVol_cyc',
    'AskTradeActAmt_cyc',
    'AskTradeActCnt_cyc',
    'AskTradeCutVol_cyc',
    'AskTradeCutAmt_cyc',
    'AskTradeCutCnt_cyc',
    'AskTradePasVol_cyc',
    'AskTradePasAmt_cyc',
    'AskTradePasCnt_cyc',
    'AskTradeNegVol_cyc',
    'AskTradeNegAmt_cyc',
    'AskTradeNegCnt_cyc',
    'AskCancelActVol_cyc',
    'AskCancelActAmt_cyc',
    'AskCancelActCnt_cyc',
    'AskCancelCutVol_cyc',
    'AskCancelCutAmt_cyc',
    'AskCancelCutCnt_cyc',
    'AskCancelPasVol_cyc',
    'AskCancelPasAmt_cyc',
    'AskCancelPasCnt_cyc',
    'AskCancelNegVol_cyc',
    'AskCancelNegAmt_cyc',
    'AskCancelNegCnt_cyc',
]
custom_file_path = D.get_custom_info()['root_url']
today = D.get_trading_dates()[-1]
yesterday = D.get_trading_dates()[-2]
factor_ready = np.ndarray


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


class CycAlphaV1(CrossSectionalFactorPool):
    
    def static_init():
        pass

    def __init__(self, calc_factor_names) -> None:
        super(CycAlphaV1, self).__init__(calc_factor_names)
        # 若无中间状态数据,将读取过去pre_date_cnt日基础特征去计算中间状态数据,并在当日回放结束时储存
        self.pre_date_cnt = D.get_cyc_info("pre_date_count")
        # 不要对这个东西排序！！
        symbol_array = D.get_all_symbols()  
        self.symbol_array = symbol_array
        # 储存当日票池 千万别改变顺序
        if not os.path.exists(os.path.join(custom_file_path, str(today))):
            os.makedirs(os.path.join(custom_file_path, str(today)))
        pkl.dump(self.symbol_array, open(os.path.join(custom_file_path, str(today), 'symbol.pkl'), 'wb'))
        if not os.path.exists(os.path.join(custom_file_path, str(yesterday), 'symbol.pkl')): # TODO 这里得改成判断所有因子是否存在
            self.last_symbol_array = None
        else:
            self.last_symbol_array = pkl.load(open(os.path.join(custom_file_path, str(yesterday), 'symbol.pkl'), 'rb'))
        index = pd.MultiIndex.from_product([D.get_time_cuts(),
                                            symbol_array,
                                            ],
                                           names=['date_time_cut', 'symbol'])
        self.n_jobs = D.get_cyc_info('n_jobs')
        # 初始化最终因子表 index:(date_time_cut,symbol)
        self.factor_data = pd.DataFrame(index=index, columns=self.calc_factor_names, dtype=np.float64)
        # 记录当前的因子
        self.timestamp = None  
        # 获取订阅所有因子池的信息
        self.baisc_factor_names = D.get_sub_basic_factor_names()
        # 记录收到了多少个人的基础因子
        self.feature_ready = 0
        # 当self.feature_ready == self.pool_num时,才开始计算因子
        self.pool_num = len(self.baisc_factor_names)
        # 记录每个人的特征池的字段名
        self.factor_names_map = {}
        # 初始化基础特征表
        self.data = pd.DataFrame(index=symbol_array, dtype=np.float32).rename_axis('symbol')
        for pool_name, pool_info in self.baisc_factor_names.items():
            self.factor_names_map[pool_name] = [x for x in pool_info['factor_names'] if x != 'is_ok']
            self.data.loc[:, self.factor_names_map[pool_name]] = np.nan

    # 盘前初始化
    def factor_init(self):
        n = self.pre_date_cnt
        pre_update_rows = int(n * 237)
        # 如果没有昨日信息,那么意味着当天要加载过去信息并算出因子
        if self.last_symbol_array is None:
            # region 加载过去n日数据
            pre_date = D.get_trading_dates()[-n-1:-1]
            cap_date = D.get_trading_dates()[-n-2:-2]
            cap_data = D.get_stock_capital_series(start_date=cap_date[0], end_date=cap_date[-1], fields=['market_cap'])
            cap_data.columns = [x + '_cyc' for x in cap_data.columns]
            cap_data.reset_index(inplace=True)
            cap_data.rename(columns={"timestamp": 'date'}, inplace=True)
            cap_data['date'] = cap_data['date'].map(dict(zip(cap_date,pre_date)))
            cap_data['date'] = cap_data['date'].map(str)
            pre_data = []
            logger.info(f"No previous day factor instance file found, therefore loading the past {n} days of data to calculate historical factors")
            for name, fields in self.factor_names_map.items():
                pre_data.append(D.qry_his_basic_factors(
                    pool=name,
                    trading_dates=pre_date, 
                    symbols=self.symbol_array, 
                    factor_names=fields,
                    ))
            pre_data = pd.concat(pre_data)
            pre_data['date'] = pre_data.index.get_level_values('date_time_cut').astype(str).str.slice(0,8)
            pre_data['ticker'] = pre_data.index.get_level_values('symbol')
            pre_data = pd.merge(pre_data.reset_index(), cap_data, how='left', on=['date', 'ticker'])
            pre_data.drop(['date', 'ticker'], axis=1, inplace=True)
            pre_data.set_index(['date_time_cut', 'symbol'], inplace=True)
            logger.info(f"Loading complete")
            # endregion

            # region 计算衍生基础特征
            pre_data['BidOrderActVol_cyc'] = pre_data.BidOrderSmallActVol_cyc + pre_data.BidOrderLargeActVol_cyc
            pre_data['BidOrderCutVol_cyc'] = pre_data.BidOrderSmallCutVol_cyc + pre_data.BidOrderLargeCutVol_cyc
            pre_data['BidOrderPasVol_cyc'] = pre_data.BidOrderSmallPasVol_cyc + pre_data.BidOrderLargePasVol_cyc
            pre_data['BidOrderNegVol_cyc'] = pre_data.BidOrderSmallNegVol_cyc + pre_data.BidOrderLargeNegVol_cyc
            pre_data['BidOrderActAmt_cyc'] = pre_data.BidOrderSmallActAmt_cyc + pre_data.BidOrderLargeActAmt_cyc
            pre_data['BidOrderCutAmt_cyc'] = pre_data.BidOrderSmallCutAmt_cyc + pre_data.BidOrderLargeCutAmt_cyc
            pre_data['BidOrderPasAmt_cyc'] = pre_data.BidOrderSmallPasAmt_cyc + pre_data.BidOrderLargePasAmt_cyc
            pre_data['BidOrderNegAmt_cyc'] = pre_data.BidOrderSmallNegAmt_cyc + pre_data.BidOrderLargeNegAmt_cyc
            pre_data['BidOrderActCnt_cyc'] = pre_data.BidOrderSmallActCnt_cyc + pre_data.BidOrderLargeActCnt_cyc
            pre_data['BidOrderCutCnt_cyc'] = pre_data.BidOrderSmallCutCnt_cyc + pre_data.BidOrderLargeCutCnt_cyc
            pre_data['BidOrderPasCnt_cyc'] = pre_data.BidOrderSmallPasCnt_cyc + pre_data.BidOrderLargePasCnt_cyc
            pre_data['BidOrderNegCnt_cyc'] = pre_data.BidOrderSmallNegCnt_cyc + pre_data.BidOrderLargeNegCnt_cyc
            pre_data['AskOrderActVol_cyc'] = pre_data.AskOrderSmallActVol_cyc + pre_data.AskOrderLargeActVol_cyc
            pre_data['AskOrderCutVol_cyc'] = pre_data.AskOrderSmallCutVol_cyc + pre_data.AskOrderLargeCutVol_cyc
            pre_data['AskOrderPasVol_cyc'] = pre_data.AskOrderSmallPasVol_cyc + pre_data.AskOrderLargePasVol_cyc
            pre_data['AskOrderNegVol_cyc'] = pre_data.AskOrderSmallNegVol_cyc + pre_data.AskOrderLargeNegVol_cyc
            pre_data['AskOrderActAmt_cyc'] = pre_data.AskOrderSmallActAmt_cyc + pre_data.AskOrderLargeActAmt_cyc
            pre_data['AskOrderCutAmt_cyc'] = pre_data.AskOrderSmallCutAmt_cyc + pre_data.AskOrderLargeCutAmt_cyc
            pre_data['AskOrderPasAmt_cyc'] = pre_data.AskOrderSmallPasAmt_cyc + pre_data.AskOrderLargePasAmt_cyc
            pre_data['AskOrderNegAmt_cyc'] = pre_data.AskOrderSmallNegAmt_cyc + pre_data.AskOrderLargeNegAmt_cyc
            pre_data['AskOrderActCnt_cyc'] = pre_data.AskOrderSmallActCnt_cyc + pre_data.AskOrderLargeActCnt_cyc
            pre_data['AskOrderCutCnt_cyc'] = pre_data.AskOrderSmallCutCnt_cyc + pre_data.AskOrderLargeCutCnt_cyc
            pre_data['AskOrderPasCnt_cyc'] = pre_data.AskOrderSmallPasCnt_cyc + pre_data.AskOrderLargePasCnt_cyc
            pre_data['AskOrderNegCnt_cyc'] = pre_data.AskOrderSmallNegCnt_cyc + pre_data.AskOrderLargeNegCnt_cyc
            pre_data['BidTradeActVol_cyc'] = pre_data.BidTradeSmallActVol_cyc + pre_data.BidTradeLargeActVol_cyc
            pre_data['BidTradeCutVol_cyc'] = pre_data.BidTradeSmallCutVol_cyc + pre_data.BidTradeLargeCutVol_cyc
            pre_data['BidTradePasVol_cyc'] = pre_data.BidTradeSmallPasVol_cyc + pre_data.BidTradeLargePasVol_cyc
            pre_data['BidTradeNegVol_cyc'] = pre_data.BidTradeSmallNegVol_cyc + pre_data.BidTradeLargeNegVol_cyc
            pre_data['BidTradeActAmt_cyc'] = pre_data.BidTradeSmallActAmt_cyc + pre_data.BidTradeLargeActAmt_cyc
            pre_data['BidTradeCutAmt_cyc'] = pre_data.BidTradeSmallCutAmt_cyc + pre_data.BidTradeLargeCutAmt_cyc
            pre_data['BidTradePasAmt_cyc'] = pre_data.BidTradeSmallPasAmt_cyc + pre_data.BidTradeLargePasAmt_cyc
            pre_data['BidTradeNegAmt_cyc'] = pre_data.BidTradeSmallNegAmt_cyc + pre_data.BidTradeLargeNegAmt_cyc
            pre_data['BidTradeActCnt_cyc'] = pre_data.BidTradeSmallActCnt_cyc + pre_data.BidTradeLargeActCnt_cyc
            pre_data['BidTradeCutCnt_cyc'] = pre_data.BidTradeSmallCutCnt_cyc + pre_data.BidTradeLargeCutCnt_cyc
            pre_data['BidTradePasCnt_cyc'] = pre_data.BidTradeSmallPasCnt_cyc + pre_data.BidTradeLargePasCnt_cyc
            pre_data['BidTradeNegCnt_cyc'] = pre_data.BidTradeSmallNegCnt_cyc + pre_data.BidTradeLargeNegCnt_cyc
            pre_data['AskTradeActVol_cyc'] = pre_data.AskTradeSmallActVol_cyc + pre_data.AskTradeLargeActVol_cyc
            pre_data['AskTradeCutVol_cyc'] = pre_data.AskTradeSmallCutVol_cyc + pre_data.AskTradeLargeCutVol_cyc
            pre_data['AskTradePasVol_cyc'] = pre_data.AskTradeSmallPasVol_cyc + pre_data.AskTradeLargePasVol_cyc
            pre_data['AskTradeNegVol_cyc'] = pre_data.AskTradeSmallNegVol_cyc + pre_data.AskTradeLargeNegVol_cyc
            pre_data['AskTradeActAmt_cyc'] = pre_data.AskTradeSmallActAmt_cyc + pre_data.AskTradeLargeActAmt_cyc
            pre_data['AskTradeCutAmt_cyc'] = pre_data.AskTradeSmallCutAmt_cyc + pre_data.AskTradeLargeCutAmt_cyc
            pre_data['AskTradePasAmt_cyc'] = pre_data.AskTradeSmallPasAmt_cyc + pre_data.AskTradeLargePasAmt_cyc
            pre_data['AskTradeNegAmt_cyc'] = pre_data.AskTradeSmallNegAmt_cyc + pre_data.AskTradeLargeNegAmt_cyc
            pre_data['AskTradeActCnt_cyc'] = pre_data.AskTradeSmallActCnt_cyc + pre_data.AskTradeLargeActCnt_cyc
            pre_data['AskTradeCutCnt_cyc'] = pre_data.AskTradeSmallCutCnt_cyc + pre_data.AskTradeLargeCutCnt_cyc
            pre_data['AskTradePasCnt_cyc'] = pre_data.AskTradeSmallPasCnt_cyc + pre_data.AskTradeLargePasCnt_cyc
            pre_data['AskTradeNegCnt_cyc'] = pre_data.AskTradeSmallNegCnt_cyc + pre_data.AskTradeLargeNegCnt_cyc
            # Small & Large
            pre_data['BidOrderSmallVol_cyc'] = pre_data.BidOrderSmallActVol_cyc + pre_data.BidOrderSmallCutVol_cyc + pre_data.BidOrderSmallPasVol_cyc + pre_data.BidOrderSmallNegVol_cyc
            pre_data['BidOrderLargeVol_cyc'] = pre_data.BidOrderLargeActVol_cyc + pre_data.BidOrderLargeCutVol_cyc + pre_data.BidOrderLargePasVol_cyc + pre_data.BidOrderLargeNegVol_cyc
            pre_data['BidOrderSmallAmt_cyc'] = pre_data.BidOrderSmallActAmt_cyc + pre_data.BidOrderSmallCutAmt_cyc + pre_data.BidOrderSmallPasAmt_cyc + pre_data.BidOrderSmallNegAmt_cyc
            pre_data['BidOrderLargeAmt_cyc'] = pre_data.BidOrderLargeActAmt_cyc + pre_data.BidOrderLargeCutAmt_cyc + pre_data.BidOrderLargePasAmt_cyc + pre_data.BidOrderLargeNegAmt_cyc
            pre_data['BidOrderSmallCnt_cyc'] = pre_data.BidOrderSmallActCnt_cyc + pre_data.BidOrderSmallCutCnt_cyc + pre_data.BidOrderSmallPasCnt_cyc + pre_data.BidOrderSmallNegCnt_cyc
            pre_data['BidOrderLargeCnt_cyc'] = pre_data.BidOrderLargeActCnt_cyc + pre_data.BidOrderLargeCutCnt_cyc + pre_data.BidOrderLargePasCnt_cyc + pre_data.BidOrderLargeNegCnt_cyc
            pre_data['AskOrderSmallVol_cyc'] = pre_data.AskOrderSmallActVol_cyc + pre_data.AskOrderSmallCutVol_cyc + pre_data.AskOrderSmallPasVol_cyc + pre_data.AskOrderSmallNegVol_cyc
            pre_data['AskOrderLargeVol_cyc'] = pre_data.AskOrderLargeActVol_cyc + pre_data.AskOrderLargeCutVol_cyc + pre_data.AskOrderLargePasVol_cyc + pre_data.AskOrderLargeNegVol_cyc
            pre_data['AskOrderSmallAmt_cyc'] = pre_data.AskOrderSmallActAmt_cyc + pre_data.AskOrderSmallCutAmt_cyc + pre_data.AskOrderSmallPasAmt_cyc + pre_data.AskOrderSmallNegAmt_cyc
            pre_data['AskOrderLargeAmt_cyc'] = pre_data.AskOrderLargeActAmt_cyc + pre_data.AskOrderLargeCutAmt_cyc + pre_data.AskOrderLargePasAmt_cyc + pre_data.AskOrderLargeNegAmt_cyc
            pre_data['AskOrderSmallCnt_cyc'] = pre_data.AskOrderSmallActCnt_cyc + pre_data.AskOrderSmallCutCnt_cyc + pre_data.AskOrderSmallPasCnt_cyc + pre_data.AskOrderSmallNegCnt_cyc
            pre_data['AskOrderLargeCnt_cyc'] = pre_data.AskOrderLargeActCnt_cyc + pre_data.AskOrderLargeCutCnt_cyc + pre_data.AskOrderLargePasCnt_cyc + pre_data.AskOrderLargeNegCnt_cyc
            pre_data['BidTradeSmallVol_cyc'] = pre_data.BidTradeSmallActVol_cyc + pre_data.BidTradeSmallCutVol_cyc + pre_data.BidTradeSmallPasVol_cyc + pre_data.BidTradeSmallNegVol_cyc
            pre_data['BidTradeLargeVol_cyc'] = pre_data.BidTradeLargeActVol_cyc + pre_data.BidTradeLargeCutVol_cyc + pre_data.BidTradeLargePasVol_cyc + pre_data.BidTradeLargeNegVol_cyc
            pre_data['BidTradeSmallAmt_cyc'] = pre_data.BidTradeSmallActAmt_cyc + pre_data.BidTradeSmallCutAmt_cyc + pre_data.BidTradeSmallPasAmt_cyc + pre_data.BidTradeSmallNegAmt_cyc
            pre_data['BidTradeLargeAmt_cyc'] = pre_data.BidTradeLargeActAmt_cyc + pre_data.BidTradeLargeCutAmt_cyc + pre_data.BidTradeLargePasAmt_cyc + pre_data.BidTradeLargeNegAmt_cyc
            pre_data['BidTradeSmallCnt_cyc'] = pre_data.BidTradeSmallActCnt_cyc + pre_data.BidTradeSmallCutCnt_cyc + pre_data.BidTradeSmallPasCnt_cyc + pre_data.BidTradeSmallNegCnt_cyc
            pre_data['BidTradeLargeCnt_cyc'] = pre_data.BidTradeLargeActCnt_cyc + pre_data.BidTradeLargeCutCnt_cyc + pre_data.BidTradeLargePasCnt_cyc + pre_data.BidTradeLargeNegCnt_cyc
            pre_data['AskTradeSmallVol_cyc'] = pre_data.AskTradeSmallActVol_cyc + pre_data.AskTradeSmallCutVol_cyc + pre_data.AskTradeSmallPasVol_cyc + pre_data.AskTradeSmallNegVol_cyc
            pre_data['AskTradeLargeVol_cyc'] = pre_data.AskTradeLargeActVol_cyc + pre_data.AskTradeLargeCutVol_cyc + pre_data.AskTradeLargePasVol_cyc + pre_data.AskTradeLargeNegVol_cyc
            pre_data['AskTradeSmallAmt_cyc'] = pre_data.AskTradeSmallActAmt_cyc + pre_data.AskTradeSmallCutAmt_cyc + pre_data.AskTradeSmallPasAmt_cyc + pre_data.AskTradeSmallNegAmt_cyc
            pre_data['AskTradeLargeAmt_cyc'] = pre_data.AskTradeLargeActAmt_cyc + pre_data.AskTradeLargeCutAmt_cyc + pre_data.AskTradeLargePasAmt_cyc + pre_data.AskTradeLargeNegAmt_cyc
            pre_data['AskTradeSmallCnt_cyc'] = pre_data.AskTradeSmallActCnt_cyc + pre_data.AskTradeSmallCutCnt_cyc + pre_data.AskTradeSmallPasCnt_cyc + pre_data.AskTradeSmallNegCnt_cyc
            pre_data['AskTradeLargeCnt_cyc'] = pre_data.AskTradeLargeActCnt_cyc + pre_data.AskTradeLargeCutCnt_cyc + pre_data.AskTradeLargePasCnt_cyc + pre_data.AskTradeLargeNegCnt_cyc
            # endregion
        
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
                factor_names=self.calc_factor_names,    # 最终因子名
                dtypes=pre_data.dtypes.tolist(),        # 基础因子类型
                symbols=self.symbol_array,              # 当日股票池
                n_jobs=D.get_cyc_info('n_jobs'),        # 多进程数量
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
            basic_fator_fields.append('market_cap_cyc')
            dtypes = [np.float32 for _ in basic_fator_fields]
            self.alpha_engine = FactorEngine(
                stock_nums=len(self.symbol_array),      # 股票数量
                basic_factor_names=basic_fator_fields,  # 基础因子名
                factor_names=self.calc_factor_names,    # 最终因子名
                dtypes=dtypes,                          # 基础因子类型
                symbols=self.symbol_array,              # 当日股票池
                n_jobs=D.get_cyc_info('n_jobs'),        # 多进程数量
                pre_update_cnt=pre_update_rows,         # 预先更新的样本行数
                last_symbols=self.last_symbol_array,    # 昨日股票池
                )
            # endregion
        # region 历史数据计算完后 读取今日的eod数据
        cap_data = D.get_stock_capital_series(start_date=yesterday, end_date=yesterday)
        cap_data.reset_index(level='timestamp', drop=True, inplace=True)
        cap_data.rename_axis('symbol', inplace=True)
        cap_data = cap_data[['market_cap']]
        cap_data.columns = ['market_cap_cyc']
        add_s = set(D.get_all_symbols()).difference(set(cap_data.index))
        if add_s:
            add_d = pd.DataFrame(index=pd.Index(add_s), columns=['market_cap_cyc'])
            cap_data = pd.concat([cap_data, add_d]).rename_axis('symbol')
        del_s = set(cap_data.index).difference(set(D.get_all_symbols()))
        if del_s:
            cap_data.drop(pd.Index(del_s), axis=0, inplace=True)
        cap_data = cap_data.loc[D.get_all_symbols(), :]
        # endregion
       
        # region 更新当日市值信息
        fe.DC.update_shared_memory(cap_data)
        # endregion

    def get_factors(self):
        return 0, self.res

    def on_end(self):
        """当天交易时间结束,会通知所有因子
        """
        # 当前因子视图转为非共享内存 防止内存泄漏
        self.res = deepcopy(self.factor_data)
        # 因子计算引擎 等待所有子进程储存最终因子类
        self.alpha_engine.wait_process_end()
        # 关闭共享内存
        fe.DC.close()
        # ======================
        logger.info('end===!!!!')

    def on_rtn_remote_factors(self, date_time_cut: int,
                              pool_name: str, data: pd.DataFrame):
        # 获取当前时间戳
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
            self.factor_data.loc[(date_time_cut, slice(None)), :] = temp_data.values
        self.feature_ready = 0
        logger.debug(f"========{date_time_cut}==={pool_name}")

    def cal_dev(self):
        self.data['BidOrderActVol_cyc'] = self.data.BidOrderSmallActVol_cyc + self.data.BidOrderLargeActVol_cyc
        self.data['BidOrderCutVol_cyc'] = self.data.BidOrderSmallCutVol_cyc + self.data.BidOrderLargeCutVol_cyc
        self.data['BidOrderPasVol_cyc'] = self.data.BidOrderSmallPasVol_cyc + self.data.BidOrderLargePasVol_cyc
        self.data['BidOrderNegVol_cyc'] = self.data.BidOrderSmallNegVol_cyc + self.data.BidOrderLargeNegVol_cyc
        self.data['BidOrderActAmt_cyc'] = self.data.BidOrderSmallActAmt_cyc + self.data.BidOrderLargeActAmt_cyc
        self.data['BidOrderCutAmt_cyc'] = self.data.BidOrderSmallCutAmt_cyc + self.data.BidOrderLargeCutAmt_cyc
        self.data['BidOrderPasAmt_cyc'] = self.data.BidOrderSmallPasAmt_cyc + self.data.BidOrderLargePasAmt_cyc
        self.data['BidOrderNegAmt_cyc'] = self.data.BidOrderSmallNegAmt_cyc + self.data.BidOrderLargeNegAmt_cyc
        self.data['BidOrderActCnt_cyc'] = self.data.BidOrderSmallActCnt_cyc + self.data.BidOrderLargeActCnt_cyc
        self.data['BidOrderCutCnt_cyc'] = self.data.BidOrderSmallCutCnt_cyc + self.data.BidOrderLargeCutCnt_cyc
        self.data['BidOrderPasCnt_cyc'] = self.data.BidOrderSmallPasCnt_cyc + self.data.BidOrderLargePasCnt_cyc
        self.data['BidOrderNegCnt_cyc'] = self.data.BidOrderSmallNegCnt_cyc + self.data.BidOrderLargeNegCnt_cyc
        self.data['AskOrderActVol_cyc'] = self.data.AskOrderSmallActVol_cyc + self.data.AskOrderLargeActVol_cyc
        self.data['AskOrderCutVol_cyc'] = self.data.AskOrderSmallCutVol_cyc + self.data.AskOrderLargeCutVol_cyc
        self.data['AskOrderPasVol_cyc'] = self.data.AskOrderSmallPasVol_cyc + self.data.AskOrderLargePasVol_cyc
        self.data['AskOrderNegVol_cyc'] = self.data.AskOrderSmallNegVol_cyc + self.data.AskOrderLargeNegVol_cyc
        self.data['AskOrderActAmt_cyc'] = self.data.AskOrderSmallActAmt_cyc + self.data.AskOrderLargeActAmt_cyc
        self.data['AskOrderCutAmt_cyc'] = self.data.AskOrderSmallCutAmt_cyc + self.data.AskOrderLargeCutAmt_cyc
        self.data['AskOrderPasAmt_cyc'] = self.data.AskOrderSmallPasAmt_cyc + self.data.AskOrderLargePasAmt_cyc
        self.data['AskOrderNegAmt_cyc'] = self.data.AskOrderSmallNegAmt_cyc + self.data.AskOrderLargeNegAmt_cyc
        self.data['AskOrderActCnt_cyc'] = self.data.AskOrderSmallActCnt_cyc + self.data.AskOrderLargeActCnt_cyc
        self.data['AskOrderCutCnt_cyc'] = self.data.AskOrderSmallCutCnt_cyc + self.data.AskOrderLargeCutCnt_cyc
        self.data['AskOrderPasCnt_cyc'] = self.data.AskOrderSmallPasCnt_cyc + self.data.AskOrderLargePasCnt_cyc
        self.data['AskOrderNegCnt_cyc'] = self.data.AskOrderSmallNegCnt_cyc + self.data.AskOrderLargeNegCnt_cyc
        self.data['BidTradeActVol_cyc'] = self.data.BidTradeSmallActVol_cyc + self.data.BidTradeLargeActVol_cyc
        self.data['BidTradeCutVol_cyc'] = self.data.BidTradeSmallCutVol_cyc + self.data.BidTradeLargeCutVol_cyc
        self.data['BidTradePasVol_cyc'] = self.data.BidTradeSmallPasVol_cyc + self.data.BidTradeLargePasVol_cyc
        self.data['BidTradeNegVol_cyc'] = self.data.BidTradeSmallNegVol_cyc + self.data.BidTradeLargeNegVol_cyc
        self.data['BidTradeActAmt_cyc'] = self.data.BidTradeSmallActAmt_cyc + self.data.BidTradeLargeActAmt_cyc
        self.data['BidTradeCutAmt_cyc'] = self.data.BidTradeSmallCutAmt_cyc + self.data.BidTradeLargeCutAmt_cyc
        self.data['BidTradePasAmt_cyc'] = self.data.BidTradeSmallPasAmt_cyc + self.data.BidTradeLargePasAmt_cyc
        self.data['BidTradeNegAmt_cyc'] = self.data.BidTradeSmallNegAmt_cyc + self.data.BidTradeLargeNegAmt_cyc
        self.data['BidTradeActCnt_cyc'] = self.data.BidTradeSmallActCnt_cyc + self.data.BidTradeLargeActCnt_cyc
        self.data['BidTradeCutCnt_cyc'] = self.data.BidTradeSmallCutCnt_cyc + self.data.BidTradeLargeCutCnt_cyc
        self.data['BidTradePasCnt_cyc'] = self.data.BidTradeSmallPasCnt_cyc + self.data.BidTradeLargePasCnt_cyc
        self.data['BidTradeNegCnt_cyc'] = self.data.BidTradeSmallNegCnt_cyc + self.data.BidTradeLargeNegCnt_cyc
        self.data['AskTradeActVol_cyc'] = self.data.AskTradeSmallActVol_cyc + self.data.AskTradeLargeActVol_cyc
        self.data['AskTradeCutVol_cyc'] = self.data.AskTradeSmallCutVol_cyc + self.data.AskTradeLargeCutVol_cyc
        self.data['AskTradePasVol_cyc'] = self.data.AskTradeSmallPasVol_cyc + self.data.AskTradeLargePasVol_cyc
        self.data['AskTradeNegVol_cyc'] = self.data.AskTradeSmallNegVol_cyc + self.data.AskTradeLargeNegVol_cyc
        self.data['AskTradeActAmt_cyc'] = self.data.AskTradeSmallActAmt_cyc + self.data.AskTradeLargeActAmt_cyc
        self.data['AskTradeCutAmt_cyc'] = self.data.AskTradeSmallCutAmt_cyc + self.data.AskTradeLargeCutAmt_cyc
        self.data['AskTradePasAmt_cyc'] = self.data.AskTradeSmallPasAmt_cyc + self.data.AskTradeLargePasAmt_cyc
        self.data['AskTradeNegAmt_cyc'] = self.data.AskTradeSmallNegAmt_cyc + self.data.AskTradeLargeNegAmt_cyc
        self.data['AskTradeActCnt_cyc'] = self.data.AskTradeSmallActCnt_cyc + self.data.AskTradeLargeActCnt_cyc
        self.data['AskTradeCutCnt_cyc'] = self.data.AskTradeSmallCutCnt_cyc + self.data.AskTradeLargeCutCnt_cyc
        self.data['AskTradePasCnt_cyc'] = self.data.AskTradeSmallPasCnt_cyc + self.data.AskTradeLargePasCnt_cyc
        self.data['AskTradeNegCnt_cyc'] = self.data.AskTradeSmallNegCnt_cyc + self.data.AskTradeLargeNegCnt_cyc
        # Small & Large
        self.data['BidOrderSmallVol_cyc'] = self.data.BidOrderSmallActVol_cyc + self.data.BidOrderSmallCutVol_cyc + self.data.BidOrderSmallPasVol_cyc + self.data.BidOrderSmallNegVol_cyc
        self.data['BidOrderLargeVol_cyc'] = self.data.BidOrderLargeActVol_cyc + self.data.BidOrderLargeCutVol_cyc + self.data.BidOrderLargePasVol_cyc + self.data.BidOrderLargeNegVol_cyc
        self.data['BidOrderSmallAmt_cyc'] = self.data.BidOrderSmallActAmt_cyc + self.data.BidOrderSmallCutAmt_cyc + self.data.BidOrderSmallPasAmt_cyc + self.data.BidOrderSmallNegAmt_cyc
        self.data['BidOrderLargeAmt_cyc'] = self.data.BidOrderLargeActAmt_cyc + self.data.BidOrderLargeCutAmt_cyc + self.data.BidOrderLargePasAmt_cyc + self.data.BidOrderLargeNegAmt_cyc
        self.data['BidOrderSmallCnt_cyc'] = self.data.BidOrderSmallActCnt_cyc + self.data.BidOrderSmallCutCnt_cyc + self.data.BidOrderSmallPasCnt_cyc + self.data.BidOrderSmallNegCnt_cyc
        self.data['BidOrderLargeCnt_cyc'] = self.data.BidOrderLargeActCnt_cyc + self.data.BidOrderLargeCutCnt_cyc + self.data.BidOrderLargePasCnt_cyc + self.data.BidOrderLargeNegCnt_cyc
        self.data['AskOrderSmallVol_cyc'] = self.data.AskOrderSmallActVol_cyc + self.data.AskOrderSmallCutVol_cyc + self.data.AskOrderSmallPasVol_cyc + self.data.AskOrderSmallNegVol_cyc
        self.data['AskOrderLargeVol_cyc'] = self.data.AskOrderLargeActVol_cyc + self.data.AskOrderLargeCutVol_cyc + self.data.AskOrderLargePasVol_cyc + self.data.AskOrderLargeNegVol_cyc
        self.data['AskOrderSmallAmt_cyc'] = self.data.AskOrderSmallActAmt_cyc + self.data.AskOrderSmallCutAmt_cyc + self.data.AskOrderSmallPasAmt_cyc + self.data.AskOrderSmallNegAmt_cyc
        self.data['AskOrderLargeAmt_cyc'] = self.data.AskOrderLargeActAmt_cyc + self.data.AskOrderLargeCutAmt_cyc + self.data.AskOrderLargePasAmt_cyc + self.data.AskOrderLargeNegAmt_cyc
        self.data['AskOrderSmallCnt_cyc'] = self.data.AskOrderSmallActCnt_cyc + self.data.AskOrderSmallCutCnt_cyc + self.data.AskOrderSmallPasCnt_cyc + self.data.AskOrderSmallNegCnt_cyc
        self.data['AskOrderLargeCnt_cyc'] = self.data.AskOrderLargeActCnt_cyc + self.data.AskOrderLargeCutCnt_cyc + self.data.AskOrderLargePasCnt_cyc + self.data.AskOrderLargeNegCnt_cyc
        self.data['BidTradeSmallVol_cyc'] = self.data.BidTradeSmallActVol_cyc + self.data.BidTradeSmallCutVol_cyc + self.data.BidTradeSmallPasVol_cyc + self.data.BidTradeSmallNegVol_cyc
        self.data['BidTradeLargeVol_cyc'] = self.data.BidTradeLargeActVol_cyc + self.data.BidTradeLargeCutVol_cyc + self.data.BidTradeLargePasVol_cyc + self.data.BidTradeLargeNegVol_cyc
        self.data['BidTradeSmallAmt_cyc'] = self.data.BidTradeSmallActAmt_cyc + self.data.BidTradeSmallCutAmt_cyc + self.data.BidTradeSmallPasAmt_cyc + self.data.BidTradeSmallNegAmt_cyc
        self.data['BidTradeLargeAmt_cyc'] = self.data.BidTradeLargeActAmt_cyc + self.data.BidTradeLargeCutAmt_cyc + self.data.BidTradeLargePasAmt_cyc + self.data.BidTradeLargeNegAmt_cyc
        self.data['BidTradeSmallCnt_cyc'] = self.data.BidTradeSmallActCnt_cyc + self.data.BidTradeSmallCutCnt_cyc + self.data.BidTradeSmallPasCnt_cyc + self.data.BidTradeSmallNegCnt_cyc
        self.data['BidTradeLargeCnt_cyc'] = self.data.BidTradeLargeActCnt_cyc + self.data.BidTradeLargeCutCnt_cyc + self.data.BidTradeLargePasCnt_cyc + self.data.BidTradeLargeNegCnt_cyc
        self.data['AskTradeSmallVol_cyc'] = self.data.AskTradeSmallActVol_cyc + self.data.AskTradeSmallCutVol_cyc + self.data.AskTradeSmallPasVol_cyc + self.data.AskTradeSmallNegVol_cyc
        self.data['AskTradeLargeVol_cyc'] = self.data.AskTradeLargeActVol_cyc + self.data.AskTradeLargeCutVol_cyc + self.data.AskTradeLargePasVol_cyc + self.data.AskTradeLargeNegVol_cyc
        self.data['AskTradeSmallAmt_cyc'] = self.data.AskTradeSmallActAmt_cyc + self.data.AskTradeSmallCutAmt_cyc + self.data.AskTradeSmallPasAmt_cyc + self.data.AskTradeSmallNegAmt_cyc
        self.data['AskTradeLargeAmt_cyc'] = self.data.AskTradeLargeActAmt_cyc + self.data.AskTradeLargeCutAmt_cyc + self.data.AskTradeLargePasAmt_cyc + self.data.AskTradeLargeNegAmt_cyc
        self.data['AskTradeSmallCnt_cyc'] = self.data.AskTradeSmallActCnt_cyc + self.data.AskTradeSmallCutCnt_cyc + self.data.AskTradeSmallPasCnt_cyc + self.data.AskTradeSmallNegCnt_cyc
        self.data['AskTradeLargeCnt_cyc'] = self.data.AskTradeLargeActCnt_cyc + self.data.AskTradeLargeCutCnt_cyc + self.data.AskTradeLargePasCnt_cyc + self.data.AskTradeLargeNegCnt_cyc