import pickle, time, copy
from dl_helper.rl.custom_imitation_module.rollout import KEYS

class Test:
    def __init__(self):
        flle = r"D:\L2_DATA_T0_ETF\train_data\RAW\BC_train_data\bc_train_data_0\1.pkl"
        with open(flle, 'rb') as f:
            self.data = pickle.load(f)
    
    def __getitem__(self, idx):
        res = {key: getattr(self.data, key)[idx] for key in KEYS}
        return copy.deepcopy(res)

if __name__ == '__main__':
    test = Test()
    ts = []

    # 平均耗时: 9.50 秒
    for i in range(10):
        print(f'第 {i} 次')
        t = time.time()
        for i in range(5000000):
            res = test[0]
        cost = time.time() - t
        ts.append(cost)

    print(f'平均耗时: {sum(ts) / len(ts):.2f} 秒')