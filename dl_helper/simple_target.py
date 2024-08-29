
import pickle
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
def yfunc(y):
    if y > 0.5:
        return 0
    elif y<-0.5:
        return 1
    else:
        return 2
def produce_simple_predict_file(folder):
    output_folder = os.path.join(folder, 'simple_predict')
    os.makedirs(output_folder, exist_ok=True)

    # 遍历所有文件夹
    for f in ['train', 'val', 'test']:
        _folder = os.path.join(folder, f)
        for file in tqdm(os.listdir(_folder)):
            if file.endswith('.pkl'):
                _date = file.replace('.pkl', '')
                output_date_folder = os.path.join(output_folder, _date)
                os.makedirs(output_date_folder, exist_ok=True)

                # 读取文件
                _ids, _, _, _y, _ = pickle.load(open(os.path.join(_folder, file), 'rb'))

                code_datas = {}

                for i in range(len(_ids)):
                    code = _ids[i].split('_')[0]
                    timestamp = _ids[i].split('_')[1]
                    if code not in code_datas:
                        code_datas[code] = []
                    code_datas[code].append((timestamp, int(yfunc(_y[i][0]))))

                # 写入文件
                for code in code_datas:
                    timestamps = [i[0] for i in code_datas[code]]
                    begin = min(timestamps)
                    end = max(timestamps)

                    out_file = os.path.join(output_date_folder, f'{code}_{begin}_{end}.csv')
                    with open(out_file, 'w') as f:
                        # 列名
                        f.write('timestamp,target,predict\n')
                        for timestamp, y in code_datas[code]:
                            f.write(f'{timestamp},{y},{y}\n')



if __name__ == '__main__':
    train_data_folder = r'Z:\L2_DATA\train_data\filter_extra_100w_2'
    produce_simple_predict_file(train_data_folder)

