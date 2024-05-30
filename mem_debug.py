# 测试
from dl_helper.trains.t20240527 import trainer
from dl_helper.train_param import params
from memory_profiler import profile

# 初始化
t = trainer(0, debug=True)# binctabl_10d_v0

# 1
# 使用上传数据
t.init_param(r"D:\code\featrue_data\notebook\20240413_滚动标准化\20240521")

# 2
# 下载数据 <tg>
# t.init_param()
# ses = '1BVtsOKABu6pKio99jf7uqjfe5FMXfzPbEDzB1N5DFaXkEu5Og5dJre4xg4rbXdjRQB7HpWw7g-fADK6AVDnw7nZ1ykiC5hfq-IjDVPsMhD7Sffuv0lTGa4-1Dz2MktHs3e_mXpL1hNMFgNm5512K1BWQvij3xkoiHGKDqXLYzbzeVMr5e230JY7yozEZRylDB_AuFeBGDjLcwattWnuX2mnTZWgs-lS1A_kZWomGl3HqV84UsoJlk9b-GAbzH-jBunsckkjUijri6OBscvzpIWO7Kgq0YzxJvZe_a1N8SFG3Gbuq0mIOkN3JNKGTmYLjTClQd2PIJuFSxzYFPQJwXIWZlFg0O2U='
# await t.download_dataset_async(ses)

# 训练
t.train()