import gc, sys

def log(msg):
    print(msg)

def debug_mem():
    log('*'* 60)
    obj_list = []
    for obj in gc.get_objects():
        size = sys.getsizeof(obj)
        obj_list.append((obj, size))

    sorted_objs = sorted(obj_list, key=lambda x: x[1], reverse=True)

    msg = ['']
    for obj, size in sorted_objs[:10]:
        msg.append(f'OBJ:{id(obj)} TYPE:{type(obj)} SIZE:{size/1024/1024:.2f}MB REPR:{str(obj)[:200]}')
        referrers = gc.get_referrers(obj)
        for ref in referrers:
            msg.append(f'   {str(ref)[:300]}')

    msg_str = '\n'.join(msg)
    log(msg_str)

# 创建对象
class MyClass:
    pass

obj = MyClass()
global_list = [obj]  # 全局引用

# 删除局部引用
del obj

# 强制垃圾回收
gc.collect()

debug_mem()