import objgraph

def take_snapshot(types):
    """
    获取指定 types 的当前对象集合。
    返回 {type_name: set(对象id)} 结构
    """
    snapshot = {}
    for t in types:
        objs = objgraph.by_type(t)
        snapshot[t] = set(id(o) for o in objs)
    return snapshot

def diff_snapshot(before, after):
    """
    比较前后两个快照，找出新增的对象 id
    返回 {type_name: list(新增对象)}
    """
    growth = {}
    for t in before:
        new_ids = after[t] - before[t]
        if new_ids:
            # 重新从 id 找回对象
            objs = [o for o in objgraph.by_type(t) if id(o) in new_ids]
            growth[t] = objs
    return growth


snapshot = None
def debug_growth():
    global snapshot
    objgraph.show_growth()
    result = objgraph.growth()
    if result:
        width = max(len(name) for name, _, _ in result)
        for name, count, delta in result:
            print('%-*s%9d %+9d\n' % (width, name, count, delta))

    watch_types = ['tuple', 'dict']
    before = snapshot
    after = take_snapshot(watch_types)

    if before is not None:
        growth = diff_snapshot(before, after)
        for t, objs in growth.items():
            print(f"\n类型 {t} 新增了 {len(objs)} 个对象")
            for o in objs:
                print(f"  -> {repr(o)}")
    snapshot = after

a = []
debug_growth()
for i in range(5):
    a.append([i])
    debug_growth()
    print()