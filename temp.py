from pympler import tracker, muppy, summary
import sys

# 1. 在测试前先获取基线内存状态
baseline_objects = muppy.get_objects()
baseline_lists = muppy.filter(baseline_objects, Type=list)

# 2. 创建独立的跟踪器
mem_tracker = tracker.SummaryTracker()

# 3. 执行测试代码
a = []
def func_to_test():
    for i in range(10):
        a.append([])  # 添加10个空列表

func_to_test()

# 4. 获取测试后状态
post_test_objects = muppy.get_objects()
post_test_lists = muppy.filter(post_test_objects, Type=list)

# 5. 计算真正的差异
print(f"显式添加的列表数量: {len(a)}")
print(f"基线列表数量: {len(baseline_lists)}")
print(f"测试后列表数量: {len(post_test_lists)}")
print(f"实际新增列表数量: {len(post_test_lists) - len(baseline_lists)}")

# 6. 更精确的差异分析
diff = tracker.SummaryTracker()._get_diff(baseline_objects, post_test_objects)
tracker._format_diff(diff, limit=10, sort='sizeinc')