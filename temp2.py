import objgraph
import gc

# 创建一些对象，模拟潜在的内存泄漏
class MyClass:
    def __init__(self, name):
        self.name = name
        self.ref = None

# 创建对象并形成循环引用
obj1 = MyClass("Object 1")
obj2 = MyClass("Object 2")
obj1.ref = obj2
obj2.ref = obj1  # 循环引用

# 显示当前内存中最常见的对象类型
print("Most common object types:")
objgraph.show_most_common_types(limit=5)

# 显示从 obj1 起始的引用关系
print("\nGenerating reference graph for obj1...")
objgraph.show_refs([obj1], filename='refs.png', refcounts=True)

# 显示哪些对象引用了 MyClass 实例
print("\nObjects referencing MyClass instances:")
objgraph.show_backrefs([obj1], filename='backrefs.png', max_depth=3)

# 手动触发垃圾回收
gc.collect()

# 检查 MyClass 对象是否仍存在
print("\nMyClass objects after GC:")
objgraph.show_most_common_types(limit=5)