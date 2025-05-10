class IndexMapper:
    """
    高效的字典索引映射类，用于快速查找全局索引对应的键和真实索引
    
    这个类在初始化时预处理数据，构建索引映射，
    之后的查找操作可以在O(log n)时间内完成，其中n是字典中的键数量
    
    输入字典的值直接为列表长度（整数）
    """
    
    def __init__(self, data_dict):
        """
        初始化索引映射器
        
        参数:
        data_dict: 字典，其中值为整数，表示原列表长度
        """
        self.data_dict = data_dict
        self.sorted_keys = sorted(data_dict.keys())
        
        # 预计算每个键对应列表的起始索引和长度
        self.prefix_sums = []
        self.list_lengths = []
        current_sum = 0
        
        for key in self.sorted_keys:
            length = data_dict[key]  # 直接使用值作为长度
            self.prefix_sums.append(current_sum)
            self.list_lengths.append(length)
            current_sum += length
            
        self.total_length = current_sum
    
    def get_key_and_index(self, idx):
        """
        根据全局索引获取对应的键和真实索引
        
        参数:
        idx: 全局索引
        
        返回:
        (key, list_idx): 元素所在的键和在列表中的真实索引
        如果索引超出范围，返回(None, None)
        """
        # 检查索引是否在有效范围内
        if idx < 0 or idx >= self.total_length:
            return None, None
        
        # 二分查找 - O(log n)时间复杂度
        left, right = 0, len(self.sorted_keys) - 1
        
        while left <= right:
            mid = (left + right) // 2
            
            start_idx = self.prefix_sums[mid]
            end_idx = start_idx + self.list_lengths[mid] - 1
            
            if start_idx <= idx <= end_idx:
                key = self.sorted_keys[mid]
                list_idx = idx - start_idx
                return key, list_idx
            elif idx < start_idx:
                right = mid - 1
            else:  # idx > end_idx
                left = mid + 1
        
        # 正常情况下不会到达此处
        return None, None
    
    def get_total_length(self):
        """返回所有列表元素的总数"""
        return self.total_length
    
    def update_data_dict(self, new_data_dict):
        """
        更新数据字典并重新构建索引
        
        参数:
        new_data_dict: 新的数据字典，值为整数表示列表长度
        """
        self.__init__(new_data_dict)


# 示例使用
if __name__ == "__main__":
    # 示例字典 - 值为列表长度
    data_dict = {
        'c': 3,  # 原来是 [10, 20, 30]
        'a': 4,  # 原来是 [1, 2, 3, 4]
        'b': 2   # 原来是 [5, 6]
    }
    
    # 创建映射器实例
    mapper = IndexMapper(data_dict)
    
    # 测试查找
    print(f"总元素数: {mapper.get_total_length()}")
    print("\n查找结果:")
    
    for i in range(9):
        key, list_idx = mapper.get_key_and_index(i)
        if key is not None:
            print(f"全局索引 {i} → 键:'{key}', 列表索引:{list_idx}")
        else:
            print(f"索引 {i} 超出范围")
            
    # 演示更新字典
    print("\n更新字典后:")
    new_data_dict = {
        'x': 2,  # 原来是 [100, 200]
        'y': 3,  # 原来是 [300, 400, 500]
        'z': 1   # 原来是 [600]
    }
    mapper.update_data_dict(new_data_dict)
    
    for i in range(6):
        key, list_idx = mapper.get_key_and_index(i)
        if key is not None:
            print(f"全局索引 {i} → 键:'{key}', 列表索引:{list_idx}")
        else:
            print(f"索引 {i} 超出范围")