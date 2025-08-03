import numpy as np

arr = np.load('/workspace/home/luotianwei/cs336/assignment1-basics/data/TinyStoriesV2_GPT4-valid.npy')
print(type(arr))         # 看是 np.ndarray 还是别的
print(arr.shape)         # 看数组形状
print(arr.dtype)         # 看数据类型（可能是 int32、int64、float32、object 等）
print(arr[:5])           # 打印前5个元素（注意：如果是嵌套结构，需要进一步展开）