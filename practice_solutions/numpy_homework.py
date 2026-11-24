"""12 NumPy practice answers, consolidated from the submitted original programs.

Run: python numpy_homework.py
The mathematical functions remain importable; examples print all calculated results.
"""
import numpy as np

def run_e1():
    # -*- coding: utf-8 -*-

    A = np.array([[0.20, -0.50, 1.10, 0.40],
                  [0.80, 0.30, -0.70, 0.60],
                  [-0.40, 0.90, 0.10, -0.20]])

    # 1) ndim / shape / size
    print('1) ndim =', A.ndim, ', shape =', A.shape, ', size =', A.size)

    # 2) 第 2 行(下标 1) 与最后 2 列
    row_1 = A[1, :]
    last_2 = A[:, -2:]
    print('2) 第2行(下标1) =', np.round(row_1, 6), ', shape =', row_1.shape)
    print('   最后2列     =', np.round(last_2, 6), ', shape =', last_2.shape)

    # 3) 复制为 B, 将 B 的负数改成 0, 保持 A 不变
    B = A.copy()
    B[B < 0] = 0.0
    print('3) B(负数改为0) =', np.round(B, 6))
    print('   A保持不变     =', np.round(A, 6))


def run_e2():
    # -*- coding: utf-8 -*-

    a = np.array([0.80, -0.60, 0.70])
    b = np.array([-0.20, 0.90, 0.40])

    # 1) * 为逐元素乘法(结果仍是数组), @ 为点乘(结果是标量)
    ab = a * b
    dot = a @ b
    print('1) a * b =', np.round(ab, 6), ', shape =', ab.shape)
    print('   a @ b =', np.round(dot, 6), ', shape =', np.shape(dot))  # 标量 shape 为 ()

    # 2) 欧氏长度: 平方 -> 求和 -> 开平方
    len_a = np.sqrt(np.sum(a ** 2))
    len_b = np.sqrt(np.sum(b ** 2))
    print('2) 欧氏长度: ||a|| =', round(float(len_a), 6), ', ||b|| =', round(float(len_b), 6))

    # 3) 用 np.linalg.norm 核对长度
    print('3) norm核对: ||a|| =', round(float(np.linalg.norm(a)), 6),
          ', ||b|| =', round(float(np.linalg.norm(b)), 6),
          ', 与手算差 =', round(float(abs(len_a - np.linalg.norm(a))), 6),
          '/', round(float(abs(len_b - np.linalg.norm(b))), 6))


def angle_deg(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape:
        raise ValueError('a 与 b 必须是同维的一维向量')
    if np.all(a == 0) or np.all(b == 0):
        raise ValueError('输入向量不能为零向量')
    # cos(theta) = (a @ b) / (||a|| * ||b||)
    cos_theta = (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)          # 数值误差下夹取到合法定义域
    return float(np.degrees(np.arccos(cos_theta)))



def run_e3():
    # -*- coding: utf-8 -*-




    a = np.array([1.00, -1.00])
    b = np.array([1.00, 0.00])
    print('夹角 angle_deg(a, b) =', round(angle_deg(a, b), 6), '度')
    # 边界输入: 零向量抛 ValueError
    try:
        angle_deg(np.array([0.0, 0.0]), b)
    except ValueError as e:
        print('边界: 零向量 ->', e)


def run_e4():
    # -*- coding: utf-8 -*-

    p = np.array([-0.50, 1.20, 2.00])
    q = np.array([0.80, -0.40, 1.10])

    # 1) 差向量
    d = p - q
    print('1) 差向量 p - q =', np.round(d, 6))

    # 2) 欧氏距离: 各坐标差的平方和, 再开平方
    euc = np.sqrt(np.sum(d ** 2))
    print('2) 欧氏距离 =', round(float(euc), 6))

    # 3) 曼哈顿距离: 各坐标差的绝对值之和
    man = np.sum(np.abs(d))
    print('3) 曼哈顿距离 =', round(float(man), 6))

    # 简答: p、q 输入 shape (3,), 差向量 d 输出 shape (3,);
    #       欧氏距离为 np.sqrt(np.sum(d ** 2)), 曼哈顿距离为 np.sum(np.abs(d))。


def unit_vector(a):
    a = np.asarray(a, dtype=float)
    if np.all(a == 0):
        raise ValueError('零向量无法单位化')
    return a / np.linalg.norm(a)



def run_e5():
    # -*- coding: utf-8 -*-




    a = np.array([-1.20, 0.40, 0.80])
    u = unit_vector(a)
    print('unit_vector(a) =', np.round(u, 6))
    print('长度核对 allclose(||u||, 1) =', np.allclose(np.linalg.norm(u), 1.0))
    print('原数组 a 未变 =', np.round(a, 6))
    # 边界输入: 零向量抛 ValueError
    try:
        unit_vector(np.array([0.0, 0.0, 0.0]))
    except ValueError as e:
        print('边界: 零向量 ->', e)

    # 简答: 输入 shape (3,), 输出 shape (3,) 的同向单位向量;
    #       除以自身长度得到长度为 1 的向量, 不修改原数组。


def run_e6():
    # -*- coding: utf-8 -*-

    A = np.array([[0.20, -0.50, 1.10],
                  [0.80, 0.40, -0.30]])

    # 1) 按列求和(消掉行轴)
    s0 = A.sum(axis=0)
    print('1) A.sum(axis=0)  =', np.round(s0, 6), ', shape =', s0.shape)

    # 2) 按行求平均(消掉列轴)
    m1 = A.mean(axis=1)
    print('2) A.mean(axis=1) =', np.round(m1, 6), ', shape =', m1.shape)

    # 3) 整个数组的最大值
    mx = A.max()
    print('3) 整个数组最大值 =', round(float(mx), 6))

    # 简答: sum(axis=0) 输出 shape (3,), 第 j 个元素 = 第 j 列两个行元素之和;
    #       mean(axis=1) 输出 shape (2,), 第 i 个元素 = 第 i 行三个列元素的平均;
    #       最大值 1.10 位于第 0 行第 2 列。


def run_e7():
    # -*- coding: utf-8 -*-

    X = np.array([[0.80, -0.60, 0.70],
                  [-0.40, 0.50, 0.20]])
    scale = np.array([1.50, 0.50, -1.00])
    bias = np.array([-0.10, 0.20, 0.30])

    # 1) 不写 Python 循环, 一条数组表达式
    Y = X * scale + bias
    print('Y = X * scale + bias =')
    print(np.round(Y, 6))
    print('shape =', Y.shape)

    # 简答: X 的 shape 为 (2, 3), scale、bias 的 shape 为 (3,);
    #       广播规则: 末尾对齐, (2, 3) 与 (3,) 配对, scale/bias 沿第 0 轴(行)
    #       复制扩展到 2 行, 即两行共享同一组缩放与偏移。


def run_e8():
    # -*- coding: utf-8 -*-

    a = np.array([[[0.20, -0.40], [0.70, 0.10]],
                  [[0.30, 0.80], [-0.20, 0.50]],
                  [[-0.10, 0.60], [0.40, -0.30]]])   # shape (C, H, W) = (3, 2, 2)
    b = a + 0.10

    # 1) np.stack 沿 axis=0 拼接, 新增第 0 轴表示图像编号
    batch = np.stack([a, b], axis=0)
    print('1) batch shape =', batch.shape)           # (2, 3, 2, 2)

    # 2) 提取两张图像的蓝色通道 C=2
    blue = batch[:, 2, :, :]
    print('2) 蓝色通道(C=2) =')
    print(np.round(blue, 6))
    print('   shape =', blue.shape)

    # 3) 转为 (B, H, W, C) 顺序
    hwc = np.transpose(batch, (0, 2, 3, 1))
    print('3) 转 (B, H, W, C) 后 shape =', hwc.shape)


def angles_to(query, X):
    query = np.asarray(query, dtype=float)
    X = np.asarray(X, dtype=float)
    if np.all(query == 0):
        raise ValueError('query 不能为零向量')
    q_norm = np.linalg.norm(query)
    x_norms = np.linalg.norm(X, axis=1)               # 每行长度, shape (n_rows,)
    with np.errstate(divide='ignore', invalid='ignore'):
        cos_theta = (X @ query) / (q_norm * x_norms)  # 矩阵乘法一次算所有行
    cos_theta[x_norms == 0] = np.nan                  # 零向量行 -> NaN
    cos_theta = np.clip(cos_theta, -1.0, 1.0)         # NaN 保持不变
    return np.degrees(np.arccos(cos_theta))



def run_m1():
    # -*- coding: utf-8 -*-




    query = np.array([0.60, -0.80, 0.00])
    X = np.array([[0.30, -0.40, 0.00],
                  [-0.60, 0.80, 0.00],
                  [0.80, 0.60, 0.00],
                  [0.00, 0.00, 0.00]])
    ang = angles_to(query, X)
    print('angles_to(query, X) =', np.round(ang, 6), ', shape =', ang.shape)
    # 边界输入: query 为零向量抛 ValueError
    try:
        angles_to(np.zeros(3), X)
    except ValueError as e:
        print('边界: query 为零向量 ->', e)


def top_k_cosine(query, gallery, k):
    query = np.asarray(query, dtype=float)
    gallery = np.asarray(gallery, dtype=float)
    if np.all(query == 0):
        raise ValueError('query 不能为零向量')
    g_norms = np.linalg.norm(gallery, axis=1)
    valid = g_norms > 0                                # 零向量行不参与排序
    n_valid = int(valid.sum())
    if not isinstance(k, (int, np.integer)) or k <= 0 or k > n_valid:
        raise ValueError('k 必须为正整数且不超过有效行数')

    # 相似度 = 点乘 / 两边长度
    sims = np.zeros(gallery.shape[0])
    sims[valid] = (gallery[valid] @ query) / (np.linalg.norm(query) * g_norms[valid])
    sims[~valid] = -np.inf                             # 零向量行排到最后

    # 降序稳定排序, 同分保持原行顺序
    idx = np.argsort(-sims, kind='stable')[:k]
    return idx, sims[idx]



def run_m2():
    # -*- coding: utf-8 -*-




    query = np.array([0.60, -0.80, 0.00])
    gallery = np.array([[0.60, -0.80, 0.00],
                        [0.30, -0.40, 0.00],
                        [0.80, 0.60, 0.00],
                        [-0.60, 0.80, 0.00],
                        [0.00, 0.00, 0.00]])
    ids = np.array(['img_a', 'img_b', 'img_c', 'img_d', 'img_zero'])
    idx, sims = top_k_cosine(query, gallery, 3)
    print('top_k_cosine(query, gallery, k=3)')
    print('行索引 =', idx.tolist())
    print('相似度 =', np.round(sims, 6).tolist())
    print('选中 ID =', ids[idx].tolist())
    # 边界输入: k 超过有效行数(4)抛 ValueError
    try:
        top_k_cosine(query, gallery, 5)
    except ValueError as e:
        print('边界: k 超过有效行数 ->', e)


def run_m3():
    # -*- coding: utf-8 -*-

    A = np.array([[0.20, -0.40],
                  [0.80, 0.30],
                  [-0.50, 0.70]])
    B = np.array([[0.10, 0.20],
                  [-0.60, -0.20]])

    # 1) 用广播构造差数组
    diff = A[:, None, :] - B[None, :, :]   # shape (3, 2, 2)
    print('1) 差数组 diff shape =', diff.shape)

    # 2) 距离矩阵 D: 对特征轴平方求和再开方
    D = np.sqrt(np.sum(diff ** 2, axis=-1))   # shape (3, 2), D[i,j] = 距离(A[i], B[j])
    print('2) 距离矩阵 D =')
    print(np.round(D, 6))
    print('   D shape =', D.shape)

    # 3) A 的每一行返回距离最近的 B 行索引
    nearest = np.argmin(D, axis=1)
    print('3) A 每行最近的 B 行索引 =', nearest.tolist())


def run_m4():
    # -*- coding: utf-8 -*-

    X = np.array([[0.80, -0.60, 0.70],
                  [-0.40, -0.50, 0.90]])
    W1 = np.array([[-0.10, 0.20, -0.80],
                   [0.30, -0.50, 0.10]])
    b1 = np.array([-0.15, 0.05])
    W2 = np.array([[0.20, -0.60],
                   [-0.80, 0.52]])
    b2 = np.array([0.10, -0.20])

    # 1) 第一层线性: W 按 (输出特征数, 输入特征数) 存储, 转置后乘
    Z = X @ W1.T + b1
    print('1) Z = X @ W1.T + b1 =')
    print(np.round(Z, 6))
    print('   Z shape =', Z.shape)

    # 2) ReLU 逐元素处理: 每个负数替换为 0
    H = np.maximum(Z, 0.0)
    print('2) H = np.maximum(Z, 0.0) =')
    print(np.round(H, 6))
    print('   H shape =', H.shape)

    # 3) 第二层线性
    Y = H @ W2.T + b2
    print('3) Y = H @ W2.T + b2 =')
    print(np.round(Y, 6))
    print('   Y shape =', Y.shape)


def main():

    print("\n===== NP-E1 =====")
    run_e1()

    print("\n===== NP-E2 =====")
    run_e2()

    print("\n===== NP-E3 =====")
    run_e3()

    print("\n===== NP-E4 =====")
    run_e4()

    print("\n===== NP-E5 =====")
    run_e5()

    print("\n===== NP-E6 =====")
    run_e6()

    print("\n===== NP-E7 =====")
    run_e7()

    print("\n===== NP-E8 =====")
    run_e8()

    print("\n===== NP-M1 =====")
    run_m1()

    print("\n===== NP-M2 =====")
    run_m2()

    print("\n===== NP-M3 =====")
    run_m3()

    print("\n===== NP-M4 =====")
    run_m4()


if __name__ == "__main__":
    main()
