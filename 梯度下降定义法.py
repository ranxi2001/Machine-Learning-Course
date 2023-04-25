# 使用梯度下降法求二元函数的最小值
# f = 2u^2+4uv+5v^2   初始点为（1，1）

# 设计函数
def function_one(u_input, v_input):  # 函数的输入 u,v
    f = 2*u_input**2+4*u_input*v_input+5*v_input**2 # 算出f 的值
    du = 4*u_input+4*v_input  # 算出 一阶u导数的值
    dv = 5*u_input+10*v_input  # 算出 一阶v导数的值
    return f, du, dv  # 返回三个值

def main():
    u = 1  # 初始点
    v = 1  # 初始点
    eta = 0.05  # 设置步长
    error = 0.000001  # 设置误差函数

    f_old, du, dv = function_one(u, v)  # 算出初始的值

    for i in range(100):  # 开始循环 梯度下降 设置循环的次数
        u -= eta * du
        v -= eta * dv
        f_result, du, dv = function_one(u, v)  # 获得第一次计算的初始值

        if abs(f_result - f_old) < error:  # 下降的结果绝对值与误差进行比较 如果在允许范围内则终止
            f_final = f_result
            print("最小值为： %.5f, 循环次数： %d, 误差：%8f" % (f_final, i, abs(f_result - f_old)))
            break
        print("第 %d 次迭代, 函数值为 %f，u为 %f,v为 %f" % (i, f_result,u,v))
        f_old = f_result

if __name__ == '__main__':
    main()