def run_experiment1():
    print("👉 正在运行：自由场爆炸压力平面云图")
    # TODO: 实验1的具体逻辑

def run_experiment2():
    print("👉 正在运行：其他")
    # TODO: 实验2的具体逻辑

def main():
    print("欢迎使用冲击波仿真程序")

    while True:
        print("\n可选实验编号：")
        print("1. 实验1---自由场无建筑压力云图")
        print("2. 实验2")
        print("0. 退出程序")

        choice = input("请输入实验编号（如 1 或 2，输入 0 退出）：")

        if choice == "1":
            run_experiment1()
        elif choice == "2":
            run_experiment2()
        elif choice == "0":
            print("✅ 程序已退出。")
            break
        else:
            print("❌ 输入无效，请重新输入 0、1 或 2。")

if __name__ == "__main__":
    main()
