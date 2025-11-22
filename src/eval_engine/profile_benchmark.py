

if __name__ == '__main__':
    # T1 * p = t1 * 2000
    res = 663.5360858440399 * 4 / 2000
    print(res)
    # T2 * p = t2 * C * (log_eta (R/U_init) + 1)
    res2 = 882.4773073196411 * 4 / 1200
    print(res2)
    arr = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,30,40,50,60,70,80,90,
           100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2650]
    # ------------------------
    # explorePhase = ExplorePhaseParallel(args)
    # print("--------" * 10)
    # num = 500
    # start = time.time()
    # for i in range(500):
    #     t = explorePhase.profiling()
    #     print(t)
    # total = time.time() - start
    # avg_time = total / num
    # print(avg_time)
    # ---------------------
    # exploitPhase = ExploitPhase(args)
    # print("--------" * 10)
    # max_epochs = 100
    # start_time = time.time()
    # total_time = exploitPhase.profiling(max_epochs)
    # avg_time = total_time / max_epochs
    # print("total_time", total_time)
    # print("avg_time", avg_time)
    # ---------------------
    # sh = BudgetAwareCoordinatorSH(args=args, budget=25, explore_profile_time=1.0, exploit_profile_time=6.0)
    #
    # res = sh.schedule()
    # print(res)
    pass