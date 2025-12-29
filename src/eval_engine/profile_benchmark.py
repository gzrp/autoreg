import argparse

from src.eval_engine.reg_selection_time import BudgetAwareCoordinatorSH
from src.profiling.profiling import get_profile_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="connect")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_cpus", type=int, default=10)
    parser.add_argument("--num_gpus", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_epochs", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=2000)
    parser.add_argument("--trail_num_cpus", type=int, default=2)
    parser.add_argument("--trail_num_gpus", type=float, default=0.5)
    parser.add_argument("--trail_metric", type=str, default="auc")
    parser.add_argument("--trail_mode", type=str, default="max")
    parser.add_argument("--exp_name", type=str, default="2phase")
    parser.add_argument("--storage", type=str, default="~/ray_results")
    parser.add_argument("--reduction_factor", type=int, default=2)
    parser.add_argument("--population_size", type=int, default=10)
    parser.add_argument("--sample_size", type=int, default=3)
    parser.add_argument("--k_n", type=float, default=0.5)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--verbose", type=bool, default=False)
    parser.add_argument("--sample_ratio", type=float, default=0.2)
    parser.add_argument("--swa_start_epoch", type=int, default=2)
    parser.add_argument("--budget", type=int, default=21)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    # T1 * p = t1 * 2000  956.2605721950531   1639.0088067054749

    # 13.752 * 4 / 50 = 1.100160
    # 2888.335 * 4 / 10000 = 1.155334


    # res = 4401.643786907196  * 4 / 10000
    # print(res)
    # # T2 * p = t2 * C * (log_eta (R/U_init) + 1)
    # res2 = 23338.780160427094 * 4 / (2000 * 6)
    # print(res2)
    #
    #  0.05 4380 219  8760 438  13160 658   17540 877
    #  0.10 3380 338  6760 676  10150 1015  13530 1353
    #  0.15 2753 413  5506 826  8260 1239   11020 1653
    #  0.20 2320 464  4645 929  6965 1393   9290 1858
    #  0.25 2008 502  4016 1004 6024 1506   8032 2008
    #  0.30 1766 530  3536 1061 5303 1591   7073 2122
    #  0.35 1577 552  3157 1105 4737 1658   6317 2211
    #  0.40 1427 571  2855 1142 4282 1713   5710 2284
    #  0.45 1300 585  2602 1171 3904 1757   5206 2343
    #  0.50 1196 598  2392 1196 3590 1795   4786 2393

    for i in [1800,3600,5400,7200, 11912]:
    # for i in range(1,   7749+1):
        kv = get_profile_data(dataset="connect")
        t1 = kv["t1"]
        t2 = kv["t2"]
        sh = BudgetAwareCoordinatorSH(args=args, budget=i, explore_profile_time=t1, exploit_profile_time=t2, only_one_phase=False)
        N, C, B_real, T_real, T1_real, T2_real = sh.schedule()
        print(N, C, B_real, T_real, T1_real, T2_real)

    # 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 18 20 22 24 26 28 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100
    # 110 120 130 140 150 160 170 180 190 200 220 240 260 280 300 320 340 360 380 400 420 440 460 480
    # 500 550 600 650 700 750 800 850 900 950 1000 1050 1100 1150 1200 1250 1300 1350 1400 1450 1500 1550 1600 1627