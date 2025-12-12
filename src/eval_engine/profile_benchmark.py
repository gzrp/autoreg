import argparse

from src.eval_engine.reg_selection_time import BudgetAwareCoordinatorSH
from src.profiling.profiling import get_profile_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="bank")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_cpus", type=int, default=10)
    parser.add_argument("--num_gpus", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_epochs", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=2000)
    parser.add_argument("--trail_num_cpus", type=int, default=2)
    parser.add_argument("--trail_num_gpus", type=float, default=0.5)
    parser.add_argument("--trail_metric", type=str, default="bacc")
    parser.add_argument("--trail_mode", type=str, default="max")
    parser.add_argument("--exp_name", type=str, default="2phase")
    parser.add_argument("--storage", type=str, default="~/ray_results")
    parser.add_argument("--reduction_factor", type=int, default=2)
    parser.add_argument("--population_size", type=int, default=10)
    parser.add_argument("--sample_size", type=int, default=3)
    parser.add_argument("--k_n", type=float, default=0.2)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--verbose", type=bool, default=False)
    parser.add_argument("--sample_ratio", type=float, default=0.2)
    parser.add_argument("--swa_start_epoch", type=int, default=2)
    parser.add_argument("--budget", type=int, default=28)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    # T1 * p = t1 * 2000  956.2605721950531   1639.0088067054749
    res = 561.2363188266754 * 4 / 2000
    print(res)
    # T2 * p = t2 * C * (log_eta (R/U_init) + 1)
    res2 = 1065.745376110077 * 4 / (400 * 3)
    print(res2)

    # total_budget = args.budget
    for i in range(1,  1627+1):
        kv = get_profile_data(dataset= args.dataset)
        t1 = kv["t1"]
        t2 = kv["t2"]
        sh = BudgetAwareCoordinatorSH(args=args, budget=i, explore_profile_time=t1, exploit_profile_time=t2, only_one_phase=False)
        N, C, B_real, T_real, T1_real, T2_real = sh.schedule()
        print(N, C, B_real, T_real, T1_real, T2_real)

    # 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 18 20 22 24 26 28 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100
    # 110 120 130 140 150 160 170 180 190 200 220 240 260 280 300 320 340 360 380 400 420 440 460 480
    # 500 550 600 650 700 750 800 850 900 950 1000 1050 1100 1150 1200 1250 1300 1350 1400 1450 1500 1550 1600 1627