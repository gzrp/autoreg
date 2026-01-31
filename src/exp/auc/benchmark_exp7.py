import argparse

import math

from src.profiling.profiling import get_profile_data

class BudgetAwareCoordinatorSH:
    def __init__(self, args, budget: float, exploit_profile_time: float):
        self.budget = budget
        self.eta = args.reduction_factor
        self.t2 = exploit_profile_time
        self.U_init = 1
        self.R = args.max_epochs
        self.num_workers = args.num_workers
        self.enable_at_least = 1 * self.R * self.U_init * self.t2

    def schedule(self):
        if self.budget < 1:
            raise Exception("budget must be larger than 1s")

        if self.budget < self.enable_at_least:
            C = 0
            T2_real = 0
            return C, self.budget, T2_real
        else:
            k = int(math.log(self.R / self.U_init, self.eta))
            C = int((self.budget * self.num_workers) / (self.U_init * self.t2 * (k+1)) )
            T2_real = C * self.U_init * self.t2 * (k+1) / self.num_workers
            return C, self.budget, T2_real

class BudgetAwareCoordinatorUniform:
    def __init__(self, args, budget: float, exploit_profile_time: float):
        self.budget = budget
        self.t2 = exploit_profile_time
        self.U_init = 1
        self.R = args.max_epochs
        self.num_workers = args.num_workers
        self.enable_at_least = 1 * self.R * self.U_init * self.t2

    def schedule(self):
        if self.budget < 1:
            raise Exception("budget must be larger than 1s")

        if self.budget < self.enable_at_least:
            C = 0
            T2_real = 0
            return C, self.budget, T2_real
        else:
            C = int((self.budget * self.num_workers) / (self.U_init * self.t2 * self.R) )
            T2_real = C * self.U_init * self.t2 * self.R / self.num_workers
            return C, self.budget, T2_real

class BudgetAwareCoordinatorSuccReject:
    def __init__(self, args, budget: float, exploit_profile_time: float):
        self.budget = budget
        self.t2 = exploit_profile_time
        self.U_init = 1
        self.R = args.max_epochs
        self.num_workers = args.num_workers
        self.enable_at_least = 1 * self.R * self.U_init * self.t2

    def schedule(self):
        if self.budget < 1:
            raise Exception("budget must be larger than 1s")

        if self.budget < self.enable_at_least:
            C = 0
            T2_real = 0
            return C, self.budget, T2_real
        else:
            C = int((self.budget * self.num_workers) / (self.U_init * self.t2 * self.R) + (self.R-1)/2)
            T2_real = self.U_init * self.t2 * (self.R * C - self.R * (self.R - 1) / 2) / self.num_workers
            return C, self.budget, T2_real

def benchmark_sha():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="frappe")
    parser.add_argument("--reduction_factor", type=int, default=2)
    parser.add_argument("--max_epochs", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    for i in range(1, 37129+1):
        kv = get_profile_data(dataset=args.dataset)
        t2 = kv["t2"]
        sh = BudgetAwareCoordinatorSH(args=args, budget=i, exploit_profile_time=t2)
        C, B_real, T2_real = sh.schedule()
        brack = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 700, 800,
                 900, 1000, 1200, 1400, 1600, 1800, 2000]

        if C<30 or C in brack:
            print(C, B_real, T2_real)

def benchmark_uniform():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="frappe")
    parser.add_argument("--reduction_factor", type=int, default=2)
    parser.add_argument("--max_epochs", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    for i in range(1, 37129*4+1):
        kv = get_profile_data(dataset=args.dataset)
        t2 = kv["t2"]
        sh = BudgetAwareCoordinatorUniform(args=args, budget=i, exploit_profile_time=t2)
        C, B_real, T2_real = sh.schedule()
        brack = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 700, 800,
                 900, 1000, 1200, 1400, 1600, 1800, 2000]

        if C<30 or C in brack:
            print(C, B_real, T2_real)


def benchmark_reject():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="frappe")
    parser.add_argument("--reduction_factor", type=int, default=2)
    parser.add_argument("--max_epochs", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    for i in range(1, 37129*4+1):
        kv = get_profile_data(dataset=args.dataset)
        t2 = kv["t2"]
        sh = BudgetAwareCoordinatorSuccReject(args=args, budget=i, exploit_profile_time=t2)
        C, B_real, T2_real = sh.schedule()
        brack = [10, 20, 30, 40, 50, 60, 70, 80, 90,  100, 150, 200, 250, 300, 350, 400,  450,  500,  550,  600,  700,  800,  900,  1000, 1200, 1400, 1600, 1800, 2000]
        if C<30 or C in brack:
            print(C, B_real, T2_real)



# adult
# num        8  10 20 30 40 50  60  70  80  90  100 150 200 250 300 350 400  450  500  550  600  700  800  900  1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000
# succhalf   20 25 50 75 99 123 146 172 195 220 245 365 486 608 730 851 972 1095 1215 1337 1460 1701 1945  2189 2430 2675 2918 3160 3403 3645 3890 4131 4375 4618 4860

# num        4  10 20  30  40  50  60  70  80  90  100 150 200 250  300  350  400  450  500  550  600  700  800  900  1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000
# uniform    20 50 100 150 195 245 295 345 390 440 490 730 975 1215 1460 1705 1945 2190 2430 2675 2920 3405 3890 4375 4860 5350 5835 6320 6805 7290 7780 8265 8750 9235 9720
#
# num        7  10 20  30  40  50  60  70  80  90  100 150 200 250  300  350  400  450  500  550  600  700  800  900  1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000
# succrejct  20 32 81  129 178 226 275 324 372 421 469 712 955 1198 1441 1684 1927 2170 2413 2656 2899 3385 3871 4357 4843 5329 5815 6301 6787 7273 7759 8245 8731 9217 9703


# ccfraud
# num        12 20  30  40  50  60  70  80  90  100 150 200   250  300  350  400  450  500  550  600  700  800  900  1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000
# succhalf   66 103 154 205 256 307 358 409 460 511 766 1022 1277 1532 1788 2043  2298 2554 2809 3064 3575 4085 4596 5107 5617 6128 6638 7149 7660 8170 8681 9192 9702 10213

# num        4 10  20  30  40  50  60  70   80   90    100  150  200   250  300  350  400  450  500  550  600  700   800   900   1000  1100  1200  1300  1400  1500  1600  1700  1800  1900  2000
# uniform   66 164 327 491 654 817 981 1144 1308 1471  1634 2451 3268 4085  4902 5719 6536 7353 8170 8987 9804 11438 13072 14706 16340 17974 19608 21242 22876 24510 26144 27778 29412 31046 32680

# num          11  20  30  40  50  60  70   80   90    100  150  200   250  300  350  400  450  500  550  600  700   800   900   1000  1100  1200  1300  1400  1500  1600  1700  1800  1900  2000
# succrejt     66 205  368 532 695 858 1022 1185 1349  1512 2329 3146 3963  4780 5597 6414 7231 8048 8865 9682 11316 12950 14584 16218 17852 19486 21119 22753 24387 26021 27655 29289 30923 32557


# diabetic
# num          12  20  30  40  50  60  70   80   90    100  150  200   250  300  350  400  450  500  550  600  700   800   900   1000  1100  1200  1300  1400  1500  1600  1700  1800  1900  2000
# succhalf     59  92 138 183 229 275 320  366  412    457  686  914   1143 1371 1600 1828 2057 2285 2514 2742 3199 3656  4113   4570  5027  5484  5941  6397  6854  7311  7768  8225  8682  9139

# num          4  10  20  30  40  50  60  70   80   90    100  150  200   250  300  350  400  450  500  550  600  700   800   900   1000  1100  1200  1300  1400  1500  1600  1700  1800  1900  2000
# uniform      59 147 293 439 585 732 878 1024 1170 1316  1463 2194 2925 3656 4387  5118 5849 6580 7311 8042 8773 10236 11698 13160 14622 16084 17546 19009 20471 21933 23395 24857 26319 27782 29244

# num          11  20  30  40  50  60  70   80   90    100  150  200   250  300  350  400  450  500  550  600  700   800   900   1000  1100  1200  1300  1400  1500  1600  1700  1800  1900  2000
# succrejct    59 183 329 476 622 768 914  1061 1207   1353 2084 2815 3546  4277 5008 5739 6471 7202 7933 8664 10126 11588 13050 14512 15975 17437 18899 20361 21823 23285 24748 26210 27672 29134

# connect
# num          13  20  30  40  50  60  70   80   90    100  150  200   250  300  350  400  450  500  550  600  700   800   900   1000  1100  1200  1300  1400  1500  1600  1700  1800  1900  2000
# succhalf     53  82  122 163 204 244 285 326  366    407  610  813   1017 1220 1423 1626 1829 2033 2236 2439 2846  3252 3658   4065  4471  4878  5284  5691  6097  6503  6910  7316  7723  8129

# num          4  10  20  30  40  50  60  70   80   90    100  150  200   250  300  350  400  450  500  550  600  700   800   900   1000  1100  1200  1300  1400  1500  1600  1700  1800  1900  2000
# uniform      53 131 261 391 521 651 781 911 1041 1171  1301  1951 2602  3252 3902 4553 5203 5853 6503 7154 7804 9105  10405 11706 13006 14307 15608 16908 18209 19509 20810 22111 23411 24712 26012

# num             11  20  30  40  50  60  70   80   90    100  150  200   250  300  350  400  450  500  550  600  700   800   900   1000  1100  1200  1300  1400  1500  1600  1700  1800  1900  2000
# succrejct       53 163 293 423 553 683 813  943  1073   1204 1854 2504 3154  3805 4455 5105 5756 6406 7056 7707 9007  10308 11608 12909 14210 15510 16811 18111 19412 20712 22013 23314 24614 25915


# bank
# succhalf
# 12  20  30  40  50  60  70   80   90    100  150  200   250  300  350  400  450  500  550  600  700   800   900   1000  1200  1400  1600  1800  2000
# 32  50  75  99  124 149 173 198   223   248  371  495   618  742  865  989  1113 1236 1360 1483 1730  1978  2225  2472  2966  3460  3955  4449  4943
# uniform
# 4  10  20  30  40  50  60  70   80   90    100  150  200   250  300  350  400  450  500  550  600  700   800   900   1000  1200  1400  1600  1800  2000
# 32 80  159 238 317 396 475 554 633  712    791  1187 582   1978 2373 2768 3164 3559 3955 4350 4746 5536  6327  7118  7909  9491  11072 12654 14236 15817
# succreject
# 11  20  30  40  50  60  70   80   90    100  150  200   250  300  350  400  450  500  550  600  700   800   900   1000  1200  1400  1600  1800  2000
# 32  99  178 258 337 416 495  574  653   732  1127 1523  1918 2314 2709 3105 3500 3895 4291 4686 5477  6268  7059  7850  9431  11013 12595 14176 15758

# clickpred
#succhalf
# 21  30  40  50  60  70   80   90    100  150  200   250  300  350  400  450  500  550  600  700   800   900   1000  1200  1400  1600  1800  2000
# 244 343 354 572 686 801  915  1029 1143  1715 2286  2858 3429 4001 4572 5143 5715 6286 6858 8001 9143  10286  11429 13715 16001 18286 20572 22858

# uniform
# 4   10  20   30   40   50   60   70   80    90    100  150  200   250   300   350   400   450   500   550   600   700   800   900   1000  1200  1400  1600  1800   2000
# 244 610 1220 1829 2439 3048 3658 4267 4877 5486   6096 9143 12191 15239 18286 21334 24382 27429 30477 33525 36572 42667 48763 54858 60953 73144 85334 97525 109716 121906

# succreject
# 19  20   30   40   50   60   70   80    90    100   150  200   250   300   350   400   450   500   550   600   700   800   900   1000  1200  1400  1600  1800    2000
# 244 275 884  1494  2103 2713 3322 3932 4541   5151  8199 11246 14294 17342 20389 23437 26484 29532 32580 35627 41723 47818 53913 60009 72199 84390 96580 108771  120961

# frappe
# succhalf
# 12  20   30   40   50   60   70   80    90    100   150  200   250   300   350   400   450   500   550   600   700   800   900   1000  1200  1400  1600  1800    2000
# 237 371 556   741 926  1111 1296 1481 1666  1851   2776 3701  4626  5552  6477  7402  8327  9252 10177  11103 12953 14803 16654 18504 22205 25905 29606 33307   37007

#uniform
# 4  10  20   30   40   50   60   70   80    90    100   150  200   250   300   350   400   450   500   550   600   700   800   900   1000  1200  1400  1600  1800    2000
#237 593 1185 1777 2369 2961 3553 4145 4737 5329  5922  8882 11843 14803 17764 20724 23685 26645 29606 32567 35527 41448 47369 53290 59211 71054 82896 94738 106580 118422

#succreject
# 11  20   30   40   50   60   70   80    90    100   150  200   250   300   350   400   450   500   550   600   700   800   900   1000  1200  1400  1600  1800    2000
# 237 741 1333 1925 2517 3109 3701 4293 4885  5478  8438  11399 14359 17320 20280 23241 26201 29162 32122 35083 41004 46925 52846 58767 70610 82452 94294 106136  117978

if __name__ == '__main__':
    # benchmark_sha()
    # benchmark_uniform()
    benchmark_reject()