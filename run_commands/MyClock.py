import time


class MyClock:
    def __init__(self):
        self.past_t = time.time()
        print(f"开始时间：{time.asctime( time.localtime(self.past_t) )}")

    def cal_time(self):
        now_t = time.time()
        seconds = now_t - self.past_t
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        print("运行时间:   %02d:%02d:%02d" % (h, m, s))
        print(f"结束时间：{time.asctime(time.localtime(now_t))}")
        time.sleep(10)
        self.past_t = time.time()