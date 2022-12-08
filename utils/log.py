from matplotlib import pyplot as plt
import os
import scipy.signal
import matplotlib
matplotlib.use('Agg')


class Log():
    def __init__(self, log_dir):
        self.save_path = log_dir
        self.f1s_train = []
        self.f1s_test = []
        self.train_accuracy = []
        self.val_accuracy = []
        self.train_lr = []

    def append_num1(self, name, val):
        if not hasattr(self, name):
            setattr(self, name, [])
        list_train = getattr(self, name)
        list_train.append(val)
        with open(os.path.join(self.save_path, name+'.txt'), 'a') as f:
            f.write(str(val)+'\n')
        self.plot_curves_num1(name)

    def plot_curves_num1(self, name):
        train = getattr(self, name)
        iters = range(len(train))
        plt.figure()
        plt.plot(iters, train, 'red',
                 linewidth=2, label=name)
        try:
            if len(train) < 25:
                num = 5
            else:
                num = 15
            plt.plot(iters, scipy.signal.savgol_filter(train, num, 3), 'green', linestyle='--', linewidth=2,
                     label='smooth train '+name)
        except:
            pass
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel(name)
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.save_path, name+".png"))
        plt.cla()
        plt.close("all")
