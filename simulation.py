
import matplotlib.pyplot as plt

from tqdm import tqdm

from platoon import Platoon

from helpers import SimType, count_dangers

class Simulation(object):
    def __init__(self, sim_type):
        self.type = sim_type
        self.duration = 73000
        self.lqgs = []
        self.dangers = []
        self.full_txs = []
        self.num_runs = 5
        self.N = 4
        self.tmp_list = None
        self.full_tx_timestamps = []

    def run_4_normal_nonid(self):
        w = 10000
        for i in range(self.num_runs):
          #  print(i)
          #  np.random.seed(i)
            if self.type == SimType.CENTRALIZED:
                mod = Platoon(self.N, [4, 2, 2, 2], w_vel=1, w_dist=w, w_small=0.1, sampling_period=0.01, sim_type=SimType.CENTRALIZED)
            elif self.type == SimType.IDEAL:
                mod = Platoon(self.N, [4], w_vel=1, w_dist=w, w_small=0.1, sampling_period=0.01, sim_type=SimType.IDEAL)
            elif self.type == SimType.DISTRIBUTED:
                mod = Platoon(self.N, [1, 2, 2, 2], w_vel=1, w_dist=w, w_small=0.1, sampling_period=0.01, sim_type=SimType.DISTRIBUTED)
            else:
                mod = Platoon(self.N, [1, 2, 2, 2], w_vel=1, w_dist=w, w_small=0.1, sampling_period=0.01, sim_type=self.type)
            for _ in tqdm(range(self.duration)):
                mod.step()
            dangers = 0
            distances = []
            velocities = []
            inputs2 = []
            inputs3 = []
            inputs4 = []
            for t in range(self.N - 1):
                dangers += count_dangers(mod.states[2 * t + 1], -5)
                for el in mod.states[2 * t + 1]:
                    distances.append(el)
                for el in mod.states[2 * t]:
                    velocities.append(el)
            self.dangers.append(dangers)
            self.lqgs.append(mod.lqg / self.duration)
            self.full_txs.append(mod.num_full_tx)
            for el in mod.u_list:
                inputs2.append(el[1])
                inputs3.append(el[2])
                inputs4.append(el[3])
            exp_name = '../set1_tr/'
            exp_end = str(self.type) +  '_' + str(i) + '.txt'

            with open(exp_name + 'dists_' + exp_end, 'w') as fp:
                for item in distances:
                    # write each item on a new line
                    fp.write("%s\n" % item)
            with open(exp_name + 'vel_' + exp_end, 'w') as fp:
                for item in velocities:
                    # write each item on a new line
                    fp.write("%s\n" % item)
            with open(exp_name + 'in2_' + exp_end, 'w') as fp:
                for item in inputs2:
                    # write each item on a new line
                    fp.write("%s\n" % item)
            with open(exp_name + 'in3_' + exp_end, 'w') as fp:
                for item in inputs3:
                    # write each item on a new line
                    fp.write("%s\n" % item)
            with open(exp_name + 'in4_' + exp_end, 'w') as fp:
                for item in inputs4:
                    # write each item on a new line
                    fp.write("%s\n" % item)


    def run_4_normal_id(self, period=1):
        lam_list = [10, 20, 30, 50, 100, 200, 300, 500]
        for lam in lam_list:
            self.full_txs = []
            for i in range(self.num_runs):
              #  print(i)
               # np.random.seed(i)
                mod = Platoon(self.N, [1, 2, 2, 2], w_vel=1, w_dist=10000, w_small=0.1, sampling_period=0.01, sim_type=self.type, id_period=period)
                mod.threshold = lam
                for _ in tqdm(range(self.duration)):
                    mod.step()
                dangers = 0
                distances = []
                velocities = []
                inputs2 = []
                inputs3 = []
                inputs4 = []
                for t in range(self.N - 1):
                    dangers += count_dangers(mod.states[2 * t + 1], -5)
                    for el in mod.states[2 * t + 1]:
                        distances.append(el)
                    for el in mod.states[2 * t]:
                        velocities.append(el)
                for el in mod.u_list:
                    inputs2.append(el[1])
                    inputs3.append(el[2])
                    inputs4.append(el[3])
                self.dangers.append(dangers)
                self.lqgs.append(mod.lqg / self.duration)
                self.full_txs.append(mod.num_full_tx)
                self.full_tx_timestamps.append(mod.full_tx)
                exp_name = '../set1_tr/'
                exp_end =  str(self.type) + '_' + str(period) + '_' + str(lam) +  '_' + str(i) + '.txt'
                with open(exp_name + 'dists_' + exp_end, 'w') as fp:
                    for item in distances:
                        # write each item on a new line
                        fp.write("%s\n" % item)
                with open(exp_name + 'vel_' + exp_end, 'w') as fp:
                    for item in velocities:
                        # write each item on a new line
                        fp.write("%s\n" % item)
                with open(exp_name + 'in2_' + exp_end, 'w') as fp:
                    for item in inputs2:
                        # write each item on a new line
                        fp.write("%s\n" % item)
                with open(exp_name + 'in3_' + exp_end, 'w') as fp:
                    for item in inputs3:
                        # write each item on a new line
                        fp.write("%s\n" % item)
                with open(exp_name + 'in4_' + exp_end, 'w') as fp:
                    for item in inputs4:
                        # write each item on a new line
                        fp.write("%s\n" % item)
            with open(exp_name + 'full_tx_' + str(self.type) + '_' + str(period) + '_' + str(lam) + '.txt', 'w') as fp:
                for item in self.full_txs:
                    # write each item on a new line
                    fp.write("%s\n" % item)

    def run_7_normal_nonid(self):
        w = 10000
        for i in range(self.num_runs):
            if self.type == SimType.CENTRALIZED:
                mod = Platoon(self.N, [7, 2, 2, 2, 2, 2, 2], w_vel=1, w_dist=w, w_small=0.1, sampling_period=0.01, sim_type=SimType.CENTRALIZED)
            elif self.type == SimType.IDEAL:
                mod = Platoon(self.N, [7], w_vel=1, w_dist=w, w_small=0.1, sampling_period=0.01, sim_type=SimType.IDEAL)
            elif self.type == SimType.DISTRIBUTED:
                mod = Platoon(self.N, [1, 2, 2, 2, 2, 2, 2], w_vel=1, w_dist=w, w_small=0.1, sampling_period=0.01, sim_type=SimType.DISTRIBUTED)

            for _ in tqdm(range(self.duration)):
                mod.step()
            distances = []
            velocities = []
            inputs2 = []
            inputs3 = []
            inputs4 = []
            inputs5 = []
            inputs6 = []
            inputs7 = []
            for t in range(self.N - 1):
                for el in mod.states[2 * t + 1]:
                    distances.append(el)
                for el in mod.states[2 * t]:
                    velocities.append(el)
            self.lqgs.append(mod.lqg / self.duration)
            self.full_txs.append(mod.num_full_tx)
            for el in mod.u_list:
                inputs2.append(el[1])
                inputs3.append(el[2])
                inputs4.append(el[3])
                inputs5.append(el[4])
                inputs6.append(el[5])
                inputs7.append(el[6])
            exp_name = '../set2_more_traffic/'
            exp_end = str(self.type) +  '_' + str(i) + '.txt'
            with open(exp_name + 'dists_' + exp_end, 'w') as fp:
                for item in distances:
                    # write each item on a new line
                    fp.write("%s\n" % item)
            with open(exp_name + 'vel_' + exp_end, 'w') as fp:
                for item in velocities:
                    # write each item on a new line
                    fp.write("%s\n" % item)
            with open(exp_name + 'in2_' + exp_end, 'w') as fp:
                for item in inputs2:
                    # write each item on a new line
                    fp.write("%s\n" % item)
            with open(exp_name + 'in3_' + exp_end, 'w') as fp:
                for item in inputs3:
                    # write each item on a new line
                    fp.write("%s\n" % item)
            with open(exp_name + 'in4_' + exp_end, 'w') as fp:
                for item in inputs4:
                    # write each item on a new line
                    fp.write("%s\n" % item)
            with open(exp_name + 'in5_' + exp_end, 'w') as fp:
                for item in inputs5:
                    # write each item on a new line
                    fp.write("%s\n" % item)
            with open(exp_name + 'in6_' + exp_end, 'w') as fp:
                for item in inputs6:
                    # write each item on a new line
                    fp.write("%s\n" % item)
            with open(exp_name + 'in7_' + exp_end, 'w') as fp:
                for item in inputs7:
                    # write each item on a new line
                    fp.write("%s\n" % item)


    def run_7_normal_id(self, period=1):
        lam_list = [1]#[10, 20, 30, 50, 100, 200, 300, 500]
        for lam in lam_list:
            self.full_txs = []
            for i in range(self.num_runs):
              #  print(i)
               # np.random.seed(i)
                mod = Platoon(self.N, [1, 2, 2, 2, 2, 2, 2], w_vel=1, w_dist=10000, w_small=0.1, sampling_period=0.01, sim_type=self.type, id_period=period)
                mod.threshold = lam
                for _ in tqdm(range(self.duration)):
                    mod.step()
                dangers = 0
                distances = []
                velocities = []
                inputs2 = []
                inputs3 = []
                inputs4 = []
                inputs5 = []
                inputs6 = []
                inputs7 = []
                for t in range(self.N - 1):
                    dangers += count_dangers(mod.states[2 * t + 1], -5)
                    for el in mod.states[2 * t + 1]:
                        distances.append(el)
                    for el in mod.states[2 * t]:
                        velocities.append(el)
                for el in mod.u_list:
                    inputs2.append(el[1])
                    inputs3.append(el[2])
                    inputs4.append(el[3])
                    inputs5.append(el[4])
                    inputs6.append(el[5])
                    inputs7.append(el[6])
                self.dangers.append(dangers)
                self.lqgs.append(mod.lqg / self.duration)
                self.full_txs.append(mod.num_full_tx)
                self.full_tx_timestamps.append(mod.full_tx)
                exp_name = '../set2_more_traffic/'
                exp_end =  str(self.type) + '_' + str(period) + '_' + str(lam) +  '_' + str(i) + '.txt'
                with open(exp_name + 'dists_' + exp_end, 'w') as fp:
                    for item in distances:
                        # write each item on a new line
                        fp.write("%s\n" % item)
                with open(exp_name + 'vel_' + exp_end, 'w') as fp:
                    for item in velocities:
                        # write each item on a new line
                        fp.write("%s\n" % item)
                with open(exp_name + 'in2_' + exp_end, 'w') as fp:
                    for item in inputs2:
                        # write each item on a new line
                        fp.write("%s\n" % item)
                with open(exp_name + 'in3_' + exp_end, 'w') as fp:
                    for item in inputs3:
                        # write each item on a new line
                        fp.write("%s\n" % item)
                with open(exp_name + 'in4_' + exp_end, 'w') as fp:
                    for item in inputs4:
                        # write each item on a new line
                            fp.write("%s\n" % item)
                with open(exp_name + 'in5_' + exp_end, 'w') as fp:
                    for item in inputs5:
                        # write each item on a new line
                        fp.write("%s\n" % item)
                with open(exp_name + 'in6_' + exp_end, 'w') as fp:
                    for item in inputs6:
                        # write each item on a new line
                        fp.write("%s\n" % item)
                with open(exp_name + 'in7_' + exp_end, 'w') as fp:
                    for item in inputs7:
                        # write each item on a new line
                        fp.write("%s\n" % item)
            with open(exp_name + 'full_tx_' + str(self.type) + '_' + str(period) + '_' + str(lam) + '.txt', 'w') as fp:
                for item in self.full_txs:
                    # write each item on a new line
                    fp.write("%s\n" % item)
    
    def run_4_braking_nonid(self):
        w = 10000
        for i in range(self.num_runs):
            if self.type == SimType.CENTRALIZED:
                mod = Platoon(self.N, [4, 2, 2, 2], w_vel=1, w_dist=w, w_small=0.1, sampling_period=0.01, sim_type=SimType.CENTRALIZED)
            elif self.type == SimType.IDEAL:
                mod = Platoon(self.N, [4], w_vel=1, w_dist=w, w_small=0.1, sampling_period=0.01, sim_type=SimType.IDEAL)
            elif self.type == SimType.DISTRIBUTED:
                mod = Platoon(self.N, [1, 2, 2, 2], w_vel=1, w_dist=w, w_small=0.1, sampling_period=0.01, sim_type=SimType.DISTRIBUTED)
            x_traj = []
            with open(r'speeds_brake.txt', 'r') as fp:
                for line in fp:
                    x = line[:-1]
                    x_traj.append(float(x))
            mod.leading_platoon.x_traj = x_traj
            mod.x[0] = 30
            mod.x[2] = 30
            mod.x[4] = 30
            mod.x[6] = 30
            for _ in tqdm(range(self.duration)):
                mod.step()
            dangers = 0
            distances = []
            velocities = []
            inputs2 = []
            inputs3 = []
            inputs4 = []
            for t in range(self.N - 1):
                dangers += count_dangers(mod.states[2 * t + 1], -5)
                for el in mod.states[2 * t + 1]:
                    distances.append(el)
                for el in mod.states[2 * t]:
                    velocities.append(el)
            self.dangers.append(dangers)
            self.lqgs.append(mod.lqg / self.duration)
            self.full_txs.append(mod.num_full_tx)
            for el in mod.u_list:
                inputs2.append(el[1])
                inputs3.append(el[2])
                inputs4.append(el[3])
            exp_name = '../set3_plus_traffic/'
            exp_end = str(self.type) +  '_' + str(i) + '.txt'
            with open(exp_name + 'dists_' + exp_end, 'w') as fp:
                for item in distances:
                    # write each item on a new line
                    fp.write("%s\n" % item)
            with open(exp_name + 'vel_' + exp_end, 'w') as fp:
                for item in velocities:
                    # write each item on a new line
                    fp.write("%s\n" % item)
            with open(exp_name + 'in2_' + exp_end, 'w') as fp:
                for item in inputs2:
                    # write each item on a new line
                    fp.write("%s\n" % item)
            with open(exp_name + 'in3_' + exp_end, 'w') as fp:
                for item in inputs3:
                    # write each item on a new line
                    fp.write("%s\n" % item)
            with open(exp_name + 'in4_' + exp_end, 'w') as fp:
                for item in inputs4:
                    # write each item on a new line
                    fp.write("%s\n" % item)

    def run_4_braking_id(self, period):
        lam_list = [10, 20, 30, 50, 100, 200, 300, 500]
        for lam in lam_list:
            self.full_txs = []
            for i in range(self.num_runs):
              #  print(i)
               # np.random.seed(i)
                mod = Platoon(self.N, [1, 2, 2, 2], w_vel=1, w_dist=10000, w_small=0.1, sampling_period=0.01, sim_type=self.type, id_period=period)
                mod.threshold = lam
                x_traj = []
                with open(r'speeds_brake.txt', 'r') as fp:
                    for line in fp:
                        # remove linebreak from a current name
                        # linebreak is the last character of each line
                        x = line[:-1]
                        x_traj.append(float(x))
                mod.leading_platoon.x_traj = x_traj
                mod.x[0] = 30
                mod.x[2] = 30
                mod.x[4] = 30
                mod.x[6] = 30
                for _ in tqdm(range(self.duration)):
                    mod.step()
                dangers = 0
                distances = []
                velocities = []
                inputs2 = []
                inputs3 = []
                inputs4 = []
                for t in range(self.N - 1):
                    dangers += count_dangers(mod.states[2 * t + 1], -5)
                    for el in mod.states[2 * t + 1]:
                        distances.append(el)
                    for el in mod.states[2 * t]:
                        velocities.append(el)
                for el in mod.u_list:
                    inputs2.append(el[1])
                    inputs3.append(el[2])
                    inputs4.append(el[3])
                self.dangers.append(dangers)
                self.lqgs.append(mod.lqg / self.duration)
                self.full_txs.append(mod.num_full_tx)
                self.full_tx_timestamps.append(mod.full_tx)
                exp_name = '../set3_plus_traffic/'
                exp_end =  str(self.type) + '_' + str(period) + '_' + str(lam) +  '_' + str(i) + '.txt'
                with open(exp_name + 'dists_' + exp_end, 'w') as fp:
                    for item in distances:
                        # write each item on a new line
                        fp.write("%s\n" % item)
                with open(exp_name + 'vel_' + exp_end, 'w') as fp:
                    for item in velocities:
                        # write each item on a new line
                        fp.write("%s\n" % item)
                with open(exp_name + 'in2_' + exp_end, 'w') as fp:
                    for item in inputs2:
                        # write each item on a new line
                        fp.write("%s\n" % item)
                with open(exp_name + 'in3_' + exp_end, 'w') as fp:
                    for item in inputs3:
                        # write each item on a new line
                        fp.write("%s\n" % item)
                with open(exp_name + 'in4_' + exp_end, 'w') as fp:
                    for item in inputs4:
                        # write each item on a new line
                        fp.write("%s\n" % item)
            with open(exp_name + '/full_tx_' + str(self.type) + '_' + str(period) + '_' + str(lam) + '.txt', 'w') as fp:
                for item in self.full_txs:
                    # write each item on a new line
                    fp.write("%s\n" % item)

    def run_7_braking_nonid(self):
        w = 10000
        for i in range(self.num_runs):
            if self.type == SimType.CENTRALIZED:
                mod = Platoon(self.N, [7, 2, 2, 2, 2, 2, 2], w_vel=1, w_dist=w, w_small=0.1, sampling_period=0.01, sim_type=self.type)
            elif self.type == SimType.IDEAL:
                mod = Platoon(self.N, [7], w_vel=1, w_dist=w, w_small=0.1, sampling_period=0.01, sim_type=self.type)
            elif self.type == SimType.DISTRIBUTED:
                mod = Platoon(self.N, [1, 2, 2, 2, 2, 2, 2], w_vel=1, w_dist=w, w_small=0.1, sampling_period=0.01, sim_type=self.type)
            x_traj = []
            with open(r'speeds_brake.txt', 'r') as fp:
                for line in fp:
                    x = line[:-1]
                    x_traj.append(float(x))
            mod.leading_platoon.x_traj = x_traj
            mod.x[0] = 30
            mod.x[2] = 30
            mod.x[4] = 30
            mod.x[6] = 30
            mod.x[8] = 30
            mod.x[10] = 30
            mod.x[12] = 30

            for _ in tqdm(range(self.duration)):
                mod.step()
            distances = []
            velocities = []
            inputs2 = []
            inputs3 = []
            inputs4 = []
            inputs5 = []
            inputs6 = []
            inputs7 = []
            for t in range(self.N - 1):
                for el in mod.states[2 * t + 1]:
                    distances.append(el)
                for el in mod.states[2 * t]:
                    velocities.append(el)
            self.lqgs.append(mod.lqg / self.duration)
            self.full_txs.append(mod.num_full_tx)
            for el in mod.u_list:
                inputs2.append(el[1])
                inputs3.append(el[2])
                inputs4.append(el[3])
                inputs5.append(el[4])
                inputs6.append(el[5])
                inputs7.append(el[6])
            exp_name = '../set4_plus_traffic/'
            with open(exp_name + 'dists_' + str(self.type) + '_' + str(i) + '.txt', 'w') as fp:
                for item in distances:
                    # write each item on a new line
                    fp.write("%s\n" % item)
            with open(exp_name + 'vel_' + str(self.type) + '_' + str(i) + '.txt', 'w') as fp:
                for item in velocities:
                    # write each item on a new line
                    fp.write("%s\n" % item)
            with open(exp_name + 'in2_' + str(self.type) + '_' +  str(i) + '.txt', 'w') as fp:
                for item in inputs2:
                    # write each item on a new line
                    fp.write("%s\n" % item)
            with open(exp_name + 'in3_' + str(self.type) + '_' +  str(i) + '.txt', 'w') as fp:
                for item in inputs3:
                    # write each item on a new line
                    fp.write("%s\n" % item)
            with open(exp_name + 'in4_' + str(self.type) + '_' +  str(i) + '.txt', 'w') as fp:
                for item in inputs4:
                    # write each item on a new line
                    fp.write("%s\n" % item)
            with open(exp_name + 'in5_' + str(self.type) + '_' +  str(i) + '.txt', 'w') as fp:
                for item in inputs5:
                    # write each item on a new line
                    fp.write("%s\n" % item)
            with open(exp_name + 'in6_' + str(self.type) + '_' +  str(i) + '.txt', 'w') as fp:
                for item in inputs6:
                    # write each item on a new line
                    fp.write("%s\n" % item)
            with open(exp_name + 'in7_' + str(self.type) + '_' +  str(i) + '.txt', 'w') as fp:
                for item in inputs7:
                    # write each item on a new line
                    fp.write("%s\n" % item)

    def run_7_braking_id(self, period=1):
        lam_list = [1, 10, 20, 30, 50, 100, 200, 300, 500]
        for lam in lam_list:
            self.full_txs = []
            for i in range(self.num_runs):
              #  print(i)
               # np.random.seed(i)
                mod = Platoon(self.N, [1, 2, 2, 2, 2, 2, 2], w_vel=1, w_dist=10000, w_small=0.1, sampling_period=0.01, sim_type=self.type, id_period=period)
                mod.threshold = lam
                x_traj = []
                with open(r'speeds_brake.txt', 'r') as fp:
                    for line in fp:
                        # remove linebreak from a current name
                        # linebreak is the last character of each line
                        x = line[:-1]

                        # add current item to the list
                        x_traj.append(float(x))
                mod.leading_platoon.x_traj = x_traj
                mod.x[0] = 30
                mod.x[2] = 30
                mod.x[4] = 30
                mod.x[6] = 30
                mod.x[8] = 30
                mod.x[10] = 30
                mod.x[12] = 30
                for _ in tqdm(range(self.duration)):
                    mod.step()
                dangers = 0
                distances = []
                #plt.plot(mod.states[6])
                velocities = []
                inputs2 = []
                inputs3 = []
                inputs4 = []
                inputs5 = []
                inputs6 = []
                inputs7 = []
                for t in range(self.N - 1):
                    dangers += count_dangers(mod.states[2 * t + 1], -5)
                    for el in mod.states[2 * t + 1]:
                        distances.append(el)
                    for el in mod.states[2 * t]:
                        velocities.append(el)
                for el in mod.u_list:
                    inputs2.append(el[1])
                    inputs3.append(el[2])
                    inputs4.append(el[3])
                    inputs5.append(el[4])
                    inputs6.append(el[5])
                    inputs7.append(el[6])
                self.dangers.append(dangers)
                self.lqgs.append(mod.lqg / self.duration)
                self.full_txs.append(mod.num_full_tx)
                self.full_tx_timestamps.append(mod.full_tx)
                exp_name = '../set4_no_rollback_hash_3/'
                #exp_name = '../set4_plus_traffic/'
                with open(exp_name + 'dists_' + str(self.type) + '_' + str(period) + '_' + str(lam) +  '_' + str(i) + '.txt', 'w') as fp:
                    for item in distances:
                        # write each item on a new line
                        fp.write("%s\n" % item)
                with open(exp_name + 'vel_' + str(self.type) + '_' + str(period) + '_'+ str(lam) +  '_'+ str(i) + '.txt', 'w') as fp:
                    for item in velocities:
                        # write each item on a new line
                        fp.write("%s\n" % item)
                with open(exp_name + 'full_tx_timestamps' + str(self.type) + '_' + str(period) + '_' + str(lam) +  '_'+ str(i) + '.txt', 'w') as fp:
                    for item1, item2 in mod.full_tx:
                        # write each item on a new line
                        fp.write("%s %s\n" % (item1, item2))
                with open(exp_name + 'in2_' + str(self.type)+ '_' + str(period) + '_' + str(lam) +  '_'+ str(i) + '.txt', 'w') as fp:
                    for item in inputs2:
                        # write each item on a new line
                        fp.write("%s\n" % item)
                with open(exp_name + 'in3_' + str(self.type)+ '_' + str(period) + '_' + str(lam) +  '_'+ str(i) + '.txt', 'w') as fp:
                    for item in inputs3:
                        # write each item on a new line
                        fp.write("%s\n" % item)
                with open(exp_name + 'in4_' + str(self.type)+ '_' + str(period) + '_' + str(lam) + '_'+ str(i) + '.txt', 'w') as fp:
                    for item in inputs4:
                        # write each item on a new line
                            fp.write("%s\n" % item)
                with open(exp_name + 'in5_' + str(self.type)+ '_' + str(period) + '_' + str(lam) + '_' + str(i) + '.txt', 'w') as fp:
                    for item in inputs5:
                        # write each item on a new line
                        fp.write("%s\n" % item)
                with open(exp_name + 'in6_' + str(self.type)+ '_' + str(period) + '_' + str(lam) + '_' + str(i) + '.txt', 'w') as fp:
                    for item in inputs6:
                        # write each item on a new line
                        fp.write("%s\n" % item)
                with open(exp_name + 'in7_' + str(self.type)+ '_' + str(period) + '_' + str(lam) + '_' + str(i) + '.txt', 'w') as fp:
                    for item in inputs7:
                        # write each item on a new line
                        fp.write("%s\n" % item)
            with open(exp_name + 'full_tx_' + str(self.type)+ '_' + str(period) +  '_' + str(lam) + '.txt', 'w') as fp:
                for item in self.full_txs:
                    # write each item on a new line
                    fp.write("%s\n" % item)

    

def set1(): # 4 vehicles, normal
    for type in [SimType.IDEAL, SimType.DISTRIBUTED, SimType.CENTRALIZED]:
        sim = Simulation(type)
        sim.run_4_normal_nonid()
    for type in [SimType.ID, SimType.ID_ON_DELTA]:
        for period in [1, 5, 10, 50]:
            sim = Simulation(type)
            sim.run_4_normal_id(period)


def set2():
    for type in [SimType.IDEAL, SimType.DISTRIBUTED, SimType.CENTRALIZED]:
        sim = Simulation(type)
        sim.N = 7
        sim.run_7_normal_nonid()
    for type in [SimType.ID, SimType.ID_ON_DELTA]:
        for period in [1, 5, 10, 50]:
            sim = Simulation(type)
            sim.N = 7
            sim.run_7_normal_id(period)

def set3():
    for type in [SimType.IDEAL, SimType.DISTRIBUTED, SimType.CENTRALIZED]:
        sim = Simulation(type)
        sim.duration = 1788
        sim.run_4_braking_nonid()
    for type in [SimType.ID, SimType.ID_ON_DELTA, SimType.HASH]:
        for period in [1, 5, 10, 50]:
            sim = Simulation(type)
            sim.duration = 1788
            sim.run_4_braking_id(period)

def set4():
    for type in [SimType.IDEAL, SimType.DISTRIBUTED, SimType.CENTRALIZED]:
        sim = Simulation(type)
        sim.duration = 1788
        sim.N = 7
        sim.run_7_braking_nonid()
    for type in [SimType.ID, SimType.ID_ON_DELTA, SimType.HASH]:
        for period in [1, 5, 10, 50]:
            sim = Simulation(type)
            sim.duration = 1788
            sim.N = 7
            sim.run_7_braking_id(period)

if __name__ == '__main__':
    set4()