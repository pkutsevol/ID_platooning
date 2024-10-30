import numpy as np
import scipy
import scipy.linalg

from helpers import generate_Q, generate_R, generate_A_full, generate_B_full, generate_A_following, generate_B_following, SimType
from network import Network, ErrorModel


class PlatoonPartial:
    def __init__(self, N, theta=-3.6e-3, delta=1.48e-5, sampling_period=1, k_e=0.148e-3, w_vel=100, w_dist=100000,
                 w_small=0.001, tau=1, leading=False, L_prev=None, sim_type=SimType.CENTRALIZED):
        if leading:
            self.N = N
            self.A = generate_A_full(N, theta, delta, sampling_period)
            self.B = generate_B_full(N, k_e)
            self.Q = generate_Q(N, w_vel, w_dist, tau)
            self.R = generate_R(N, w_small)
            X = np.array(scipy.linalg.solve_discrete_are(self.A, self.B, self.Q, self.R))
            self.K = np.array(np.dot(scipy.linalg.inv(np.dot(np.dot(self.B.T,X),self.B) + self.R),
                                     (np.dot(self.B.T, np.dot(X, self.A)))))
            self.x_traj = []
            with open(r'speeds.txt', 'r') as fp:
                for line in fp:
                    # remsove linebreak from a current name
                    # linebreak is the last character of each line
                    x = line[:-1]

                    # add current item to the list
                    self.x_traj.append(float(x))
        else:
            self.N = N
            self.A = generate_A_following(N, theta, delta, sampling_period, L_prev, k_e)
            self.B = generate_B_following(N, k_e)
            self.Q = generate_Q(N, w_vel, w_dist, tau)
            self.R = generate_R(N-1, w_small)
            X = np.array(scipy.linalg.solve_discrete_are(self.A, self.B, self.Q, self.R))
            self.K = np.array(np.dot(scipy.linalg.inv(np.dot(np.dot(self.B.T,X),self.B) + self.R),
                                     (np.dot(self.B.T, np.dot(X, self.A)))))
        self.k = 0
        self.leading = leading
        if sim_type == SimType.CENTRALIZED or sim_type == SimType.IDEAL:
            self.u = np.zeros(N)
            self.u_next = np.zeros(N)
        if sim_type == SimType.DISTRIBUTED or sim_type == SimType.ID or sim_type == SimType.ID_ON_DELTA or sim_type == SimType.HASH:
            if leading:
                self.u = np.zeros(N)
                self.u_next = np.zeros(N)
            else:
                self.u = np.zeros(N - 1)
                self.u_next = np.zeros(N - 1)
        if sim_type == SimType.CENTRALIZED or sim_type == SimType.DISTRIBUTED or sim_type == SimType.ID or sim_type == sim_type.ID_ON_DELTA or sim_type == SimType.HASH:
            self.scheduling_param = 1
        if sim_type == SimType.IDEAL:
            self.scheduling_param = N
        self.sim_type = sim_type
        self.link = ErrorModel()


    def step(self, x):
        x_est = np.dot(self.A, x) + np.dot(self.B, -np.dot(self.K, x))
        self.u_next = -np.dot(self.K, x_est)
        self.k += 1
        self.link.transit()


    def get_controls(self):
        self.u = self.schedule(self.u_next, self.u).copy()
        return np.array(self.u)

    def opt_control(self, x):
        return -np.dot(self.K, x)

    def schedule(self, u_ideal, u_prev):
        if False and self.sim_type == SimType.CENTRALIZED:
            scheduled = np.arange(self.N)
            for inp in scheduled:
                p = np.random.random()
                if p < 0.1:#225:
                    u_prev[inp] = u_ideal[inp].copy()
            return u_prev
        return u_ideal


class Platoon:
    def __init__(self, N, N_partition, theta=-3.6e-3, delta=1.48e-5, sampling_period=1, k_e=0.148e-3, w_vel=100,
                 w_dist=100000, w_small=0.001, tau=1,  sim_type=SimType.CENTRALIZED, id_period = 1):
        self.leading_platoon = PlatoonPartial(N_partition[0], theta, delta,
                                              sampling_period, k_e, w_vel, w_dist,
                                              w_small, tau, leading=True,  sim_type=sim_type)
        L_prev = self.leading_platoon.K[-1]
        if len(L_prev) > 1:
            L_prev = 0
        self.following_platoons = []
        for n in N_partition[1:]:
            self.following_platoons.append(PlatoonPartial(n, theta, delta,
                                                          sampling_period, k_e, w_vel, w_dist,
                                                          w_small, tau, L_prev=L_prev, sim_type=sim_type))
            L_prev = self.following_platoons[-1].K[0][-1]


        self.N = N
        self.N_partition = N_partition
        self.x = np.zeros(2 * N - 1)

        self.k = 0

        self.A = generate_A_full(N, theta, delta, sampling_period)
        self.B = generate_B_full(N, k_e)
        self.u = np.zeros(N)
        self.Q = generate_Q(N, w_vel, w_dist, tau)
        self.R = generate_R(N, w_small)
        self.lqg = 0
        self.num_full_tx = 0
        self.states = [ [] for _ in range(2 * N - 1) ]
        self.opt_control = np.zeros(N)
        self.ref = PlatoonPartial(N, w_vel=w_vel, w_dist=w_dist, w_small=w_small, sampling_period=sampling_period, leading=True)#3, w_vel=1, w_dist=100, w_small=1, sampling_period=0.01
        if sim_type == SimType.CENTRALIZED:
            self.threshold = 1e10
        if sim_type == SimType.IDEAL:
            self.threshold = 1e10
        if sim_type == SimType.DISTRIBUTED:
            self.threshold = 1e10
        if sim_type == SimType.ID or sim_type == SimType.ID_ON_DELTA or sim_type == SimType.HASH:
            self.threshold = 50
            self.full_tx = []
            self.timers_to_sync = np.zeros(N)
            self.sync_probs = np.zeros(N)
            self.sync_probs[0] = (1 - self.leading_platoon.link.errorrate_control) * 255 / 256
            for i in range(1, N):
                self.sync_probs[i] = (1 - self.following_platoons[i - 1].link.errorrate_control) * 255 / 256
            if sim_type == sim_type.HASH:
                self.threshold_to_sync = 3
            else:
                self.threshold_to_sync = 0
            self.flags_in_desync = np.zeros(N)
            self.last_sent_cntrl = np.zeros(N)
            self.id_period_max = id_period
            self.id_period = [id_period for _ in range(N)]


        self.sim_type = sim_type

        self.network = Network(self.N, self)

        self.u_list = []
        self.x_list = []

    def upd_sync_prob(self):
        self.sync_probs[0] = (1 - self.leading_platoon.link.errorrate_control) * 255 / 256
        for i in range(1, self.N):
            self.sync_probs[i] = (1 - self.following_platoons[i - 1].link.errorrate_control) * 255 / 256

    def step(self):

        self.u[:self.N_partition[0]] = self.leading_platoon.get_controls().copy()

        i_first = self.N_partition[0]
        if self.sim_type != SimType.CENTRALIZED:
            for i in range(len(self.following_platoons)):
                self.u[i_first:i_first + 1] = self.following_platoons[i].get_controls().copy()
                i_first += 1
        if self.sim_type == SimType.CENTRALIZED:
            trucks_states = self.u.copy()
            self.u = self.network.step(trucks_states).copy()

        if self.sim_type == SimType.IDEAL:
            self.u = self.leading_platoon.opt_control(self.x).copy()

        if self.sim_type == SimType.ID or self.sim_type == SimType.ID_ON_DELTA or self.sim_type == SimType.HASH:
            trucks_states = self.get_centralized_control_h(self.threshold).copy()
            u_retrieved = self.network.step(trucks_states).copy()
            for i in range(len(u_retrieved)):
                if u_retrieved[i] is not None:
                    self.u[i] = u_retrieved[i]

        self.leading_platoon.step(self.x[:2 * self.N_partition[0] - 1])
        i_first = self.N_partition[0]
        if self.sim_type != SimType.CENTRALIZED:
            for i in range(len(self.following_platoons)):
                self.following_platoons[i].step(self.x[(i_first - 1) * 2:i_first * 2 + 1])
                i_first += 1
        self.opt_control = self.ref.opt_control(np.dot(self.A, self.x) + np.dot(self.B, -np.dot(self.ref.K, self.x))).copy()
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u) + np.random.normal(0,0.02,len(self.x))
        self.x[0] = self.leading_platoon.x_traj[self.k]
        u_ideal = self.u.copy()
        u_ideal[0] = 0
        self.lqg += np.dot(np.dot(self.x.T, self.Q), self.x) + np.dot(np.dot(u_ideal.T, self.R), u_ideal)
        self.x_list.append(self.x)
        self.u_list.append(u_ideal.copy())
        self.k += 1

        self.collect_state()
        if self.sim_type == SimType.ID or self.sim_type == SimType.ID_ON_DELTA or self.sim_type == SimType.HASH:
            self.upd_sync_prob()

    def get_centralized_control(self, threshold):
        ret_states = []
        for i in range(len(self.u)):
            if np.abs(self.opt_control[i] - self.u[i]) >= threshold:
            #    print("at k ", self.k)
                self.full_tx.append((i, self.k))
                p = np.random.random()
                if p < 255/256:
                    self.num_full_tx += 1
                    ret_states.append(self.opt_control[i])
                else:
                    ret_states.append(None)
            else:
                ret_states.append(None)
        return ret_states

    def get_centralized_control_h(self, threshold):
        ret_states = []
        for i in range(len(self.u)):
            if self.k % self.id_period[i]:
                ret_states.append(None)
                continue
            if not self.flags_in_desync[i]:
                if np.abs(self.opt_control[i] - self.u[i]) >= threshold:
                    self.timers_to_sync[i] += 1
                    if self.timers_to_sync[i] >= self.threshold_to_sync:
                        #if self.sim_type == SimType.ID_ON_DELTA or self.sim_type == SimType.HASH:
                        self.flags_in_desync[i] = 1
                        self.id_period[i] = 1
                        p = np.random.random()
                        if p < self.sync_probs[i]:
                            self.num_full_tx += 1
                            self.full_tx.append((i, self.k))
                            ret_states.append(self.opt_control[i])
                            self.last_sent_cntrl[i] = self.opt_control[i].copy()
                    else:
                        ret_states.append(None)
                        continue
                else:
                    ret_states.append(None)
                    continue
            else:
                if np.abs(self.opt_control[i] - self.u[i]) <=  threshold:
                    self.flags_in_desync[i] = 0
                    self.id_period[i] = self.id_period_max
                    self.timers_to_sync[i] = 0
                    ret_states.append(None)
                    continue
                else:
                    if self.sim_type == SimType.ID or self.sim_type == SimType.HASH:
                        if np.abs(self.opt_control[i] - self.u[i]) >= threshold:# np.abs(self.opt_control[i] - self.last_sent_cntrl[i]) >= threshold: #np.abs(self.opt_control[i] - self.u[i]) >= threshold:
                            self.timers_to_sync[i] += 1
                            #    print("at k ", self.k)
                            self.flags_in_desync[i] = 1
                            self.id_period[i] = 1
                            if self.timers_to_sync[i] >= self.threshold_to_sync:
                                p = np.random.random()
                                if p < self.sync_probs[i]:
                                    self.num_full_tx += 1
                                    self.full_tx.append((i, self.k))
                                    ret_states.append(self.opt_control[i])
                                    self.last_sent_cntrl[i] = self.opt_control[i].copy()
                                    # print(self.k, "send ", self.opt_control[i], "instead of ", self.u[i])
                            else:
                                ret_states.append(None)
                                continue
                        else:
                            ret_states.append(None)
                            continue

                    elif self.sim_type == SimType.ID_ON_DELTA:#  or self.sim_type == SimType.HASH:
                        if np.abs(self.opt_control[i] - self.last_sent_cntrl[i]) >= threshold:# np.abs(self.opt_control[i] - self.last_sent_cntrl[i]) >= threshold: #np.abs(self.opt_control[i] - self.u[i]) >= threshold:
                            self.timers_to_sync[i] += 1
                            #    print("at k ", self.k)
                            self.flags_in_desync[i] = 1
                            self.id_period[i] = 1
                            if self.timers_to_sync[i] >= self.threshold_to_sync:
                                p = np.random.random()
                                if p < self.sync_probs[i]:
                                    self.num_full_tx += 1
                                    self.full_tx.append((i, self.k))
                                    ret_states.append(self.opt_control[i])
                                    self.last_sent_cntrl[i] = self.opt_control[i].copy()
                            else:
                                ret_states.append(None)
                                continue
                        else:
                            ret_states.append(None)
                            continue
        return ret_states

    def collect_state(self):
        for i in range(2 * self.N - 1):
            self.states[i].append(self.x[i])

