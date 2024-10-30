import numpy as np
from helpers import generate_Q, generate_R, generate_A_full, generate_B_full, generate_A_following, generate_B_following, SimType, ChannelType, ChannelState


class ErrorModel:
    def __init__(self):
        self.losnlos = 0.1
        self.nloslos = 0.5
        self.current_state = ChannelState.LoS

        self.errorrate_data = 0.001
        self.errorrate_data_good = 0.001
        self.errorrate_data_bad = 0.1

        self.errorrate_control = 0.0001
        self.errorrate_control_good = 0.0001
        self.errorrate_control_bad = 0.001


    def transit(self):
        p = np.random.random()
        if self.current_state == ChannelState.LoS:
            if p <= self.losnlos:
                self.current_state = ChannelState.NLoS
                self.errorrate_data = self.errorrate_data_bad
                self.errorrate_control = self.errorrate_control_bad
        elif self.current_state == ChannelState.NLoS:
            if p <= self.nloslos:
                self.current_state = ChannelState.LoS
                self.errorrate_data = self.errorrate_data_good
                self.errorrate_control = self.errorrate_control_good


class Network:
    def __init__(self, N, platoon):
        self.k = 0 # time step
        self.N = N
        self.truck_buffers = [[] for _ in range(N)]
        self.error_prob = 0.1
        self.arrival_rate = 13000#12000#        10000#15000#463.96 # per 10 ms #10000
        self.PRBs_per_ms = 52
        self.data_per_PRB = 36 # bytes
        self.scheduling_periodicity = 0.01 # s
        self.current_arrivals = 0
        self.scheduling_decision = None
        self.last_u = [0 for _ in range(self.N)]
        self.platoon = platoon


    def schedule(self):
        budget = np.random.poisson(lam=10 * self.PRBs_per_ms)

        waiting_user = np.sum([len(lst) for lst in self.truck_buffers]) + self.current_arrivals
        scheduled = np.random.choice(np.arange(waiting_user), min(budget, waiting_user), replace=False)
        i_last = 0
        for i in range(self.N):
            for place in range(i_last, i_last + len(self.truck_buffers[i])):
                if place in scheduled:
                    self.schedule_truck(i, self.truck_buffers[i][0])
                    budget -= 1
                    waiting_user -= 1
            i_last += len(self.truck_buffers[i]) + 1



    def step(self, trucks_states):
        self.current_arrivals = np.random.poisson(lam=self.arrival_rate)
        for i in range(len(trucks_states)):
            if trucks_states[i] is not None and len(self.truck_buffers[i]) <= 30:
                self.truck_buffers[i].append((self.k, trucks_states[i]))
        if not (self.platoon.sim_type == SimType.CENTRALIZED or self.platoon.sim_type == SimType.ID or self.platoon.sim_type == SimType.ID_ON_DELTA or self.platoon.sim_type == SimType.HASH):
            self.last_u = [None for _ in range(self.N)]
        if self.platoon.sim_type == SimType.ID or self.platoon.sim_type == SimType.HASH:
            for i in range(len(trucks_states)):
                self.last_u[i] = None
        if self.platoon.sim_type == SimType.ID_ON_DELTA:# or self.platoon.sim_type == SimType.HASH:
            for i in range(len(trucks_states)):
                if not self.platoon.flags_in_desync[i]:
                    self.last_u[i] = None
        self.schedule()
        self.k += 1

        return np.array(self.last_u)

    def schedule_truck(self, i, state):
        p = np.random.random()
        if p < 1 - self.platoon.following_platoons[i-1].link.errorrate_data:
            self.last_u[i] = state[1]
            self.truck_buffers[i] = self.truck_buffers[i][1:]


