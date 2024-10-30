import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque, OrderedDict
from itertools import count
import numpy as np
import scipy
import scipy.linalg
from enum import Enum

class SimType(Enum):
    CENTRALIZED = 1
    IDEAL = 2
    DISTRIBUTED = 3
    ID = 4
    ID_ON_DELTA = 5
    HASH = 6

class ChannelType(Enum):
    CONTROL = 1
    DATA = 2

class ChannelState(Enum):
    LoS = 1
    NLoS = 2

def generate_A_full(N, theta, delta, sampling_period): #state matrix with leader
    A = []
    for i in range(2 * N - 1):
        line = np.zeros(2 * N - 1)
        if i == 0:
            line[i] = theta
        elif i%2:
            line[i-1] = 1
            line[i+1] = -1
        else:
            line[i-1] = delta
            line[i] = theta
        A.append(line)
    return np.array(A) * sampling_period + np.eye(2 * N - 1)


def generate_A_following(N, theta, delta, sampling_period, L_prev, k_e): #state matrix without leader
    A = []
    for i in range(2 * N - 1):
        line = np.zeros(2 * N - 1)
        if i == 0:
            line[i] = theta - k_e * L_prev
        elif i%2:
            line[i-1] = 1
            line[i+1] = -1
        else:
            line[i-1] = delta
            line[i] = theta
        A.append(line)
    return np.array(A) * sampling_period + np.eye(2 * N - 1)


def generate_B_full(N, k_e): #input matrix with leader
    B = []
    for i in range(2 * N - 1):
        line = np.zeros(N)
        if i%2 or i==0:
            pass
        else:
            line[int(i/2)] = k_e
        B.append(line)
    return np.array(B)


def generate_B_following(N, k_e): #input matrix without leader
    B = []
    for i in range(2 * N - 1):
        line = np.zeros(N-1)
        if i%2 or i==0:
            pass
        else:
            line[int(i/2)-1] = k_e
        B.append(line)
    return np.array(B)


def generate_Q(N, w_vel, w_dist, tau):
    Q = []
    w_tau = 0
    tau = 1
    if N == 1:
        return np.array([w_vel])
    for i in range(2 * N - 1):
        line = np.zeros(2 * N - 1)
        if i == 0:
            line[i] = w_vel
            line[i+2] = -w_vel
        elif i%2:
            line[i] = w_dist + w_tau
            line[i+1] = -tau * w_tau
        else:
            if i == 2 * N - 2:
                line[i] = w_vel + tau**2 * w_tau
                line[i-1] = -tau * w_tau
                line[i-2] = -w_vel
            else:
                line[i] = 2 * w_vel + tau**2 * w_tau
                line[i-1] =  -tau * w_tau
                line[i-2] =  -w_vel
                line[i+2] =  -w_vel
        Q.append(line)
    return np.array(Q)


def generate_R(N, w):
    ret = w * np.eye(N)
    return ret


def count_dangers(lst, th):
    cnt = 0
    for el in lst:
        if el < th:
            cnt += 1
    return cnt