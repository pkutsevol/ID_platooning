# ID_platooning
Simulation framework for testing applying identification codes for distributed control verification in cellular setup.

Simulation Numerical Parameters:
| Parameter | Value | Parameter | Value |
|-----------|--------|-----------|--------|
| Simapling period $\Delta$| $10$ ms | Control cost weights $w_d, w_v, w_T$ | $1000, 1, 1$ |
| Dynamic model parameters $\Theta_i, delta_i, k_{e_i}$ | $-3.6\times10^{-3}, 1.48\times10^{-5}, 0.148\times10^{-3}$ | Radio resources pool | $52$ Physical Resource Blocks |
| Error covariance components | $0.02$ | Markov Chain transition probabilities | LoS -> nLoS: $0.1$, nLoS -> LoS: $0.5$ |
| Error rates in LoS, nLoS | Data: $0.001, 0.1$, Control: $0.0001, 0.001$ |  Message length | $100$ B |
| ID tag length | $1$ B | Avg type 2 error probability | $\frac{1}{256}$ |
| External traffic intensity | $46.8$ MBps | Platoon length $N$| $7$ trucks |


Details on network model:

In the Centralized Control (CC) and Hybrid Control (HC) scenarios, we consider the BSs collocated with the RSUs, and all the vehicles in the platoon keep a stable connection with the same BS. Thus, we assume stable coverage and simultaneous handover of all the vehicles to a new BS. The wireless resources are shared with other users, such as other vehicles or humans. BS employs the RoundRobin Radio Resource Management scheme for DL, meaning all the connected UEs that request data transmission are assigned the same share of the resources. 

For a CC, the application requests the transmission of control inputs to every Follower Vehicle (FV) at every sampling interval $\Delta$, i.e., $\xi^C_i[k] = 1 \;\forall i \; \forall k$, where $\xi^C_i[k]$ is a binary indicator of requesting transmitting CC command to the vehicle $i$ at time step $k$. The envisioned drawbacks of CC are unpredictable delays and losses of critical control information in wireless mediums, bringing the danger of accidents on the road.

In the HC setup, the BS requests transmissions to FVs according to the utilized triggering scheme. Less frequent transmissions are envisioned to cause minor congestion compared to CC.

Other users generate Poisson traffic with constant average intensity, consuming part of the bandwidth. We consider the FIFO MAC queue at the BS that stores the packets containing calculated control inputs until sufficient resources are assigned for their transmission. 

The channel reliability includes the LoS effects vital for V2X communication due to high mobility in the corresponding scenarios. V2V connections are affected more drastically by LoS conditions. This fact favors prioritizing V2I links for collaboration. For each vehicle, we model the current channel with a 2-state Markov chain corresponding to LoS and nLoS conditions. Markov chain is defined by the transition probabilities $p_{LoSnLoS}$ and $p_{nLoSLos}$. The probabilities of remaining in the same state are residuals. The uplink and downlink channels transit to LoS or nLoS simultaneously since it is a geometrical effect that depends on the relative positions of the FVs and the BS. The error probabilities for LoS and nLoS in both downlink and uplink channels are set as parameters. Note that we require reliability. Thus, the BS retransmits lost DL packets.

