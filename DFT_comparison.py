#imports
from mpi4py import MPI
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
import sys

#unchangeable constants
MASTER = 0
AVG_FLOPS_COST = 0.000000000001 #1e-12
AVG_BANDWIDTH_COST = 0.000000001 #1e-9
AVG_LATENCY_COST = 0.000001 #1e-6

#changeable constants
TASKS_TRESHOLD = 1024
MIN_P = 1
I = 17

MATRIX_SIZE = 2**14
SMALL_MATRIX_SIZE = 2**12

B_LIST = np.array([0.5, 1, 250])
R = 100
MDS_PARAM = 1.2

def simulate_run(N, P, B):
    results = np.zeros(shape=[7, 3], dtype=float)
    number_of_tasks = N * N

    delay_factor_list = np.random.exponential(1, P)/B + 1

    if (P == 1):
        for i in range(7):
            results[i][0] = AVG_FLOPS_COST * number_of_tasks * N * delay_factor_list[0]

        return results.transpose()

    delay_factor_list = delay_factor_list * ((1 + 1/B) / delay_factor_list.mean())
    help_rep_delay_list = delay_factor_list.reshape(2, P//2).min(axis=0)
    delay_factor_list.sort()

    # --------- multiplicaiton cost ---------#
    base_cost = AVG_FLOPS_COST * number_of_tasks * N
    results[0][0] += base_cost * delay_factor_list.mean()            # LB
    results[1][0] += base_cost * delay_factor_list[-1]               # NM
    results[2][0] += base_cost * delay_factor_list.mean()            # WS
    results[3][0] += base_cost * help_rep_delay_list[-1] * 2         #Rep(2)
    results[4][0] += base_cost * delay_factor_list[int(P/MDS_PARAM)] * MDS_PARAM #MDS(1.2)
    results[5][0] += base_cost * delay_factor_list.mean()  # LT+
    results[6][0] += base_cost * delay_factor_list.mean()            # SLB

    #--------- additional cost ---------#
    # --------- additional bandwidth cost ---------#

    memory_size = 2 * N * int(np.sqrt(number_of_tasks))
    results[3][1] += AVG_BANDWIDTH_COST * memory_size  # Rep(2)
    results[4][1] += AVG_BANDWIDTH_COST * memory_size * np.log(P)  # MDS(1.1)
    results[5][1] += AVG_BANDWIDTH_COST * memory_size * np.log(P)  # LT+

    # --------- additional latency cost ---------#
    results[3][2] += AVG_LATENCY_COST             # Rep(2)
    results[4][2] += AVG_LATENCY_COST * np.log(P) # MDS(1.1)
    results[5][2] += AVG_LATENCY_COST * N * np.log(P) * np.log(P)  # LT+

    number_of_tasks = int((1 - (delay_factor_list[0] / delay_factor_list)).sum() * (number_of_tasks / P))
    while number_of_tasks > TASKS_TRESHOLD:
        # --------- additional bandwidth cost (WS + SLB) ---------#
        memory_size = 2 * N * int(np.sqrt(number_of_tasks))
        results[2][1] += AVG_BANDWIDTH_COST * memory_size  # WS
        results[6][1] += AVG_BANDWIDTH_COST * memory_size  # SLB

        # --------- additional latency cost (WS + SLB) ---------#
        results[2][2] += AVG_LATENCY_COST * P          #WS
        results[6][2] += AVG_LATENCY_COST * np.log(P)  #SLB

        number_of_tasks = int((1 - (delay_factor_list[0] / delay_factor_list)).sum() * (number_of_tasks / P))

    return results.transpose()

# ------------------------------------- #
# ----------- main function ----------- #
# ------------------------------------- #
#create main communicator (all the regular processors)
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

graph_number = int(sys.argv[1])

if graph_number == 1:
    delay_factor = B_LIST[1]
    matrix_dimension = 2**14

    graph = np.zeros(shape=[I, 3, 7], dtype=float)
    for j in range(R):
        P = MIN_P
        for i in range(I):
            graph[i] += simulate_run(matrix_dimension, P, delay_factor) / R
            P *= 2

    graph = graph.transpose()

    # weak_scale_graph
    LB_normalization = graph[6].sum(axis=0)
    LB_graph = graph[0] / LB_normalization
    NM_graph = graph[1] / LB_normalization
    WS_graph = graph[2] / LB_normalization
    REP_graph = graph[3] / LB_normalization
    MDS_graph = graph[4] / LB_normalization
    LTP_graph = graph[5] / LB_normalization
    SLB_graph = graph[6] / LB_normalization

    x = np.arange(I)

    # print("-------- NM --------")
    plt.plot(x, NM_graph.sum(axis=0), '-H', label="Non Mitigated")

    # print("-------- MDS --------")
    plt.plot(x, MDS_graph.sum(axis=0), '-v', label="MDS", color='m')

    # print("-------- LT+ --------")
    plt.plot(x, LTP_graph.sum(axis=0), '-s', label="LT+", color='y')

    # print("-------- REP --------")
    plt.plot(x, REP_graph.sum(axis=0), '-^', label="Replication", color='c')

    # print("-------- WS --------")
    plt.plot(x, WS_graph.sum(axis=0), '-D', label="Work Stealing", color='r')

    # print("-------- SLB --------")
    plt.plot(x, SLB_graph.sum(axis=0), '-o', label="SLB", color='g')

    # print("-------- LB --------")
    plt.plot(x, LB_graph.sum(axis=0), '-x', label="Lower Bound", color='k')

    plt.xlabel('Number of Processors', size=18)
    plt.ylabel('Relative Time to SLB', size=18)
    ticks_help = np.arange(I // 2 + 1) * 2
    plt.xticks(ticks_help, [(2 ** i) for i in ticks_help], fontsize=14)
    plt.yticks(0.5 + np.arange(12)/2, fontsize=14)
    plt.legend(prop={'size': 18}, loc=2)
    plt.show()

if graph_number == 2:
    delay_factor = B_LIST[0]
    matrix_dimension = 2**14

    graph = np.zeros(shape=[I, 3, 7], dtype=float)
    for j in range(R):
        P = MIN_P
        for i in range(I):
            graph[i] += simulate_run(matrix_dimension, P, delay_factor) / R
            P *= 2

    graph = graph.transpose()

    # weak_scale_graph
    LB_normalization = graph[6].sum(axis=0)
    LB_graph = graph[0] / LB_normalization
    NM_graph = graph[1] / LB_normalization
    WS_graph = graph[2] / LB_normalization
    REP_graph = graph[3] / LB_normalization
    MDS_graph = graph[4] / LB_normalization
    LTP_graph = graph[5] / LB_normalization
    SLB_graph = graph[6] / LB_normalization

    x = np.arange(I)

    # print("-------- NM --------")
    #plt.plot(x, NM_graph.sum(axis=0), '-H', label="Non Mitigated")

    # print("-------- MDS --------")
    plt.plot(x, MDS_graph.sum(axis=0), '-v', label="MDS", color='m')

    # print("-------- LT+ --------")
    plt.plot(x, LTP_graph.sum(axis=0), '-s', label="LT+", color='y')

    # print("-------- REP --------")
    plt.plot(x, REP_graph.sum(axis=0), '-^', label="Replication", color='c')

    # print("-------- WS --------")
    plt.plot(x, WS_graph.sum(axis=0), '-D', label="Work Stealing", color='r')

    # print("-------- SLB --------")
    plt.plot(x, SLB_graph.sum(axis=0), '-o', label="SLB", color='g')

    # print("-------- LB --------")
    plt.plot(x, LB_graph.sum(axis=0), '-x', label="Lower Bound", color='k')

    plt.xlabel('Number of Processors', size=18)
    plt.ylabel('Relative Time to SLB', size=18)
    ticks_help = np.arange(I // 2 + 1) * 2
    plt.xticks(ticks_help, [(2 ** i) for i in ticks_help], fontsize=14)
    plt.yticks(0.8 + np.arange(8) / 5, fontsize=14)
    plt.legend(prop={'size': 18}, loc=2)
    plt.show()

if graph_number == 3:
    delay_factor = B_LIST[2]
    matrix_dimension = 2 ** 14

    graph = np.zeros(shape=[I, 3, 7], dtype=float)
    for j in range(R):
        P = MIN_P
        for i in range(I):
            graph[i] += simulate_run(matrix_dimension, P, delay_factor) / R
            P *= 2

    graph = graph.transpose()

    # weak_scale_graph
    LB_normalization = graph[6].sum(axis=0)
    LB_graph = graph[0] / LB_normalization
    NM_graph = graph[1] / LB_normalization
    WS_graph = graph[2] / LB_normalization
    REP_graph = graph[3] / LB_normalization
    MDS_graph = graph[4] / LB_normalization
    LTP_graph = graph[5] / LB_normalization
    SLB_graph = graph[6] / LB_normalization

    x = np.arange(I)

    # print("-------- LT+ --------")
    plt.plot(x, LTP_graph.sum(axis=0), '-s', label="LT+", color='y')

    # print("-------- MDS --------")
    plt.plot(x, MDS_graph.sum(axis=0), '-v', label="MDS", color='m')

    # print("-------- REP --------")
    plt.plot(x, REP_graph.sum(axis=0), '-^', label="Replication", color='c')

    # print("-------- NM --------")
    plt.plot(x, NM_graph.sum(axis=0), '-H', label="Non Mitigated")

    # print("-------- WS --------")
    plt.plot(x, WS_graph.sum(axis=0), '-D', label="Work Stealing", color='r')

    # print("-------- SLB --------")
    plt.plot(x, SLB_graph.sum(axis=0), '-o', label="SLB", color='g')

    # print("-------- LB --------")
    plt.plot(x, LB_graph.sum(axis=0), '-x', label="Lower Bound", color='k')

    plt.xlabel('Number of Processors', size=18)
    plt.ylabel('Relative Time to SLB', size=18)
    ticks_help = np.arange(I // 2 + 1) * 2
    plt.xticks(ticks_help, [(2 ** i) for i in ticks_help], fontsize=14)
    plt.yticks(0.8 + np.arange(12) / 5, fontsize=14)
    plt.legend(prop={'size': 18}, loc=2)
    plt.show()

if graph_number == 32:
    delay_factor = B_LIST[2]
    matrix_dimension = 2 ** 14

    graph = np.zeros(shape=[I, 3, 7], dtype=float)
    for j in range(R):
        P = MIN_P
        for i in range(I):
            graph[i] += simulate_run(matrix_dimension, P, delay_factor) / R
            P *= 2

    graph = graph.transpose()

    # weak_scale_graph
    LB_normalization = graph[6].sum(axis=0)
    LB_graph = graph[0] / LB_normalization
    NM_graph = graph[1] / LB_normalization
    WS_graph = graph[2] / LB_normalization
    REP_graph = graph[3] / LB_normalization
    MDS_graph = graph[4] / LB_normalization
    LTP_graph = graph[5] / LB_normalization
    SLB_graph = graph[6] / LB_normalization

    x = np.arange(I)

    # print("-------- NM --------")
    plt.plot(x, NM_graph.sum(axis=0), '-H', label="Non Mitigated")

    # print("-------- WS --------")
    plt.plot(x, WS_graph.sum(axis=0), '-D', label="Work Stealing", color='r')

    # print("-------- SLB --------")
    plt.plot(x, SLB_graph.sum(axis=0), '-o', label="SLB", color='g')

    # print("-------- LB --------")
    plt.plot(x, LB_graph.sum(axis=0), '-x', label="Lower Bound", color='k')

    plt.xlabel('Number of Processors', size=18)
    plt.ylabel('Relative Time to SLB', size=18)
    ticks_help = np.arange(I // 2 + 1) * 2
    plt.xticks(ticks_help, [(2 ** i) for i in ticks_help], fontsize=14)
    plt.yticks(0.990 + np.arange(11) / 200, fontsize=14)
    plt.legend(prop={'size': 18}, loc=2)
    plt.show()

if graph_number == 4:
    delay_factor = B_LIST[1]
    matrix_dimension = 2 ** 12

    graph = np.zeros(shape=[I, 3, 7], dtype=float)
    for j in range(R):
        P = MIN_P
        for i in range(I):
            graph[i] += simulate_run(matrix_dimension, P, delay_factor) / R
            P *= 2

    graph = graph.transpose()

    # weak_scale_graph
    LB_normalization = graph[6].sum(axis=0)
    LB_graph = graph[0] / LB_normalization
    NM_graph = graph[1] / LB_normalization
    WS_graph = graph[2] / LB_normalization
    REP_graph = graph[3] / LB_normalization
    MDS_graph = graph[4] / LB_normalization
    LTP_graph = graph[5] / LB_normalization
    SLB_graph = graph[6] / LB_normalization

    x = np.arange(I)

    # print("-------- LT+ --------")
    plt.plot(x, LTP_graph.sum(axis=0), '-s', label="LT+", color='y')

    # print("-------- NM --------")
    plt.plot(x, NM_graph.sum(axis=0), '-H', label="Non Mitigated")

    # print("-------- WS --------")
    plt.plot(x, WS_graph.sum(axis=0), '-D', label="Work Stealing", color='r')

    # print("-------- MDS --------")
    plt.plot(x, MDS_graph.sum(axis=0), '-v', label="MDS", color='m')

    # print("-------- REP --------")
    plt.plot(x, REP_graph.sum(axis=0), '-^', label="Replication", color='c')

    # print("-------- SLB --------")
    plt.plot(x, SLB_graph.sum(axis=0), '-o', label="SLB", color='g')

    # print("-------- LB --------")
    plt.plot(x, LB_graph.sum(axis=0), '-x', label="Lower Bound", color='k')

    plt.xlabel('Number of Processors', size=18)
    plt.ylabel('Relative Time to SLB', size=18)
    ticks_help = np.arange(I // 2 + 1) * 2
    plt.xticks(ticks_help, [(2 ** i) for i in ticks_help], fontsize=14)
    plt.yticks(0.5 + np.arange(11) / 2, fontsize=14)
    plt.legend(prop={'size': 18}, loc=2)
    plt.show()

if graph_number == 5:
    delay_factor = B_LIST[0]
    matrix_dimension = 2 ** 12

    graph = np.zeros(shape=[I, 3, 7], dtype=float)
    for j in range(R):
        P = MIN_P
        for i in range(I):
            graph[i] += simulate_run(matrix_dimension, P, delay_factor) / R
            P *= 2

    graph = graph.transpose()

    # weak_scale_graph
    LB_normalization = graph[6].sum(axis=0)
    LB_graph = graph[0] / LB_normalization
    NM_graph = graph[1] / LB_normalization
    WS_graph = graph[2] / LB_normalization
    REP_graph = graph[3] / LB_normalization
    MDS_graph = graph[4] / LB_normalization
    LTP_graph = graph[5] / LB_normalization
    SLB_graph = graph[6] / LB_normalization

    x = np.arange(I)

    # print("-------- NM --------")
    #plt.plot(x, NM_graph.sum(axis=0), '-H', label="Non Mitigated")

    # print("-------- WS --------")
    plt.plot(x, WS_graph.sum(axis=0), '-D', label="Work Stealing", color='r')

    # print("-------- LT+ --------")
    plt.plot(x, LTP_graph.sum(axis=0), '-s', label="LT+", color='y')

    # print("-------- MDS --------")
    plt.plot(x, MDS_graph.sum(axis=0), '-v', label="MDS", color='m')

    # print("-------- REP --------")
    plt.plot(x, REP_graph.sum(axis=0), '-^', label="Replication", color='c')

    # print("-------- SLB --------")
    plt.plot(x, SLB_graph.sum(axis=0), '-o', label="SLB", color='g')

    # print("-------- LB --------")
    plt.plot(x, LB_graph.sum(axis=0), '-x', label="Lower Bound", color='k')

    plt.xlabel('Number of Processors', size=18)
    plt.ylabel('Relative Time to SLB', size=18)
    ticks_help = np.arange(I // 2 + 1) * 2
    plt.xticks(ticks_help, [(2 ** i) for i in ticks_help], fontsize=14)
    plt.yticks(0.5 + np.arange(9) / 2, fontsize=14)
    plt.legend(prop={'size': 18}, loc=2)
    plt.show()

if graph_number == 6:
    delay_factor = B_LIST[2]
    matrix_dimension = 2 ** 12

    graph = np.zeros(shape=[I, 3, 7], dtype=float)
    for j in range(R):
        P = MIN_P
        for i in range(I):
            graph[i] += simulate_run(matrix_dimension, P, delay_factor) / R
            P *= 2

    graph = graph.transpose()

    # weak_scale_graph
    LB_normalization = graph[6].sum(axis=0)
    LB_graph = graph[0] / LB_normalization
    NM_graph = graph[1] / LB_normalization
    WS_graph = graph[2] / LB_normalization
    REP_graph = graph[3] / LB_normalization
    MDS_graph = graph[4] / LB_normalization
    LTP_graph = graph[5] / LB_normalization
    SLB_graph = graph[6] / LB_normalization

    x = np.arange(I)

    # print("-------- LT+ --------")
    plt.plot(x, LTP_graph.sum(axis=0), '-s', label="LT+", color='y')

    # print("-------- MDS --------")
    plt.plot(x, MDS_graph.sum(axis=0), '-v', label="MDS", color='m')

    # print("-------- REP --------")
    plt.plot(x, REP_graph.sum(axis=0), '-^', label="Replication", color='c')

    # print("-------- WS --------")
    plt.plot(x, WS_graph.sum(axis=0), '-D', label="Work Stealing", color='r')

    # print("-------- NM --------")
    plt.plot(x, NM_graph.sum(axis=0), '-H', label="Non Mitigated")

    # print("-------- SLB --------")
    plt.plot(x, SLB_graph.sum(axis=0), '-o', label="SLB", color='g')

    # print("-------- LB --------")
    plt.plot(x, LB_graph.sum(axis=0), '-x', label="Lower Bound", color='k')

    plt.xlabel('Number of Processors', size=18)
    plt.ylabel('Relative Time to SLB', size=18)
    ticks_help = np.arange(I // 2 + 1) * 2
    plt.xticks(ticks_help, [(2 ** i) for i in ticks_help], fontsize=14)
    plt.yticks(np.arange(8) * 2, fontsize=14)
    plt.legend(prop={'size': 18}, loc=2)
    plt.show()

if graph_number == 62:
    delay_factor = B_LIST[2]
    matrix_dimension = 2 ** 12

    graph = np.zeros(shape=[I, 3, 7], dtype=float)
    for j in range(R):
        P = MIN_P
        for i in range(I):
            graph[i] += simulate_run(matrix_dimension, P, delay_factor) / R
            P *= 2

    graph = graph.transpose()

    # weak_scale_graph
    LB_normalization = graph[6].sum(axis=0)
    LB_graph = graph[0] / LB_normalization
    NM_graph = graph[1] / LB_normalization
    WS_graph = graph[2] / LB_normalization
    REP_graph = graph[3] / LB_normalization
    MDS_graph = graph[4] / LB_normalization
    LTP_graph = graph[5] / LB_normalization
    SLB_graph = graph[6] / LB_normalization

    x = np.arange(I)

    # print("-------- NM --------")
    plt.plot(x, NM_graph.sum(axis=0), '-H', label="Non Mitigated")

    # print("-------- SLB --------")
    plt.plot(x, SLB_graph.sum(axis=0), '-o', label="SLB", color='g')

    # print("-------- LB --------")
    plt.plot(x, LB_graph.sum(axis=0), '-x', label="Lower Bound", color='k')

    plt.xlabel('Number of Processors', size=18)
    plt.ylabel('Relative Time to SLB', size=18)
    ticks_help = np.arange(I // 2 + 1) * 2
    plt.xticks(ticks_help, [(2 ** i) for i in ticks_help], fontsize=14)
    plt.yticks(0.97 + np.arange(6) / 100, fontsize=14)
    plt.legend(prop={'size': 18}, loc=2)
    plt.show()