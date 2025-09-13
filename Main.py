import numpy as np
import pandas as pd

from Global_Vars import Global_Vars

# Read Dataset
an = 0
if an == 1:
    df = pd.read_csv('./archive/diabetes.csv')
    data = df.values[:, :-1]
    target = df.values[:, -1]
    np.save('Data.npy', data)
    np.save('Target.npy', target)

num_of_node = [50, 100, 150, 200]

# Network Initialization
an = 0
if an == 1:
    Net = []
    Tar = []
    for i in range(len(num_of_node)):
        Node_name = np.random.randint(1, num_of_node[i], size=(num_of_node[i], 1))
        Node_Pos = np.random.uniform(0, num_of_node[i], size=(num_of_node[i], num_of_node[i]))
        Topology = np.random.uniform(0, 1, size=(50, 50))
        Traversal = np.random.uniform(0, 1, size=(50, 50))
        Response_Time = np.random.randint(1, 100, size=(num_of_node[i], 1))
        Packet_Delivery_Rate = np.random.randint(1, 100, size=(num_of_node[i], 1))
        squared_dists = np.sum((Topology[:, None] - Traversal) ** 2, axis=2)
        distance_matrix = np.sqrt(squared_dists)
        Data = [Node_name, Node_Pos, Topology, Traversal, distance_matrix, Response_Time, Packet_Delivery_Rate]
        Target = np.random.uniform(0, 1, size=(num_of_node[i], 1))
        Net.append(Data)
        Tar.append(Target)
    np.save('Init_Network.npy', Net)
    np.save('Target.npy', Tar)


# creating an structure
class Structure:
    S_in_G = []
    S_in_type = []
    S_in_E = []
    S_in_ENERGY = []
    S_in_xd = []
    S_in_yd = []


def Routing():
    # Optimization for Routing
    an = 0
    if an == 1:
        mitigation = []
        num_of_node = [50, 100, 150, 200]
        # Field Dimensions - x and y maximum (in meters)
        xm = 100
        ym = 100
        #  maximum number of rounds
        rmax = 2000
        Global_Vars.rmax = rmax
        #  Optimization paramateres
        no_sol = 10
        dim_sol = 30
        iteration_count = 10
        S_in = Structure()
        # x and y Coordinates of the Sink
        sink_x = 0.5 * xm
        sink_y = 0.5 * ym
        for network in range(len(num_of_node)):
            # Number of Nodes in the field
            n = num_of_node[network]
            Global_Vars.n = n
            # Optimal Election Probability of a node
            # to become cluster head
            p = 0.1
            Global_Vars.p = p
            # Energy Model (all values in Joules)
            # Initial Energy
            Eo = 0.3
            # Eelec=Etx=Erx
            ETX = 50 * 1e-09
            ERX = 50 * 1e-09
            # Transmit Amplifier types
            Efs = 10 * 1e-12
            Emp = 0.0013 * 1e-12
            # Data Aggregation Energy
            EDA = 5 * 1e-09

            Total_Packets = 1000
            Packet_loss = np.random.randint(0, 5, size=(1, n))
            # Values for Hetereogeneity
            # Percentage of nodes than are advanced
            m = 0.1
            # alpha
            a = 1
            ## Creation of the random Sensor Network
            S_in_xd = np.zeros((n + 1))
            S_in_yd = np.zeros((n + 1))
            XR = np.zeros((n + 1))
            YR = np.zeros((n + 1))
            S_in_G = np.zeros((n + 1))
            S_in_type = []
            S_in_ENERGY = np.zeros((n + 1))
            S_in_E = np.zeros((n + 1))
            for i in range(n + 1):
                S_i_xd = np.random.rand(1) * xm
                S_in_xd[i] = S_i_xd
                XR[i] = S_i_xd
                S_i_yd = (np.random.rand(1, 1) * ym)
                S_in_yd[i] = (S_i_yd)
                YR[i] = (S_i_yd)
                S_i_G = 0

                # initially there are no cluster heads only nodes
                S_i_type = 'N'
                S_in_G[i] = S_i_G
                S_in_type.append(S_i_type)
                temp_rnd0 = i

                # Random Election of Normal Nodes
                if temp_rnd0 >= m * n + 1:
                    S_i_E = Eo
                    S_i_ENERGY = 0

                    S_in_E[i] = S_i_E
                    S_in_ENERGY[i] = S_i_ENERGY

                # Random Election of Advanced Nodes
                if temp_rnd0 < m * n + 1:
                    S_i_E = Eo * (1 + a)
                    S_i_ENERGY = 1

                    S_in_E[i] = S_i_E
                    S_in_ENERGY[i] = S_i_ENERGY

            S_in_xd[n] = sink_x
            S_in_yd[n] = sink_y
            S_in.S_in_G = S_in_G
            S_in.S_in_type = S_in_type
            S_in.S_in_xd = S_in_xd
            S_in.S_in_yd = S_in_yd
            S_in.S_in_E = S_in_E
            S_in.S_in_ENERGY = S_in_ENERGY
            Global_Vars.S_in = S_in

            n_nodes = num_of_node[network]
            Npop = 10
            Ch_len = 10
            xmin = np.ones((1, Ch_len))
            xmax = np.multiply(n, np.ones((Npop, Ch_len)))
            initsol = np.random.uniform(xmin, xmax)
            max_iter = 100
            fname = Objective_Function

            print("RDA...")
            [bestfit1, fitness1, bestsol1, time1] = RDA(initsol, fname, xmin, xmax, max_iter)
            mitigation1 = fname(bestsol1)

            print("EOA...")
            [bestfit2, fitness2, bestsol2, time2] = EOA(initsol, fname, xmin, xmax, max_iter)
            mitigation2 = fname(bestsol2)

            print("SGO...")
            [bestfit3, fitness3, bestsol3, time3] = SGO(initsol, fname, xmin, xmax, max_iter)
            mitigation3 = fname(bestsol3)

            print("DSA...")
            [bestfit4, fitness4, bestsol4, time4] = DSA(initsol, fname, xmin, xmax, max_iter)
            mitigation4 = fname(bestsol4)

            print("PROPOSED...")
            [bestfit5, fitness5, bestsol5, time5] = Proposed(initsol, fname, xmin, xmax, max_iter)
            mitigation5 = fname(bestsol5)

            mitigation.append([mitigation1, mitigation2, mitigation3, mitigation4, mitigation5])
            Fitness = [fitness1, fitness2, fitness3, fitness4, fitness5]

            np.save('Fitness.npy', Fitness)
        np.save('Res.npy', mitigation)

