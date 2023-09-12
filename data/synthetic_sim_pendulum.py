import numpy as np
import matplotlib.pyplot as plt
import time


class PendulumSim(object):
    def __init__(self, n_sticks=3, stick_length=1, stick_mass=1, loc_std=.1, vel_norm=.5,
                noise_var=0.):
        self.n_sticks = n_sticks
        self.stick_length = stick_length
        self.loc_std = loc_std
        self.vel_norm = vel_norm
        
        self.noise_var = noise_var
        self.stick_mass = stick_mass
        
        self._delta_T = 0.00001
        self.g = 9.8

    def _energy(self, loc, vel):
        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide='ignore'):
            U=-self.stick_mass*self.stick_length*self.g/2*(5*np.cos(loc[0])+3*np.cos(loc[1])+1*np.cos(loc[2]))
            K=self.stick_mass*self.stick_length*self.stick_length/6*(9*vel[1]*vel[0]*np.cos(loc[0]-loc[1])+3*vel[2]*vel[0]*np.cos(loc[0]-loc[2])+3*vel[2]*vel[1]*np.cos(loc[1]-loc[2])+7*vel[0]*vel[0]+4*vel[1]*vel[1]+1*vel[2]*vel[2])

            print('U: ', U)
            print('K: ', K)
            print('energy:',U+K)

            return U + K
    def generate_static_graph(self):
        # Sample edges: without self-loop
        edges = np.eye(self.n_sticks, k=1) + np.eye(self.n_sticks, k=-1)

        return edges


    def calculate_angular_speed(self, loc_next, p_next):
        vel_next = np.zeros((1, self.n_sticks))
        vel_next[0,0] = 6 * (9*p_next[0,0]*np.cos(2*(loc_next[0,1]-loc_next[0,2])) + 27*p_next[0,1]*np.cos(loc_next[0,0]-loc_next[0,1]) - 9*p_next[0,1]*np.cos(loc_next[0,0]+loc_next[0,1]-2*loc_next[0,2]) + 21*p_next[0,2]*np.cos(loc_next[0,0]-loc_next[0,2])- 27*p_next[0,2]*np.cos(loc_next[0,0]-2*loc_next[0,1]+loc_next[0,2]) - p_next[0,0]*23) / (self.stick_mass*self.stick_length*self.stick_length*(81*np.cos(2*(loc_next[0,0]-loc_next[0,1])) - 9*np.cos(2*(loc_next[0,0]-loc_next[0,2])) + 45*np.cos(2*(loc_next[0,1]-loc_next[0,2]))- 169))
        vel_next[0,1] = 6 * (27*p_next[0,0]*np.cos(loc_next[0,0]-loc_next[0,1]) -9* p_next[0,0]*np.cos(loc_next[0,0]+loc_next[0,1]-2*loc_next[0,2]) + 9*p_next[0,1]*np.cos(2*(loc_next[0,0]-loc_next[0,2])) - 27*p_next[0,2]*np.cos(2*loc_next[0,0]-loc_next[0,1]-loc_next[0,2]) + 57*p_next[0,2]*np.cos(loc_next[0,1]-loc_next[0,2]) - p_next[0,1]*47) / (self.stick_mass*self.stick_length*self.stick_length*(81*np.cos(2*(loc_next[0,0]-loc_next[0,1])) - 9*np.cos(2*(loc_next[0,0]-loc_next[0,2])) + 45*np.cos(2*(loc_next[0,1]-loc_next[0,2]))- 169))
        vel_next[0,2] = 6 * (21*p_next[0,0]*np.cos(loc_next[0,0]-loc_next[0,2]) - 27*p_next[0,0]*np.cos(loc_next[0,0]-2*loc_next[0,1]+loc_next[0,2]) - 27*p_next[0,1]*np.cos(2*loc_next[0,0]-loc_next[0,1]-loc_next[0,2])+ 57*p_next[0,1]*np.cos(loc_next[0,1]-loc_next[0,2])+ 81*p_next[0,2]*np.cos(2*(loc_next[0,0]-loc_next[0,1])) - p_next[0,2]*143)  / (self.stick_mass*self.stick_length*self.stick_length*(81*np.cos(2*(loc_next[0,0]-loc_next[0,1])) - 9*np.cos(2*(loc_next[0,0]-loc_next[0,2])) + 45*np.cos(2*(loc_next[0,1]-loc_next[0,2]))- 169))
        up=6 * (9*p_next[0,0]*np.cos(2*(loc_next[0,1]-loc_next[0,2])) + 27*p_next[0,1]*np.cos(loc_next[0,0]-loc_next[0,1]) - 9*p_next[0,1]*np.cos(loc_next[0,0]+loc_next[0,1]-2*loc_next[0,2]) + 21*p_next[0,2]*np.cos(loc_next[0,0]-loc_next[0,2])- 27*p_next[0,2]*np.cos(loc_next[0,0]-2*loc_next[0,1]+loc_next[0,2]) - p_next[0,0]*23)
        down=self.stick_mass*self.stick_length*self.stick_length*(81*np.cos(2*(loc_next[0,0]-loc_next[0,1])) - 9*np.cos(2*(loc_next[0,0]-loc_next[0,2])) + 45*np.cos(2*(loc_next[0,1]-loc_next[0,2]))- 169)
        return vel_next

    def calculate_p_dot(self, loc_next, vel_next):
        p_dot = np.zeros((1, self.n_sticks))
        p_dot[0,0] = -1/2 * self.stick_mass * self.stick_length * (3*vel_next[0,1]*vel_next[0,0]*self.stick_length*np.sin(loc_next[0,0]-loc_next[0,1]) + vel_next[0,0]*vel_next[0,2]*self.stick_length*np.sin(loc_next[0,0]-loc_next[0,2]) + 5*self.g*np.sin(loc_next[0,0])) 
        p_dot[0,1] = -1/2 * self.stick_mass * self.stick_length * (-3*vel_next[0,1]*vel_next[0,0]*self.stick_length*np.sin(loc_next[0,0]-loc_next[0,1]) + vel_next[0,1]*vel_next[0,2]*self.stick_length*np.sin(loc_next[0,1]-loc_next[0,2])+ 3*self.g*np.sin(loc_next[0,1]))
        p_dot[0,2] = -1/2 * self.stick_mass * self.stick_length * (vel_next[0,0]*vel_next[0,2]*self.stick_length*np.sin(loc_next[0,0]-loc_next[0,2]) + vel_next[0,1]*vel_next[0,2]*self.stick_length*np.sin(loc_next[0,1]-loc_next[0,2]) - self.g*np.sin(loc_next[0,2]))

        return p_dot

    def sample_trajectory_static_graph_irregular_difflength_each(self, args, edges, isTrain = True):
        '''
        every node have different observations
        train observation length [ob_min, ob_max]
        :param args:
        :param edges:
        :param isTrain:
        :param sample_freq:
        :param step_train:
        :param step_test:
        :return:
        '''
        sample_freq = args.sample_freq
        ode_step = args.ode
        max_ob = ode_step//sample_freq

        num_test_box = args.num_test_box
        num_test_extra  = args.num_test_extra

        ob_max = args.ob_max
        ob_min = args.ob_min


        #########Modified sample_trajectory with static graph input, irregular timestamps.

        n = self.n_sticks

        if isTrain:
            T = ode_step
        else:
            T = ode_step * (1 + num_test_box)

        step = T//sample_freq



        counter = 1 #reserve initial point
        # Initialize location and velocity
        loc = np.zeros((step, 1, n))
        vel = np.zeros((step, 1, n))

        angle_in_degrees = 90
        angle_in_radians = np.radians(angle_in_degrees)

        # loc_next = np.random.uniform(0, np.pi / 2, (1, 3)) * self.loc_std
        loc_next = np.full((1, 3), np.pi / 2)
        loc_next = np.mod(loc_next, np.pi)
        print('initial',loc_next)

        p_next = np.zeros((1, 3))
        # p_norm = np.sqrt((p_next ** 2).sum(axis=0)).reshape(1, -1)
        # p_next = p_next * self.vel_norm / p_norm

        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide='ignore'):
            vel_next = self.calculate_angular_speed(loc_next, p_next)


            # run leapfrog
            for i in range(1, T):
                loc_next += self._delta_T * vel_next

                if i % sample_freq ==0:
                    loc[counter, :, :], vel[counter, :, :] = loc_next, vel_next
                    counter += 1

                p_dot = self.calculate_p_dot(loc_next, vel_next)
                p_next += self._delta_T * p_dot
                vel_next = self.calculate_angular_speed(loc_next, p_next)
            # Add noise to observations
            loc += np.random.randn(step, 1, self.n_sticks) * self.noise_var
            vel += np.random.randn(step, 1, self.n_sticks) * self.noise_var

            # sampling

            loc_sample = []
            vel_sample = []
            time_sample = []
            if isTrain:
                for i in range(n):
                    # number of timesteps
                    num_steps = np.random.randint(low = ob_min, high = ob_max +1 , size = 1)[0]
                    # value of timesteps
                    Ts_ball = self.sample_timestamps_with_initial(num_steps,0,max_ob)
                    loc_sample.append(loc[Ts_ball,:,i])
                    vel_sample.append(vel[Ts_ball, :, i])
                    time_sample.append(Ts_ball)

            else:
                for i in range(n):
                    # number of timesteps
                    num_steps = np.random.randint(low = ob_min, high = ob_max, size = 1)[0]
                    # value of timesteps
                    Ts_ball = self.sample_timestamps_with_initial(num_steps,0,max_ob)
                    

                    for j in range(num_test_box):
                        start = max_ob + j*max_ob
                        end  = min(T//sample_freq,max_ob + (j+1)*max_ob)
                        Ts_append = self.sample_timestamps_with_initial(num_test_extra,start,end)
                        Ts_ball = np.append(Ts_ball,Ts_append)

                    loc_sample.append(loc[Ts_ball,:,i])
                    vel_sample.append(vel[Ts_ball, :, i])
                    time_sample.append(Ts_ball)


            return loc_sample, vel_sample, time_sample


    def sample_timestamps_with_initial(self, num_sample, start, end):
        times = set()
        assert(num_sample<=(end-start-1))
        times.add(start)
        while len(times) < num_sample:
            times.add(int(np.random.randint(low=start+1, high=end, size=1)[0]))
        times = np.sort(np.asarray(list(times)))
        return times

    def sample_trajectory(self, T=10000, sample_freq=10,
                          spring_prob=[1. / 2, 0, 1. / 2]):

        n = self.n_sticks
        assert (T % sample_freq == 0)
        T_save = int(T / sample_freq - 1)
        diag_mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(diag_mask, 0)
        counter = 0
        # Def edges
        edges = np.eye(self.n_sticks, k=1) + np.eye(self.n_sticks, k=-1)

        # Initialize location and velocity
        loc = np.zeros((T_save, 1, n))
        vel = np.zeros((T_save, 1, n))

        #
        # loc_next = np.random.uniform(0, np.pi/2, (1, 3)) * self.loc_std
        # loc_next = np.mod(loc_next, np.pi)
        loc_next = np.full((1, 3), np.pi / 2)
        loc_next = np.mod(loc_next, np.pi)
        print('initial loc:', loc_next)
        p_next =np.zeros((1, 3))
        print('initial p: ', p_next)
        loc_dot = np.zeros((1, 3))
        print('initial vel: ', loc_dot )
        initial_energy = sim._energy(loc_next[0], loc_dot[0])
        print('initial_energy:', initial_energy)

        #---------RK4 solver--------------
        with np.errstate(divide='ignore'):
            for i in range(1, T):
                d_loc_1=self.calculate_angular_speed(loc_next, p_next)*self._delta_T
                d_p_1 = self.calculate_p_dot(loc_next, loc_dot)*self._delta_T
                loc_dot_1 = self.calculate_angular_speed(loc_next + 1 / 2 * d_loc_1, p_next + 1 / 2 * d_p_1)

                d_loc_2 =  loc_dot_1 * self._delta_T
                d_p_2 = self.calculate_p_dot(loc_next+1/2*d_loc_1, loc_dot_1) * self._delta_T
                loc_dot_2 = self.calculate_angular_speed(loc_next + 1 / 2 * d_loc_2, p_next + 1 / 2 * d_p_2)

                d_loc_3 = loc_dot_2 * self._delta_T
                d_p_3 = self.calculate_p_dot(loc_next + 1 / 2 * d_loc_2, loc_dot_2) * self._delta_T
                loc_dot_3 = self.calculate_angular_speed(loc_next + d_loc_3, p_next +  d_p_3)

                d_loc_4 = loc_dot_3 * self._delta_T
                d_p_4 = self.calculate_p_dot(loc_next +  d_loc_3, loc_dot_3) * self._delta_T

                d_loc=(1/6)*(d_loc_1+2*d_loc_2+2*d_loc_3+d_loc_4)
                d_p=(1/6)*(d_p_1+2*d_p_2+2*d_p_3+d_p_4)
                loc_next +=d_loc
                p_next +=d_p
                loc_dot=self.calculate_angular_speed(loc_next, p_next)

                if i % sample_freq == 0:
                    loc[counter, :, :], vel[counter, :, :] = loc_next, loc_dot
                    counter += 1

        #-----Leapfrog solver--------
        # disables division by zero warning, since I fix it with fill_diagonal
        # with np.errstate(divide='ignore'):
        #     vel_next = np.zeros((1, 3))
        #     print('initial vel: ', vel_next)
        #     initial_energy = sim._energy(loc_next[0], vel_next[0])
        #     print('initial_energy:', initial_energy)
        #
        #     for i in range(1, T):
        #         p_dot = self.calculate_p_dot(loc_next, vel_next)
        #         p_mid = p_next + 1 / 2 * self._delta_T * p_dot
        #         vel_next = self.calculate_angular_speed(loc_next, p_mid)
        #         loc_next += self._delta_T * vel_next
        #         loc_next = np.mod(loc_next, np.pi)
        #         p_dot = self.calculate_p_dot(loc_next, vel_next)
        #         p_next=p_mid+1 / 2 * self._delta_T * p_dot
        #         vel_next = self.calculate_angular_speed(loc_next, p_next)
        #         if i % sample_freq == 0:
        #             loc[counter, :, :], vel[counter, :, :] = loc_next, vel_next
        #             counter += 1

        # ----------Euler solver-----------
        # with np.errstate(divide='ignore'):
        #     vel_next = self.calculate_angular_speed(loc_next, p_next)
        #     # run leapfrog
        #     for i in range(1, T):
        #         loc_next += self._delta_T * vel_next
        #         loc_next = np.mod(loc_next, np.pi)
        #
        #         if i % sample_freq == 0:
        #             loc[counter, :, :], vel[counter, :, :] = loc_next, vel_next
        #             counter += 1
        #
        #         p_dot = self.calculate_p_dot(loc_next, vel_next)
        #         p_next += self._delta_T * p_dot
        #         print('p_dot: ', p_dot)
        #         print('p_next: ', p_next)
        #         vel_next = self.calculate_angular_speed(loc_next, p_next)

            # Add noise to observations
            loc += np.random.randn(T_save, 1, self.n_sticks) * self.noise_var
            vel += np.random.randn(T_save, 1, self.n_sticks) * self.noise_var
            return loc, vel, edges


if __name__ == '__main__':

    sim = PendulumSim()

    t = time.time()
    loc, vel, edges = sim.sample_trajectory(T=5000, sample_freq=100)
    print("Simulation time: {}".format(time.time() - t))
    vel_norm = np.sqrt((vel ** 2).sum(axis=1))
    plt.figure()
    axes = plt.gca()
    # axes.set_xlim([-5., 5.])
    # axes.set_ylim([-5., 5.])
    for i in range(loc.shape[-1]):
        theta= loc[:, 0, i]
        r=1
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        # print(x)
        # print(y)

        plt.plot(x,y)
        plt.plot(x[0], y[0], 'd')
    plt.figure()
    energies = [sim._energy(loc[i][0], vel[i][0]) for i in
                range(loc.shape[0])]
    plt.plot(energies)
    plt.show()
