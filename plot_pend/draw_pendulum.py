import numpy as np
import matplotlib.pyplot as plt
import os
from synthetic_sim_pendulum import PendulumSim
import pickle

paint_res = 300  # TODO change to 300 when publish
label_font = 24
markersize = 18
tick_font = 24
line_width = 3
markers = ['o', 's', 'v', 'D', 'h', 'H', 'd', '*', 'p', 'P', 'X', 'x', '+', '|', '_', '1', '2', '3', '4', '8', 's', 'p', '*', 'h', 'H', 'd', 'D', 'v', '^', '<', '>']
colors = ["#0072BD", "#D95319", "#EDB120", "#7E2F8E", "#77AC30", "#4DBEEE", "#A2142F", "#000000"]
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["legend.frameon"] = False
plt.rcParams['figure.dpi'] = paint_res
plt.rcParams.update({'figure.autolayout': True})


def gen_trajtory(dir, initial_thetas=np.full((1, 3), np.pi / 2), T=32000, sample_freq=40):
    sim = PendulumSim()
    loc, vel, loc_theta, vel_theta, edges = sim.sample_trajectory(T=T, sample_freq=sample_freq, initial_thetas=initial_thetas)
    # dump to a folder traj/initial_thetas_T_sample_freq
    # create dir if necessary
    cache_dir = os.path.join(dir, f'traj_{initial_thetas[0, 0]}_{initial_thetas[0, 1]}_{initial_thetas[0, 2]}_{T}_{sample_freq}')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    # use pickle to save all loc, vel, loc_theta, vel_theta, edges
    pickle.dump([loc, vel, loc_theta, vel_theta, edges], open(os.path.join(cache_dir, 'data.pkl'), 'wb'))


def plot_trajtory_full(dir, initial_thetas=np.full((1, 3), np.pi / 2), T=32000, sample_freq=40):
    # now uses the initial_thetas to simulate a traj in the same way as the synthetic sim
    fig, ax = plt.subplots(1, 1, figsize=(18, 12))

    # read from the folder traj/initial_thetas_T_sample_freq/data.pkl
    # the loaded data is [loc, vel, loc_theta, vel_theta, edges]
    cache_dir = os.path.join(dir, f'traj_{initial_thetas[0, 0]}_{initial_thetas[0, 1]}_{initial_thetas[0, 2]}_{T}_{sample_freq}')
    loc, vel, loc_theta, vel_theta, edges = pickle.load(open(os.path.join(dir, cache_dir, 'data.pkl'), 'rb'))

    for t in range(loc.shape[0]):
        # clear the plot
        ax.cla()
        plots = []
        legends = []

        # end point pos is [0,0]
        end_pos = np.array([0, 0])
        plot_end, = ax.plot(end_pos[0], end_pos[1], marker='o', markersize=markersize, markevery=500, fillstyle='full', linewidth=line_width, color='k', zorder=10)
        plots.append(plot_end)
        legends.append('Joint Locations')

        for joint_idx in range(loc.shape[-1]):
            ball_data = loc[t, :, joint_idx]
            ax.plot(ball_data[0], ball_data[1], marker='o', markersize=markersize, markevery=500, fillstyle='full', linewidth=line_width, color='k', zorder=10)

        # plot the rigid rod (thick line) between each ball
        # for 0, its the end point and ball 1
        for joint_idx in range(loc.shape[-1]):
            ball_data = loc[t, :, joint_idx]
            prev_ball_data = loc[t, :, joint_idx - 1] if joint_idx > 0 else end_pos
            plot_rod, = ax.plot([prev_ball_data[0], ball_data[0]], [prev_ball_data[1], ball_data[1]], linewidth=line_width * 4, color=colors[joint_idx], zorder=1)
            plots.append(plot_rod)
            legends.append(f'Rod {joint_idx + 1}')

        ax.legend(plots, legends, loc='upper center', fontsize=label_font, ncol=len(legends))
        ax.tick_params(top=False, bottom=True, left=True, right=False, labelleft=True, labelsize=tick_font)
        ax.xaxis.offsetText.set_fontsize(label_font)
        ax.set_xlabel(r'X [m]', fontsize=label_font)
        ax.set_ylabel(r'Y [m]', fontsize=label_font)
        # plot in [3x3] box
        # # set the x lenght to be the same as y length
        plt.xlim(-3.25, 3.25)
        plt.ylim(-3, 1.0)
        ax.grid(True, linestyle='--', linewidth=1.5)

        # dump to the same cache dir
        plt.savefig(os.path.join(dir, cache_dir, f'frame{t}.png'), transparent=False, dpi=paint_res, bbox_inches="tight")


def plot_trajtory_compare(dir, initial_thetas1=np.full((1, 3), np.pi / 2), initial_thetas2=np.full((1, 3), np.pi / 2), initial_thetas3=np.full((1, 3), np.pi / 2), T=32000, sample_freq=40):
    # now uses the initial_thetas to simulate a traj in the same way as the synthetic sim
    fig, ax = plt.subplots(1, 1, figsize=(18, 12))

    # read from the folder traj/initial_thetas_T_sample_freq/data.pkl
    # the loaded data is [loc, vel, loc_theta, vel_theta, edges]
    cache_dir1 = os.path.join(dir, f'traj_{initial_thetas1[0, 0]}_{initial_thetas1[0, 1]}_{initial_thetas1[0, 2]}_{T}_{sample_freq}')
    loc1, vel1, loc_theta1, vel_theta1, edges1 = pickle.load(open(os.path.join(dir, cache_dir1, 'data.pkl'), 'rb'))
    cache_dir2 = os.path.join(dir, f'traj_{initial_thetas2[0, 0]}_{initial_thetas2[0, 1]}_{initial_thetas2[0, 2]}_{T}_{sample_freq}')
    loc2, vel2, loc_theta2, vel_theta2, edges2 = pickle.load(open(os.path.join(dir, cache_dir2, 'data.pkl'), 'rb'))
    cache_dir3 = os.path.join(dir, f'traj_{initial_thetas3[0, 0]}_{initial_thetas3[0, 1]}_{initial_thetas3[0, 2]}_{T}_{sample_freq}')
    loc3, vel3, loc_theta3, vel_theta3, edges3 = pickle.load(open(os.path.join(dir, cache_dir3, 'data.pkl'), 'rb'))

    for t in range(loc1.shape[0]):
        # clear the plot
        ax.cla()
        plots = []
        legends = []

        # end point pos is [0,0]
        end_pos = np.array([0, 0])
        plot_end, = ax.plot(end_pos[0], end_pos[1], marker='o', markersize=markersize, markevery=500, fillstyle='full', linewidth=line_width, color='k', zorder=10)
        plots.append(plot_end)
        legends.append('Joint Locations')

        # plot the first traj in solid line

        # plot the joints
        for joint_idx in range(loc1.shape[-1]):
            ball_data = loc1[t, :, joint_idx]
            ax.plot(ball_data[0], ball_data[1], marker='o', markersize=markersize, markevery=500, fillstyle='full', linewidth=line_width, color='k', zorder=10)

        # plot the rigid rod (thick line) between each ball
        # for 0, its the end point and ball 1
        for joint_idx in range(loc1.shape[-1]):
            ball_data = loc1[t, :, joint_idx]
            prev_ball_data = loc1[t, :, joint_idx - 1] if joint_idx > 0 else end_pos
            plot_rod, = ax.plot([prev_ball_data[0], ball_data[0]], [prev_ball_data[1], ball_data[1]], linewidth=line_width * 4, color=colors[joint_idx], zorder=9)
            if joint_idx == 0:
                plots.append(plot_rod)
                legends.append('Original initial condition')

        # plot the second traj in dashed line

        # plot the joints
        for joint_idx in range(loc2.shape[-1]):
            ball_data = loc2[t, :, joint_idx]
            ax.plot(ball_data[0], ball_data[1], marker='o', markersize=markersize, markevery=500, fillstyle='left', linewidth=line_width, color='k', zorder=8)

        # plot the rigid rod (thick line) between each ball
        # for 0, its the end point and ball 1
        for joint_idx in range(loc2.shape[-1]):
            ball_data = loc2[t, :, joint_idx]
            prev_ball_data = loc2[t, :, joint_idx - 1] if joint_idx > 0 else end_pos
            plot_rod, = ax.plot([prev_ball_data[0], ball_data[0]], [prev_ball_data[1], ball_data[1]], linewidth=line_width * 2, color=colors[joint_idx], zorder=7, linestyle='--')
            if joint_idx == 0:
                plots.append(plot_rod)
                legends.append('w/ 1e-3 perturbation')

        # plot the third traj in dotdash line

        # plot the joints
        for joint_idx in range(loc3.shape[-1]):
            ball_data = loc3[t, :, joint_idx]
            ax.plot(ball_data[0], ball_data[1], marker='o', markersize=markersize, markevery=500, fillstyle='none', linewidth=line_width, color='k', zorder=6)

        # plot the rigid rod (thick line) between each ball
        # for 0, its the end point and ball 1
        for joint_idx in range(loc3.shape[-1]):
            ball_data = loc3[t, :, joint_idx]
            prev_ball_data = loc3[t, :, joint_idx - 1] if joint_idx > 0 else end_pos
            plot_rod, = ax.plot([prev_ball_data[0], ball_data[0]], [prev_ball_data[1], ball_data[1]], linewidth=line_width * 2, color=colors[joint_idx], zorder=5, linestyle='dotted')
            if joint_idx == 0:
                plots.append(plot_rod)
                legends.append('w/ 1e-2 perturbation')

        ax.legend(plots, legends, loc='upper center', fontsize=label_font, ncol=len(legends) // 2)
        ax.tick_params(top=False, bottom=True, left=True, right=False, labelleft=True, labelsize=tick_font)
        ax.xaxis.offsetText.set_fontsize(label_font)
        ax.set_xlabel(r'X [m]', fontsize=label_font)
        ax.set_ylabel(r'Y [m]', fontsize=label_font)
        # plot in [3x3] box
        # # set the x lenght to be the same as y length
        plt.xlim(-3.25, 3.25)
        plt.ylim(-3, 1.0)
        ax.grid(True, linestyle='--', linewidth=1.5)

        # the compare cache dir is traj_thetas1_thetas2_thetas_T_sample_freq
        cache_dir = os.path.join(
            dir,
            f'compare_traj_{initial_thetas1[0, 0]}_{initial_thetas1[0, 1]}_{initial_thetas1[0, 2]}_{initial_thetas2[0, 0]}_{initial_thetas2[0, 1]}_{initial_thetas2[0, 2]}_{initial_thetas3[0, 0]}_{initial_thetas3[0, 1]}_{initial_thetas3[0, 2]}_{T}_{sample_freq}'
        )
        # create dir if necessary
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        # dump to the same cache dir
        plt.savefig(os.path.join(dir, cache_dir, f'frame{t}.png'), transparent=False, dpi=paint_res, bbox_inches="tight")


def plot_theta_vel_compare(dir, initial_thetas1=np.full((1, 3), np.pi / 2), initial_thetas2=np.full((1, 3), np.pi / 2), initial_thetas3=np.full((1, 3), np.pi / 2), T=32000, sample_freq=40):
    # now uses the initial_thetas to simulate a traj in the same way as the synthetic sim
    fig, ax = plt.subplots(1, 1, figsize=(18, 12))

    # read from the folder traj/initial_thetas_T_sample_freq/data.pkl
    # the loaded data is [loc, vel, loc_theta, vel_theta, edges]
    cache_dir1 = os.path.join(dir, f'traj_{initial_thetas1[0, 0]}_{initial_thetas1[0, 1]}_{initial_thetas1[0, 2]}_{T}_{sample_freq}')
    loc1, vel1, loc_theta1, vel_theta1, edges1 = pickle.load(open(os.path.join(dir, cache_dir1, 'data.pkl'), 'rb'))
    cache_dir2 = os.path.join(dir, f'traj_{initial_thetas2[0, 0]}_{initial_thetas2[0, 1]}_{initial_thetas2[0, 2]}_{T}_{sample_freq}')
    loc2, vel2, loc_theta2, vel_theta2, edges2 = pickle.load(open(os.path.join(dir, cache_dir2, 'data.pkl'), 'rb'))
    cache_dir3 = os.path.join(dir, f'traj_{initial_thetas3[0, 0]}_{initial_thetas3[0, 1]}_{initial_thetas3[0, 2]}_{T}_{sample_freq}')
    loc3, vel3, loc_theta3, vel_theta3, edges3 = pickle.load(open(os.path.join(dir, cache_dir3, 'data.pkl'), 'rb'))

    min_theta1 = np.min(loc_theta1[:, 0, :])
    max_theta1 = np.max(loc_theta1[:, 0, :])
    min_theta2 = np.min(loc_theta2[:, 0, :])
    max_theta2 = np.max(loc_theta2[:, 0, :])
    min_theta3 = np.min(loc_theta3[:, 0, :])
    max_theta3 = np.max(loc_theta3[:, 0, :])

    min_theta = min(min_theta1, min_theta2, min_theta3)
    max_theta = max(max_theta1, max_theta2, max_theta3)

    for t in range(loc1.shape[0] - 1, loc1.shape[0]):
        # clear the plot
        ax.cla()
        plots = []
        legends = []

        # plot the last joint theta log
        lines = ['-', '--', '-.']
        for joint_idx in range(loc1.shape[-1]):
            # joint_idx = -1
            # print(loc_theta1.shape)
            theta1 = loc_theta1[:t + 1, 0, joint_idx]
            theta2 = loc_theta2[:t + 1, 0, joint_idx]
            theta3 = loc_theta3[:t + 1, 0, joint_idx]

            frames = np.arange(t + 1)

            plot1, = ax.plot(frames, theta1, marker=markers[joint_idx * 3 + 0], markersize=markersize, markevery=50, fillstyle='none', linewidth=line_width, linestyle='-', color=colors[joint_idx])
            plot2, = ax.plot(frames, theta2, marker=markers[joint_idx * 3 + 1], markersize=markersize, markevery=50, fillstyle='none', linewidth=line_width, linestyle='--', color=colors[joint_idx])
            plot3, = ax.plot(frames, theta3, marker=markers[joint_idx * 3 + 2], markersize=markersize, markevery=50, fillstyle='none', linewidth=line_width, linestyle='-.', color=colors[joint_idx])

            if joint_idx == 0:
                legends.append(r'Original initial condition: ' + rf'$\theta_{joint_idx}$')
                legends.append(r'w/ 1e-3 perturbation: ' + rf'$\theta_{joint_idx}$')
                legends.append(r'w/ 1e-2 perturbation: ' + rf'$\theta_{joint_idx}$')
            else:
                legends.append(rf'$\theta_{joint_idx}$')
                legends.append(rf'$\theta_{joint_idx}$')
                legends.append(rf'$\theta_{joint_idx}$')

            plots.append(plot1)
            plots.append(plot2)
            plots.append(plot3)

            ax.legend(plots, legends, loc='upper center', fontsize=label_font, ncol=3)
            ax.tick_params(top=False, bottom=True, left=True, right=False, labelleft=True, labelsize=tick_font)
            ax.get_xaxis().set_visible(True)
            ax.set_xlabel(r'Time steps', fontsize=label_font)
            ax.set_ylabel(r'Joint ' + r'$\theta$', fontsize=label_font)
            ax.set_xlim([0, loc1.shape[0]])
            ax.set_ylim([min_theta, max_theta])
            ax.grid(True, linestyle='--', linewidth=1.5)

            # the compare cache dir is traj_thetas1_thetas2_thetas_T_sample_freq
            cache_dir = os.path.join(
                dir,
                f'compare_traj_{initial_thetas1[0, 0]}_{initial_thetas1[0, 1]}_{initial_thetas1[0, 2]}_{initial_thetas2[0, 0]}_{initial_thetas2[0, 1]}_{initial_thetas2[0, 2]}_{initial_thetas3[0, 0]}_{initial_thetas3[0, 1]}_{initial_thetas3[0, 2]}_{T}_{sample_freq}'
            )
            # create dir if necessary
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)

            plt.savefig(os.path.join(dir, cache_dir, f'compare_theta3_{t}.png'), transparent=False, dpi=300, bbox_inches="tight")


def plot_trajtory_learned(dir, model_name, traj_idx=0):
    # now uses the initial_thetas to simulate a traj in the same way as the synthetic sim
    fig, ax = plt.subplots(1, 1, figsize=(18, 12))

    # read from the folder dir/pendulum/pendulum_{model_name}
    # under this folder there are forward_trajectory.npy and groundtruth_trajectory.npy
    folder_name = os.path.join(dir, 'pendulum', f'pendulum_{model_name}')
    forward_traj = np.load(os.path.join(folder_name, 'forward_trajectory.npy'))
    groundtruth_traj = np.load(os.path.join(folder_name, 'groundtruth_trajectory.npy'))

    # read normalizer from dir/pendulum/normalizer.json
    import json
    with open(os.path.join(dir, 'pendulum', 'normalizer.json'), 'r') as f:
        normalizer = json.load(f)

    min_loc = normalizer['min_loc']
    max_loc = normalizer['max_loc']
    min_vel = normalizer['min_vel']
    max_vel = normalizer['max_vel']

    # reshape to [-1,3,60,2,2]
    forward_traj = forward_traj.reshape(-1, 3, 60, 4)
    groundtruth_traj = groundtruth_traj.reshape(-1, 3, 60, 4)
    # print(forward_traj.shape)
    # print(groundtruth_traj.shape)
    # exit()

    # choose traj_idx at 0th dim
    forward_traj = forward_traj[traj_idx]
    groundtruth_traj = groundtruth_traj[traj_idx]

    # de-normalize
    min_vec=np.array([min_loc, min_loc, min_vel, min_vel])
    max_vec=np.array([max_loc, max_loc, max_vel, max_vel])
    forward_traj = (forward_traj + 1) / 2 * (max_vec - min_vec) + min_vec
    groundtruth_traj = (groundtruth_traj + 1) / 2 * (max_vec - min_vec) + min_vec

    for t in range(forward_traj.shape[1]):
        # clear the plot
        ax.cla()
        plots = []
        legends = []

        # end point pos is [0,0]
        end_pos = np.array([0, 0])
        plot_end, = ax.plot(end_pos[0], end_pos[1], marker='o', markersize=markersize, markevery=500, fillstyle='full', linewidth=line_width, color='k', zorder=10)
        plots.append(plot_end)
        legends.append('Joint Locations')

        # plot the first traj in solid line

        # plot the joints
        for joint_idx in range(forward_traj.shape[0]):
            ball_data = forward_traj[joint_idx, t, :2]
            # # de-normalize
            # ball_data = (ball_data + 1) / 2 * (max_loc - min_loc) + min_loc
            ax.plot(ball_data[0], ball_data[1], marker='o', markersize=markersize, markevery=500, fillstyle='full', linewidth=line_width, color='k', zorder=10)

        # plot the rigid rod (thick line) between each ball
        # for 0, its the end point and ball 1
        for joint_idx in range(forward_traj.shape[0]):
            ball_data = forward_traj[joint_idx, t, :2]
            # # de-normalize
            # ball_data = (ball_data + 1) / 2 * (max_loc - min_loc) + min_loc
            prev_ball_data = forward_traj[joint_idx - 1, t, :2] if joint_idx > 0 else end_pos
            # if joint_idx > 0:
            #     # de-normalize
            #     prev_ball_data = (prev_ball_data + 1) / 2 * (max_loc - min_loc) + min_loc
            plot_rod, = ax.plot([prev_ball_data[0], ball_data[0]], [prev_ball_data[1], ball_data[1]], linewidth=line_width * 4, color=colors[joint_idx], zorder=9)
            if joint_idx == 0:
                plots.append(plot_rod)
                legends.append('Learned model')

        # plot the second traj in dashed line

        # plot the joints
        for joint_idx in range(groundtruth_traj.shape[0]):
            ball_data = groundtruth_traj[joint_idx, t, :2]
            # # de-normalize
            # ball_data = (ball_data + 1) / 2 * (max_loc - min_loc) + min_loc
            ax.plot(ball_data[0], ball_data[1], marker='o', markersize=markersize, markevery=500, fillstyle='left', linewidth=line_width, color='k', zorder=8)

        # plot the rigid rod (thick line) between each ball
        # for 0, its the end point and ball 1
        for joint_idx in range(groundtruth_traj.shape[0]):
            ball_data = groundtruth_traj[joint_idx, t, :2]
            # # de-normalize
            # ball_data = (ball_data + 1) / 2 * (max_loc - min_loc) + min_loc
            prev_ball_data = groundtruth_traj[joint_idx - 1, t, :2] if joint_idx > 0 else end_pos
            # if joint_idx > 0:
            #     # de-normalize
            #     prev_ball_data = (prev_ball_data + 1) / 2 * (max_loc - min_loc) + min_loc
            plot_rod, = ax.plot([prev_ball_data[0], ball_data[0]], [prev_ball_data[1], ball_data[1]], linewidth=line_width * 2, color=colors[joint_idx], zorder=7, linestyle='--')
            if joint_idx == 0:
                plots.append(plot_rod)
                legends.append('GT')

        ax.legend(plots, legends, loc='upper center', fontsize=label_font, ncol=len(legends))
        ax.tick_params(top=False, bottom=True, left=True, right=False, labelleft=True, labelsize=tick_font)
        ax.xaxis.offsetText.set_fontsize(label_font)
        ax.set_xlabel(r'X [m]', fontsize=label_font)
        ax.set_ylabel(r'Y [m]', fontsize=label_font)
        # plot in [3x3] box
        # # set the x lenght to be the same as y length
        plt.xlim(-3.25, 3.25)
        plt.ylim(-3, 1.0)
        ax.grid(True, linestyle='--', linewidth=1.5)

        # the compare cache dir is traj_thetas1_thetas2_thetas_T_sample_freq
        cache_dir = os.path.join(dir, f'learned_model_{model_name}_traj_{traj_idx}')
        # create dir if necessary
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        # dump to the same cache dir
        plt.savefig(os.path.join(dir, cache_dir, f'frame{t}.png'), transparent=False, dpi=paint_res, bbox_inches="tight")


if __name__ == '__main__':
    eps = 1e-3
    theta1 = np.full((1, 3), np.pi / 2)
    theta2 = np.array([[np.pi / 2 + eps, np.pi / 2 - eps, np.pi / 2 + eps]])
    theta3 = np.array([[np.pi / 2 + 10 * eps, np.pi / 2 - 10 * eps, np.pi / 2 - 10 * eps]])

    ### sample some trajtories
    # gen_trajtory('.', initial_thetas=theta1)
    # gen_trajtory('.', initial_thetas=theta2)
    # gen_trajtory('.', initial_thetas=theta3)

    ### plot the traj without perturbation
    # plot_trajtory_full('.', initial_thetas=theta1)

    ### plot the traj comparisons with perturbation
    # plot_trajtory_compare('.', initial_thetas1=theta1, initial_thetas2=theta2, initial_thetas3=theta3)
    # plot_theta_vel_compare('.', initial_thetas1=theta1, initial_thetas2=theta2, initial_thetas3=theta3)

    ### plot the learned results
    plot_trajtory_learned('.', '60_DCODE_ob0.40_rflambda100.00')
    # plot_trajtory_learned('.', '60_Ham_ob0.40')
    # plot_trajtory_learned('.', '60_LGODE_ob0.40_rflambda0.00')

    pass
