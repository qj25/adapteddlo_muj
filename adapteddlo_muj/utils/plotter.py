import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams.update({'pdf.fonttype': 42})   # to prevent type 3 fonts in pdflatex
img_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data/figs/"
)

def plot3d(nodes,nodes2=None):
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the points as a scatter plot
    ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], c='r', marker='o')
    
    # Optionally, connect the nodes with lines (to visualize the path)
    ax.plot(nodes[:, 0], nodes[:, 1], nodes[:, 2], c='b')
    if nodes2 is not None:
        nodes2 = np.array(nodes2)
        if nodes2.ndim < 3:
            ax.scatter(nodes2[:, 0], nodes2[:, 1], nodes2[:, 2], c='y', marker='o')
            ax.plot(nodes2[:, 0], nodes2[:, 1], nodes2[:, 2], c='g')
        else:
            for i in range(len(nodes2)):
                ax.scatter(nodes2[i,:, 0], nodes2[i,:, 1], nodes2[i,:, 2], c='y', marker='o')
                ax.plot(nodes2[i,:, 0], nodes2[i,:, 1], nodes2[i,:, 2], c='g',alpha=(i+1)/len(nodes2))

    # Labels for clarity
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    
    # Set title
    ax.set_title('3D Plot of Node Points')

    # Ensure equal scaling of the axes by setting the limits
    max_range = np.array([nodes[:, 0].max() - nodes[:, 0].min(),
                        nodes[:, 1].max() - nodes[:, 1].min(),
                        nodes[:, 2].max() - nodes[:, 2].min()]).max()
    
    # Centering the plot (so that the axes are symmetric)
    mid_x = (nodes[:, 0].max() + nodes[:, 0].min()) / 2
    mid_y = (nodes[:, 1].max() + nodes[:, 1].min()) / 2
    mid_z = (nodes[:, 2].max() + nodes[:, 2].min()) / 2
    
    # Setting the limits of all axes based on the max range
    ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
    ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
    ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)
    
    # Show the plot
    plt.tight_layout()
    plt.show()

def plot_bars(error):
    index = np.arange(len(error[0,0]))*2
    bar_width = 0.2
    # colors_model1 = ['lightblue', 'lightgreen', 'lightcoral']
    rope_positions = [1,2,3,4]
    wire_colors = ['grey','black','red']
    model_types = ['adapted', 'native']
    # model_edgecolors = ['lightgreen', 'lightpink']
    model_hatch = [None, '\\\\\\']

    bar_adapt = []
    bar_native = []

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, rope in enumerate(wire_colors):
        bar_adapt.append(ax.bar(index + (2*i * bar_width), error[0,i,], bar_width-0.04,
            color=wire_colors[i], capsize=5, alpha=0.7, hatch=model_hatch[0]))
        bar_native.append(ax.bar(index + (2*i * bar_width) + bar_width, error[1,i,], bar_width,
            color=wire_colors[i], capsize=5, alpha=0.7, hatch=model_hatch[1]))
        ax.bar(index + (2*i * bar_width), np.zeros_like(error[0,i,]), bar_width-0.04,
            label=f'{rope}', color=wire_colors[i], capsize=5, alpha=0.7)
        # for j in range(len(rope_positions)):
        #     # bar_adapt[-1][j].set_color('lightgreen')
        #     # bar_adapt[-1][j].set_edgecolor(wire_colors[i])
        #     bar_adapt[-1][j].set_edgecolor(model_edgecolors[0])
        #     bar_adapt[-1][j].set_linewidth(5)
        #     # bar_native[-1][j].set_color('lightpink')
        #     # bar_native[-1][j].set_edgecolor(wire_colors[i])
        #     bar_native[-1][j].set_edgecolor(model_edgecolors[1])
        #     bar_native[-1][j].set_linewidth(5)

    for i in range(len(model_types)):
        t1 = ax.bar(index + (2*i * bar_width), np.zeros_like(error[0,i,]), bar_width-0.04,
            label=f'{model_types[i]}', color='white', capsize=5, alpha=0.7, hatch=model_hatch[i], zorder = 10)
        t1[0].set_edgecolor('black')
        # t1[0].set_linewidth(5)

    ax.set_xlabel('Robot Pose',fontsize=14)
    ax.set_ylabel('Normalized Position Error',fontsize=14)
    # ax.set_title('Comparison of Simulation Models and Real-world Data with Error Bars')
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(rope_positions)
    # ax.legend(title='')
    # Remove the top and right borders (spines)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed')
    plt.tick_params(axis='both', which='major', labelsize=14)
    
    # Position the legend at the center top
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2, fontsize=14)
    plt.tight_layout()
    plt.savefig(img_path + "valid_bars.pdf",bbox_inches='tight')
    plt.show()

def plot_computetime(pieces_list, data_list):
    plt.style.use('seaborn-v0_8')
    # Sample data for 4 different results
    x = np.array(pieces_list)  # Number of discrete pieces simulated (from 1 to 100)
    y = np.array(data_list)

    # compute percentage difference
    y_percent = (y[:] - y[0]) / y[0] * 100.0

    # Create the plot
    fig, twin1 = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(right=0.75)
    ax = twin1.twinx()
    twin1.set_facecolor('white')
    twin1.grid(color='#DDDDDD', linewidth=0.8, zorder=0)
    ax.grid(False)
    ax.yaxis.tick_left()
    ax.yaxis.set_label_position("left")
    twin1.yaxis.tick_right()
    twin1.yaxis.set_label_position("right")

    ax.spines['top'].set_visible(False)
    twin1.spines['top'].set_visible(False)
    for spine_str in [
        'bottom',
        'left',
        'right'    
    ]:
        ax.spines[spine_str].set_linewidth(1)
        ax.spines[spine_str].set_edgecolor('k')

    # Plot each line with a label
    ax.plot(x, y[0], label='plain', alpha=0.7, color='k', linewidth=2,zorder=3)
    ax.plot(x, y[3], label='adapted', alpha=0.7, linewidth=2)
    ax.plot(x, y[2], label='direct', alpha=0.7, linewidth=2)
    ax.plot(x, y[1], label='native', alpha=0.7, linewidth=2)
    twin1.plot(0,0, label='raw time', alpha=1.0, color='k', linewidth=2)
    twin1.plot(0,0, label='percent increase', alpha=1.0, color='k', linewidth=2, linestyle='--')
    # twin1.plot(x, y_percent[0], alpha=0.7, color='k', linewidth=2,linestyle='--')
    twin1.plot(x, y_percent[3], alpha=0.7, linewidth=2,linestyle='--')
    twin1.plot(x, y_percent[2], alpha=0.7, linewidth=2,linestyle='--')
    twin1.plot(x, y_percent[1], alpha=0.7, linewidth=2,linestyle='--')

    ax.set_ylim(0, 50)
    ax.set_xlim(40, 180)
    twin1.set_ylim(0, 15)
    ax.set_yticks(np.linspace(0., 50., 6))
    ax.set_xticks(x)
    twin1.set_yticks(np.linspace(0., 15., 6))
    tkw = dict(size=4, width=1.5)
    ax.tick_params(axis='y', **tkw)
    twin1.tick_params(axis='y', **tkw)

    # Add labels and title
    ax.set_ylabel('Computation Time to Simulate 1s (seconds)', fontsize=14)
    twin1.set_xlabel('Number of Discrete Pieces', fontsize=14)
    twin1.set_ylabel("Percentage Increase from plain", fontsize=14)
    # plt.title('Speed test', fontsize=14)

    # Add a legend
    ax.legend(fontsize=14, loc='upper left', bbox_to_anchor=(0.21, 0.98),ncol=1)
    twin1.legend(fontsize=14, loc='upper left', bbox_to_anchor=(0.39, 0.98),ncol=1)

    # Grid for better readability
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.tick_params(axis='both', which='major', labelsize=12)
    twin1.tick_params(axis='both', which='major', labelsize=12)
    # # Set x and y axis limits for better visualization
    # plt.xlim([0, 100])
    # plt.ylim([0, 120])

    plt.tight_layout()
    # Save the plot if needed
    plt.savefig(img_path + "compute_time.pdf",bbox_inches='tight')

    # Show the plot
    plt.show()