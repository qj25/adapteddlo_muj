import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
    plt.show()

def plot_bars(error):
    index = np.arange(len(error[0,0]))*2
    bar_width = 0.2
    # colors_model1 = ['lightblue', 'lightgreen', 'lightcoral']
    rope_positions = [1,2,3,4]
    wire_colors = ['grey','black','red']
    model_types = ['adapt', 'native']
    model_edgecolors = ['lightgreen', 'lightpink']
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
            label=f'{model_types[i]}', color='white', capsize=5, alpha=0.7, hatch=model_hatch[i])
        t1[0].set_edgecolor('black')
        # t1[0].set_linewidth(5)

    ax.set_xlabel('Robot Pose')
    ax.set_ylabel('Position Difference (Simulated - Real)')
    # ax.set_title('Comparison of Simulation Models and Real-world Data with Error Bars')
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(rope_positions)
    # ax.legend(title='')
    # Remove the top and right borders (spines)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Position the legend at the center top
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2)

    plt.tight_layout()
    plt.show()