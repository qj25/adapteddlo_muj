import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

plt.rcParams.update({'pdf.fonttype': 42})   # to prevent type 3 fonts in pdflatex

# Helper to plot a frame at a point with custom labels and small arrow heads
def plot_frame(ax, origin, t, m1, m2, length=0.5, label_prefix='', color='#003366', label_color='#003366', offsets=None, arrow_length_ratio=0.08, fontweight='bold', label_names=None, superscript=''):
    # Plot arrows
    ax.quiver(*origin, *(t*length), color=color, arrow_length_ratio=arrow_length_ratio)
    ax.quiver(*origin, *(m1*length), color=color, arrow_length_ratio=arrow_length_ratio)
    ax.quiver(*origin, *(m2*length), color=color, arrow_length_ratio=arrow_length_ratio)
    # Offset label positions for visibility (now supports 3D vector offsets)
    if offsets is None:
        offsets = {'t': np.zeros(3), 'm1': np.zeros(3), 'm2': np.zeros(3)}
    if label_names is None:
        label_names = ['t', 'm_1', 'm_2']
    def offset_label(vec, offset):
        norm_vec = vec / np.linalg.norm(vec)
        return origin + vec*length + offset
    # Use LaTeX for bold main symbol and non-bold superscript
    ax.text(*offset_label(t, offsets['t']), fr'$\mathbf{{{label_prefix}{label_names[0]}}}^{{\mathrm{{{superscript}}}}}$', color=label_color, fontsize=10, fontweight=fontweight)
    ax.text(*offset_label(m1, offsets['m1']), fr'$\mathbf{{{label_prefix}{label_names[1]}}}^{{\mathrm{{{superscript}}}}}$', color=label_color, fontsize=10, fontweight=fontweight)
    ax.text(*offset_label(m2, offsets['m2']), fr'$\mathbf{{{label_prefix}{label_names[2]}}}^{{\mathrm{{{superscript}}}}}$', color=label_color, fontsize=10, fontweight=fontweight)

# Rotation matrix about axis by theta (Rodrigues' formula)
def rotate_about_axis(v, axis, theta):
    axis = axis / np.linalg.norm(axis)
    v = v / np.linalg.norm(v)
    return (v * np.cos(theta) +
            np.cross(axis, v) * np.sin(theta) +
            axis * np.dot(axis, v) * (1 - np.cos(theta)))

# Helper to draw an arc in 3D between two vectors at a point
def plot_arc(ax, center, v1, v2, radius=0.35, n_points=50, color='k', lw=2, label=None, label_offset=np.zeros(3)):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    n = np.cross(v1, v2)
    n = n / np.linalg.norm(n)
    angle = np.arccos(np.clip(np.dot(v1, v2), -1, 1))
    arc_points = []
    for t in np.linspace(0, angle, n_points):
        vec = (v1 * np.cos(t) +
               np.cross(n, v1) * np.sin(t) +
               n * np.dot(n, v1) * (1 - np.cos(t)))
        arc_points.append(center + radius * vec)
    arc_points = np.array(arc_points)
    ax.plot(arc_points[:,0], arc_points[:,1], arc_points[:,2], color=color, lw=lw, linestyle='-')
    if label:
        mid_idx = len(arc_points) // 2
        label_pos = arc_points[mid_idx] + label_offset
        ax.text(*label_pos, label, fontsize=13, color=color, fontweight='bold')

# Define three points: x_{i-1}, x_i, x_{i+1} (equal length segments)
L = 1.0
arc_radius = 0.2 * L  # <--- You can set this to control the arc's radius
x_im1 = np.array([0, 0, 0], dtype=float)
x_i   = np.array([0, L, 0], dtype=float)
# Make the second segment bent the other way (negative y)
angle = 0*np.pi / 2  # 45 degrees
x_ip1 = x_i + L * np.array([np.cos(angle), -np.sin(angle), 0], dtype=float)

# Compute tangent vectors for each segment
t1 = (x_i - x_im1)
t1 /= np.linalg.norm(t1)
t2 = (x_ip1 - x_i)
t2 /= np.linalg.norm(t2)

# Compute normal and binormal for each segment
# For the first segment, pick a normal not parallel to t1
temp1 = np.array([0, 0, 1]) if abs(t1[2]) < 0.9 else np.array([0, 1, 0])
m1_1 = np.cross(t1, temp1)
m1_1 /= np.linalg.norm(m1_1)
m2_1 = np.cross(t1, m1_1)
m2_1 /= np.linalg.norm(m2_1)

# For the second segment, pick a normal not parallel to t2
temp2 = np.array([0, 0, 1]) if abs(t2[2]) < 0.9 else np.array([0, 1, 0])
m1_2 = np.cross(t2, temp2)
m1_2 /= np.linalg.norm(m1_2)
m2_2 = np.cross(t2, m1_2)
m2_2 /= np.linalg.norm(m2_2)

# Rotate the frames 180 degrees about the tangent (flip m1 and m2)
m1_1 = -m1_1
m2_1 = -m2_1
m1_2 = -m1_2
m2_2 = -m2_2

# Compute midpoints for each segment
mid1 = (x_im1 + x_i) / 2
mid2 = (x_i + x_ip1) / 2

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=25, azim=-45)
ax.grid(False)
ax.set_axis_off()

# Plot the lines as dotted lines
ax.plot([x_im1[0], x_i[0]], [x_im1[1], x_i[1]], [x_im1[2], x_i[2]], 'k:', lw=2)
ax.plot([x_i[0], x_ip1[0]], [x_i[1], x_ip1[1]], [x_i[2], x_ip1[2]], 'k:', lw=2)

# Mark the start and end points with solid spots
ax.scatter(*x_im1, color='k', s=40)
ax.scatter(*x_i, color='k', s=40)
ax.scatter(*x_ip1, color='k', s=40)

# Per-label offsets for each frame (now as 3D vectors)
label_offsets_1 = {'t': np.array([-0.02, -0.02, 0.05]), 'm1': np.array([0, 0.025, 0]), 'm2': np.array([0, 0, 0.025])}
label_offsets_2 = {'t': np.array([0.025, 0, 0.03]), 'm1': np.array([0, 0.025, 0]), 'm2': np.array([0, 0, 0.025])}
label_offsets_2_rot = {'t': np.array([0.025, 0, 0.03]), 'm1': np.array([0, 0.025, 0]), 'm2': np.array([0, 0, 0.025])}

# Per-label offsets for x labels
x_label_offsets = {
    'x_im1': np.array([-0.06, -0.08, 0.07]),
    'x_i':   np.array([0.0, 0.0, 0.04]),
    'x_ip1': np.array([0.0, 0.04, 0.0]),
}

# Plot frames at the chosen position along each line (dark blue)
dark_blue = '#003366'
frame_pos_ratio = 0.35  # 0=start, 1=end, 0.5=midpoint
frame1_pos = x_im1 + frame_pos_ratio * (x_i - x_im1)
frame2_pos = x_i + frame_pos_ratio * (x_ip1 - x_i)
plot_frame(
    ax, frame1_pos, t1, m1_1, m2_1, length=0.4, color=dark_blue, label_color=dark_blue, offsets=label_offsets_1,
    label_names=['t', 'u', 'v'], superscript='i-1')
plot_frame(
    ax, frame2_pos, t2, m1_2, m2_2, length=0.4, color=dark_blue, label_color=dark_blue, offsets=label_offsets_2,
    label_names=['t', 'u', 'v'], superscript='i')

# Plot a duplicate of the 2nd frame, rotated 30 degrees about the tangent (dark brown)
theta = np.deg2rad(30)
m1_2_rot = rotate_about_axis(m1_2, t2, theta)
m2_2_rot = rotate_about_axis(m2_2, t2, theta)
dark_brown = '#5C4033'
plot_frame(
    ax, frame2_pos, t2, m1_2_rot, m2_2_rot, length=0.4, color=dark_brown, label_color=dark_brown, offsets=label_offsets_2_rot,
    label_names=['t', 'm_1', 'm_2'], superscript='i')

# Draw arc at the joint (x_i) between t1 and t2, label it
arc_label_offset = np.array([-0.04, -0.04, -0.06])  # <--- You can set this to control the arc label's 3D offset
# plot_arc(ax, x_i, -t1, t2, radius=arc_radius, color='k', lw=0.5, label=r'$\pi - \phi_i$', label_offset=arc_label_offset)
plot_arc(ax, x_i, -t1, t2, radius=arc_radius, color='k', lw=0.5, label=r'', label_offset=arc_label_offset)
# Draw arc at frame2_pos between m1_2 and m1_2_rot (non-rotated and rotated frame of the second line)
arc2_label_offset = np.array([0.04, 0.0, 0.02])  # <--- You can set this to control the new arc label's 3D offset
# plot_arc(
#     ax,
#     frame2_pos,
#     m1_2,
#     m1_2_rot,
#     radius=0.23,  # You can adjust this radius as needed
#     color='brown',
#     lw=1.2,
#     label=r'$\theta^i$',
#     label_offset=arc2_label_offset
# )

# Annotate points with offsets
ax.text(*(x_im1 + x_label_offsets['x_im1']), r'$x_{i-1}$', fontsize=12, fontweight='bold')
ax.text(*(x_i + x_label_offsets['x_i']), r'$x_{i}$', fontsize=12, fontweight='bold')
ax.text(*(x_ip1 + x_label_offsets['x_ip1']), r'$x_{i+1}$', fontsize=12, fontweight='bold')

ax.set_xlabel('')
ax.set_ylabel('')
ax.set_zlabel('')
# ax.set_title('Two connected 3D lines with local frames at segment centers')

def set_axes_equal(ax):
    '''Set 3D plot axes to equal scale visually.'''
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    max_range = max([x_range, y_range, z_range])
    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)
    ax.set_xlim3d([x_middle - max_range/2, x_middle + max_range/2])
    ax.set_ylim3d([y_middle - max_range/2, y_middle + max_range/2])
    ax.set_zlim3d([z_middle - max_range/2, z_middle + max_range/2])

set_axes_equal(ax)
ax.set_box_aspect([1,1,1])

def plot_arc_with_arrow(ax, origin, vec, radius=0.08, color='purple', lw=1.5, n_points=60, arc_angle=np.pi*1.6, label=None, label_offset=np.zeros(3)):
    """
    Plots an arc (not a full circle) perpendicular to 'vec', centered at 'origin', with an arrowhead at the end.
    arc_angle: total angle of the arc in radians (default: 1.2*pi, i.e., 216 degrees)
    """
    vec = vec / np.linalg.norm(vec)
    # Find two orthonormal vectors perpendicular to vec
    if np.allclose(vec, [0, 0, 1]):
        ortho = np.array([1, 0, 0])
    else:
        ortho = np.array([0, 0, 1])
    v1 = np.cross(vec, ortho)
    v1 /= np.linalg.norm(v1)
    v2 = np.cross(vec, v1)
    v2 /= np.linalg.norm(v2)
    # Arc points
    ts = np.linspace(0, arc_angle, n_points)
    arc_points = np.array([
        origin + radius * (np.cos(t) * v1 + np.sin(t) * v2)
        for t in ts
    ])
    ax.plot(arc_points[:,0], arc_points[:,1], arc_points[:,2], color=color, lw=lw)
    # Arrowhead
    from mpl_toolkits.mplot3d import proj3d
    # Arrow direction: tangent to the arc at the end
    if n_points > 1:
        p0 = arc_points[-2]
        p1 = arc_points[-1]
        arrow_vec = p1 - p0
        arrow_vec = arrow_vec / np.linalg.norm(arrow_vec) * (radius * 0.2)
        ax.quiver(
            p1[0], p1[1], p1[2],
            arrow_vec[0], arrow_vec[1], arrow_vec[2],
            color=color, lw=lw, arrow_length_ratio=2, length=radius*0.2, normalize=True
        )
    # Label
    if label is not None:
        label_pos = arc_points[n_points//2] + label_offset
        ax.text(*label_pos, label, fontsize=13, color=color, fontweight='bold')

# For frame 2
plot_arc_with_arrow(
    ax, frame2_pos + m2_2_rot*0.25, m2_2_rot, radius=0.04, color='purple',
    label=r'$\gamma^i_1$', label_offset=np.array([-0.05, 0.05, 0.06])
)
plot_arc_with_arrow(
    ax, frame2_pos + m1_2_rot*0.25, m1_2_rot, radius=0.04, color='green',
    label=r'$\gamma^i_2$', label_offset=np.array([0.03, -0.08, 0.1])
)
plot_arc_with_arrow(
    ax, frame2_pos + t2*0.25, t2, radius=0.04, color='blue',
    label=r'$\theta^i$', label_offset=np.array([0.0, 0.0, 0.06])
)

plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
fig_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "adapteddlo_muj/data/figs/plgn/"
)
plt.savefig(fig_dir+'sampleDER2.pdf', bbox_inches='tight', pad_inches=0)
plt.show()