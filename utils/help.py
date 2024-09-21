'''
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def show_point_cloud(points, points1, idb):
    if idb == 1:
        color = 'b' 
    else:
        color = 'y'
    #features = features.permute(1,2,3,0).flatten(0, 2).cpu().detach().numpy()
    points = points.flatten(0, 2).cpu().numpy()
    points1 = points1.cpu().numpy()
    
    #all_zero_features = np.all(features == 0, axis=1)
    #colors = np.where(all_zero_features, 'purple', 'yellow')
    
    ax.scatter(points1[:, 0], points1[:, 1], points1[:, 2], c=color, cmap='viridis')
    #ax.scatter(points1[:, 0], points1[:, 1], points1[:, 2], c='b', cmap='viridis')
    #ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='y', cmap='viridis')
    
    
    ax.plot([0, 0], [0, 0], [0, 2])
    ax.plot([0, 0], [0, 25], [0, 0])
    ax.plot([0, 25], [0, 0], [0, 0])
    
    ax.plot([0, 25], [25, 25], [2, 2])
    ax.plot([0, 0], [25, 25], [2, 0])
    ax.plot([0, 0], [25, 0], [2, 2])
    
    ax.plot([25, 25], [25, 25], [0, 2])
    ax.plot([25, 0], [25, 25], [0, 0])
    ax.plot([25, 25], [25, 0], [0, 0])
    
    ax.plot([25, 0], [0, 0], [2, 2])
    ax.plot([25, 25], [0, 25], [2, 2])
    ax.plot([25, 25], [0, 0], [2, 0])
    
    ax.set_box_aspect([1.0, 1.0, 1.0])
        
    set_axes_equal(ax)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    
    plt.show()
    ax.cla()
    return
'''