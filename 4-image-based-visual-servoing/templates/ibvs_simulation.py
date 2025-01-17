import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv, norm
from liegroups import SE3
from ibvs_controller import ibvs_controller
from ibvs_depth_finder import ibvs_depth_finder

import os
import imageio.v2 as imageio

 # Image plane size for plotting.
IMAGE_SIZE = (800.0, 600.0)

# Initial guess for all depths.
Z_GUESS = 2.0

# Maximum number of iterations to run.
MAX_ITERS = 150

def project_into_camera(Twc, K, pts):
    """Project points into camera. Returns truth depths, too."""
    pts = np.vstack((pts, np.ones((1, pts.shape[1]))))
    pts_cam = (inv(Twc)@pts)[0:3, :]
    zs = pts_cam[2, :]  # Depths in the camera frame.
    pts_cam = K@pts_cam/pts_cam[2:3, :]
    return pts_cam[0:2, :], zs # Discard last 1 rows.

def plot_image_points(pts_des, pts_obs, save_path=None):
    """Plot observed and desired image plane points."""
    plt.clf()
    plt.plot(pts_des[0, :], pts_des[1, :], 'rx', label="Desired Points")
    plt.plot(pts_obs[0, :], pts_obs[1, :], 'bo', label="Observed Points")
    plt.xlim([0, IMAGE_SIZE[0]])
    plt.ylim([0, IMAGE_SIZE[1]])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)  # Save frame to disk
    
    plt.show(block=False)
    plt.pause(0.2)

def ibvs_simulation(Twc_init, 
                    Twc_last, 
                    pts, 
                    K, 
                    gain, 
                    do_depth = False, 
                    do_plot  = True):
    """
    Run simple IBVS simulation and plot the results.
    
    Parameters:
    -----------
    Twc_init - 4x4 np.array, initial camera pose in target frame.
    Twc_last - 4x4 np.array, final (desired) camera pose in target frame.
    pts      - 3xn np.array, feature points (in 3D) in target frame.
    K        - 3x3 np.array, camera intrinsic calibration matrix.
    gain     - Controller gain (lambda).
    do_depth - Use depth estimates, rather than true depths.
    do_plot  - Plot image plane points.
    """
    do_video = False # create video from sequence of plots
    if do_video:
        frames = []
        video_filename="output_video.webm"

        base_path = "frames/"
        if not os.path.exists(base_path):
            os.makedirs(base_path)
    # Find desired point coordinates.
    pts_des, _ = project_into_camera(Twc_last, K, pts)

    # Find initial point coordinates.
    pts_obs, zs = project_into_camera(Twc_init, K, pts)
    pts_prev = pts_obs # Store last positions of points.

    # If we are pretending we don't know the depths, then
    # make an initial guess here...
    if do_depth:
        zs = Z_GUESS*np.ones(zs.shape)

    if do_plot:
        # Plot initial configuration, before moving.
        plt.figure(1)
        plot_image_points(pts_des, pts_obs)

    # Loop until velocity is close to zero.
    Twc_now = Twc_init
    i = 0

    while i < MAX_ITERS:        
        # Get next control command.
        v = ibvs_controller(K, pts_des, pts_obs, zs, gain)

        # If nearly stopped, bail out.
        if norm(v) < 1e-4: break 

        # Delta pose update for camera.
        Tdelta = SE3.exp(v[:, 0])

        # Update camera pose - note that after a large number of
        # incremental updates, the rotation matrix may not be
        # orthonormal.
        Twc_now = Twc_now@Tdelta.as_matrix()

        # Reproject points in new camera frame.
        if do_depth:
            pts_obs, _ = project_into_camera(Twc_now, K, pts)
            zs = ibvs_depth_finder(K, pts_obs, pts_prev, v)
            pts_prev = pts_obs
        else:
            pts_obs, zs = project_into_camera(Twc_now, K, pts)

        # if do_plot:
        #     # Plot current configuration, while moving.
        #     plt.figure(1)
        #     plot_image_points(pts_des, pts_obs)


        if do_plot or do_video:
            plt.figure(1)
            if do_video:
                frame_path = base_path + f"frame_{i:04d}.png"
                plot_image_points(pts_des, pts_obs, save_path=frame_path)
                frames.append(frame_path)  # Save frame paths
            else:
                plot_image_points(pts_des, pts_obs)
        
        # Increment counter.
        i += 1
        # if not do_plot:
        #     print(norm(v))

    if do_video:
        # Generate video using ImageIO
        with imageio.get_writer(video_filename, fps=15, codec='libaom-av1', ffmpeg_params=['-strict', '-2']) as writer:
            for frame_path in frames:
                writer.append_data(imageio.imread(frame_path))
                # Optionally clean up frame files
                # os.remove(frame_path)  # Uncomment to delete frames after video creation
        print(f"Video saved to {video_filename}")
        # with imageio.get_writer("vp9_" + video_filename, fps=10, codec='vp9') as writer:
        #     for frame_path in frames:
        #         writer.append_data(imageio.imread(frame_path))
        #         # Optionally clean up frame files
        #         # os.remove(frame_path)  # Uncomment to delete frames after video creation
        # print(f"Video saved to {video_filename}")

    # print(f"Number of iterations: {i}")
    return i