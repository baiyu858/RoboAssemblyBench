import argparse
import os
import shlex
import sys

def main():
    parser = argparse.ArgumentParser(description="Run RoboFactory planner with task config input.")
    parser.add_argument('config', type=str, help="Task config file to use")
    parser.add_argument('--save-video', action='store_true', help="Save an mp4 for the run.")
    parser.add_argument('--record-dir', type=str, default='demos', help="Directory used to save trajectories/videos.")
    parser.add_argument('--render-mode', type=str, default=None, help="Video render mode, usually 'rgb_array' or 'sensors'.")
    parser.add_argument('--num-traj', type=int, default=1, help="Number of trajectories to run.")
    parser.add_argument('--sim-backend', type=str, default='cpu', help="Simulation backend: auto/cpu/gpu.")
    parser.add_argument('--vis', dest='vis', action='store_true', help="Open the live GUI viewer.")
    parser.add_argument('--no-vis', dest='vis', action='store_false', help="Disable the live GUI viewer.")
    parser.set_defaults(vis=True)
    args = parser.parse_args()

    python_bin = shlex.quote(sys.executable)
    config_path = shlex.quote(args.config)
    record_dir = shlex.quote(args.record_dir)
    render_mode = args.render_mode
    if render_mode is None:
        render_mode = "rgb_array" if args.save_video else "human"
    command = (
        f"{python_bin} -m robofactory.planner.run "
        f"-c {config_path} "
        f"--render-mode=\"{render_mode}\" "
        f"-b=\"{args.sim_backend}\" "
        f"-n {args.num_traj} "
        f"--record-dir {record_dir} "
        + ("--vis " if args.vis else "")
        + ("--save-video" if args.save_video else "")
    )

    os.system(command)

if __name__ == "__main__":
    main()
