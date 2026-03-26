# Factory Box Carry Toolkit

This toolkit adds a first-pass factory-like cooperative carry demo to `InternUtopia`.

What it contains:

- `FactoryBoxCarryTask`: a two-robot cooperative carry task
- `HumanoidBenchH1RobotCfg`: Isaac Sim compatible H1-with-hand wrapper with HumanoidBench-style atomic skill names
- `generate_demos.py`: RoboTwin-style demo generation script that first searches successful seeds, then replays them for trajectory export
- `render_isaac_video.py`: fixed-camera Isaac Sim 3D renderer that exports per-frame RGB images and an MP4 preview

Run:

```bash
cd /home/baiyu24/model/InternUtopia
python toolkits/factory_box_carry/generate_demos.py --num-demos 4 --max-trials 20 --headless
```

Render a 3D Isaac Sim video:

```bash
cd /home/baiyu24/model/InternUtopia
python toolkits/factory_box_carry/render_isaac_video.py --seed 0 --headless
```

Outputs:

- `seed.txt`: successful seeds
- `manifest.json`: summary of generated demos
- `episode_XXXX.json`: per-step observations, actions, box trajectory, and task phase

Notes:

- The first version reuses InternUtopia's Isaac Sim compatible H1-with-hand embodiment and exposes HumanoidBench-style skill interfaces.
- If HumanoidBench reach assets are available, the policy will use the reach primitive; otherwise it falls back to locomotion-only cooperative carry.
