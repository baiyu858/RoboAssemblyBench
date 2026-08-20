import socket
from typing import Dict, List, Optional, Union

from internutopia.core.config import Config, DistributedConfig, TaskCfg


def runtime_root_path(task_config: TaskCfg, env_id: int) -> str:
    root_path = f'/World/env_{str(env_id)}'
    episode_idx = getattr(task_config, 'episode_idx', None)
    if episode_idx is not None:
        root_path += f'/episode_{int(episode_idx):06d}'
    return root_path


def _runtime_scoped_sensor_path(prim_path: str, root_path: str) -> str:
    """Move environment-owned absolute sensor paths into the active episode."""
    parts = str(prim_path).split('/')
    if (
        len(parts) >= 4
        and parts[0] == ''
        and parts[1] == 'World'
        and parts[2].startswith('env_')
        and parts[2].removeprefix('env_').isdigit()
    ):
        return root_path.rstrip('/') + '/' + '/'.join(parts[3:])
    return str(prim_path)


def setup_offset_for_assets(task_config: TaskCfg, env_id: int, offset: List[float]):
    root_path = runtime_root_path(task_config, env_id)

    task_config.robots = [r.model_copy(deep=True) for r in task_config.robots]
    task_config.objects = [o.model_copy(deep=True) for o in task_config.objects if task_config.objects]

    for r in task_config.robots:
        r.name = f'{r.name}_{env_id}'
        r.prim_path = root_path + task_config.robots_root_path + r.prim_path
        r.position = [offset[idx] + pos for idx, pos in enumerate(r.position)]
        for sensor in r.sensors or []:
            if getattr(sensor, 'prim_path', None):
                sensor.prim_path = _runtime_scoped_sensor_path(sensor.prim_path, root_path)
    if task_config.objects is not None:
        for o in task_config.objects:
            o.name = f'{o.name}_{env_id}'
            o.prim_path = root_path + task_config.objects_root_path + o.prim_path
            o.position = [offset[idx] + pos for idx, pos in enumerate(o.position)]


class BaseTaskConfigManager:
    """Base class of config manager.

    Attributes:
        env_num (int): Env number.
        task_configs (TaskConfig): Task config that user input.
        env_offset_size (float): Offset size.
    """

    managers = {}

    def __init__(self, config: Config):
        self.task_configs: List[TaskCfg] = config.task_configs
        self.current_task_config_idx = 0
        self.env_num = config.env_num
        self.env_offset_size = config.env_offset_size

        # validate episodes
        self.len_tasks = self.validate_task_configs()

        # validate env_num
        if self.env_num <= 0:
            raise RuntimeError('env_num must be greater than 0')

    def validate_task_configs(self):
        task_cfg_count = 0
        _order = None
        for idx, episode in enumerate(self.task_configs):
            task_cfg_count += 1
            robot_order = [robot.type for robot in episode.robots]
            if _order is None:
                _order = robot_order
                continue
            if robot_order != _order:
                raise ValueError('robot types must be identical across all episodes')
        if task_cfg_count == 0:
            raise RuntimeError('The len of task_configs must be greater than 0')
        return task_cfg_count

    def get_active_task_configs(self) -> Dict[int, TaskCfg]:
        """
        Get active task configs with env id as key.
        """
        raise NotImplementedError()

    def get_next(self, last_env_id: Optional[int] = None) -> tuple[str, int, list[float], TaskCfg]:
        """
        Get next task's [task_name, env_id, env_offset, task_config] with env of last episode.
        Args:
            last_env_id (Optional[int]): env_id of last episode.

        Returns:
            TaskName (str): Task name.
            EnvID (int): env id of next task.
            EnvOffset (List[float] | None): env_offset of next task.
            TaskCfg: TaskCfg for next task.
        """
        raise NotImplementedError()

    def _pop_next_episode(self) -> Union[TaskCfg, None]:
        """
        Pop task_config of next episode from task_config list.

        Returns:
            TaskCfg: The TaskCfg of next episode.
        """
        if self.current_task_config_idx < self.len_tasks:
            next_episode = self.task_configs[self.current_task_config_idx]
            self.current_task_config_idx += 1
            return next_episode
        return None


def create_task_config_manager(config: Config):
    if not isinstance(config, DistributedConfig):
        from .local import LocalTaskConfigManager

        return LocalTaskConfigManager(config)

    import ray

    address = config.distribution_config.head_address
    if address is not None:
        try:
            local_ip = socket.gethostbyname(socket.gethostname())
        except Exception:
            local_ip = None
        if local_ip == address:
            address = f'{address}:6379'
        else:
            address = f'ray://{address}:10001'
    runtime_env = None
    if config.distribution_config.working_dir is not None:
        runtime_env = {
            'working_dir': config.distribution_config.working_dir,
        }
    try:
        ray.init(
            address=address,
            runtime_env=runtime_env,
            ignore_reinit_error=True,
        )
    except ConnectionError as e:
        raise RuntimeError(
            f'Fail to create distributed task config manager: {e}, Please confirm that the address {address} can be accessed normally'
        )
    except Exception as e:
        raise RuntimeError(f'Fail to create distributed task config manager: {e}')
    _remote_args = {
        'num_cpus': 1,
    }
    from .distributed import DistributedTaskConfigManager

    remote_class = ray.remote(**_remote_args)(DistributedTaskConfigManager)
    return remote_class.options(placement_group_bundle_index=-1).remote(config)
