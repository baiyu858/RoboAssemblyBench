"""Reusable contact insertion RL. Importing this package never starts a simulator."""
from .control import InsertionConfig
from .task import InsertionTask

__all__ = ['InsertionConfig', 'InsertionTask']
