import numpy as np
import torch
from torch.utils.data import RandomSampler, TensorDataset
from torchdata.stateful_dataloader import StatefulDataLoader

from agent_system.environments.base import EnvironmentManagerBase
from agent_system.environments.env_package.webshop.envs import WebshopMultiProcessEnv


def _make_search_loader() -> StatefulDataLoader:
    dataset = TensorDataset(torch.arange(64))
    generator = torch.Generator().manual_seed(1)
    sampler = RandomSampler(dataset, generator=generator)
    return StatefulDataLoader(dataset, batch_size=8, sampler=sampler)


def _make_webshop_sampler() -> WebshopMultiProcessEnv:
    env = WebshopMultiProcessEnv.__new__(WebshopMultiProcessEnv)
    env._rng = np.random.RandomState(7)
    env._validation_size = None
    env._validation_goal_order = None
    env._validation_goal_cursor = 0
    env.goal_idxs = range(500, 1500)
    env.env_num = 8
    env._closed = True
    return env


def test_search_resumes_next_real_batch_from_dataloader_state():
    loader = _make_search_loader()
    iterator = iter(loader)
    for _ in range(3):
        next(iterator)
    state = loader.state_dict()
    expected = next(iterator)[0]

    resumed = _make_search_loader()
    resumed.load_state_dict(state)

    assert torch.equal(next(iter(resumed))[0], expected)


def test_webshop_rebuilds_next_real_goal_batch_from_step():
    uninterrupted = _make_webshop_sampler()
    for _ in range(5):
        uninterrupted._sample_goal_indices()
    expected = uninterrupted._sample_goal_indices()

    resumed = _make_webshop_sampler()
    manager = EnvironmentManagerBase(resumed, projection_f=None, config=None)
    assert manager.advance_data_state(num_batches=5)

    assert resumed._sample_goal_indices() == expected
