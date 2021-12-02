# from gym.envs.robotics.fetch_env import FetchEnv
# from gym.envs.robotics.fetch.slide import FetchSlideEnv
# from gym.envs.robotics.fetch.pick_and_place import FetchPickAndPlaceEnv
# from gym.envs.robotics.fetch.push import FetchPushEnv
# from gym.envs.robotics.fetch.reach import FetchReachEnv
#
# from gym.envs.robotics.hand.reach import HandReachEnv

from envs.gym_robotics.hand.manipulate import HandBlockEnv


from gym.envs.registration import registry, register, make, spec

def _merge(a, b):
    a.update(b)
    return a

kwargs = {
        'reward_type': 'sparse',
    }


# Pos
register(
        id='HandManipulateBlockPos-v0',
        entry_point='envs.gym_robotics.hand.manipulate:HandBlockEnv',
        kwargs=_merge({'target_position': 'random', 'target_rotation': 'ignore'}, kwargs),
        max_episode_steps=100,
    )

register(
        id='HandManipulateBlockPosEp200Sim10-v0',
        entry_point='envs.gym_robotics.hand.manipulate:HandBlockEnv',
        kwargs=_merge({'target_position': 'random', 'target_rotation': 'ignore',
                       'n_substeps': 10}, kwargs),
        max_episode_steps=200,
    )

register(
        id='HandManipulateBlockPosEp400Sim5-v0',
        entry_point='envs.gym_robotics.hand.manipulate:HandBlockEnv',
        kwargs=_merge({'target_position': 'random', 'target_rotation': 'ignore',
                       'n_substeps': 5}, kwargs),
        max_episode_steps=400,
    )


# Full
register(
        id='HandManipulateBlock500-v0',
        entry_point='envs.gym_robotics.hand.manipulate:HandBlockEnv',
        kwargs=_merge({'target_position': 'random', 'target_rotation': 'xyz'}, kwargs),
        max_episode_steps=500,
    )


register(
        id='HandManipulateBlockEp200Sim10-v0',
        entry_point='envs.gym_robotics.hand.manipulate:HandBlockEnv',
        kwargs=_merge({'target_position': 'random', 'target_rotation': 'xyz',
                       'n_substeps': 10}, kwargs),
        max_episode_steps=200,
    )


# rotation xyz
register(
        id='HandManipulateBlockRotateXYZEp400Sim5-v0',
        entry_point='envs.gym_robotics.hand.manipulate:HandBlockEnv',
        kwargs=_merge({'target_position': 'ignore', 'target_rotation': 'xyz',
                       'n_substeps': 5}, kwargs),
        max_episode_steps=400,
    )

register(
        id='HandManipulateBlockRotateXYZEp200Sim10-v0',
        entry_point='envs.gym_robotics.hand.manipulate:HandBlockEnv',
        kwargs=_merge({'target_position': 'ignore', 'target_rotation': 'xyz',
                       'n_substeps': 10}, kwargs),
        max_episode_steps=200,
    )

