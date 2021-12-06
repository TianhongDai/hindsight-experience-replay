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
        id='HandManipulateBlockPosEp80Sim25-v0',
        entry_point='envs.gym_robotics.hand.manipulate:HandBlockEnv',
        kwargs=_merge({'target_position': 'random', 'target_rotation': 'ignore',
                       'n_substeps': 25}, kwargs),
        max_episode_steps=80,
    )

register(
        id='HandManipulateBlockPosEp50Sim40-v0',
        entry_point='envs.gym_robotics.hand.manipulate:HandBlockEnv',
        kwargs=_merge({'target_position': 'random', 'target_rotation': 'ignore',
                       'n_substeps': 40}, kwargs),
        max_episode_steps=50,
    )


# Full
register(
        id='HandManipulateBlock500-v0',
        entry_point='envs.gym_robotics.hand.manipulate:HandBlockEnv',
        kwargs=_merge({'target_position': 'random', 'target_rotation': 'xyz'}, kwargs),
        max_episode_steps=500,
    )


register(
        id='HandManipulateBlockEp400Sim25-v0',
        entry_point='envs.gym_robotics.hand.manipulate:HandBlockEnv',
        kwargs=_merge({'target_position': 'random', 'target_rotation': 'xyz',
                       'n_substeps': 25}, kwargs),
        max_episode_steps=400,
    )


# rotation xyz
register(
        id='HandManipulateBlockRotateXYZEp80Sim25-v0',
        entry_point='envs.gym_robotics.hand.manipulate:HandBlockEnv',
        kwargs=_merge({'target_position': 'ignore', 'target_rotation': 'xyz',
                       'n_substeps': 25}, kwargs),
        max_episode_steps=80,
    )

register(
        id='HandManipulateBlockRotateXYZEp50Sim40-v0',
        entry_point='envs.gym_robotics.hand.manipulate:HandBlockEnv',
        kwargs=_merge({'target_position': 'ignore', 'target_rotation': 'xyz',
                       'n_substeps': 40}, kwargs),
        max_episode_steps=50,
    )

# rotate xyz with fixed pos
register(
        id='HandBlockFixedPosRotateXYZ-v0',
        entry_point='envs.gym_robotics.hand.manipulate:HandBlockEnv',
        kwargs=_merge({'target_position': 'fixed', 'target_rotation': 'xyz'}, kwargs),
        max_episode_steps=100,
    )

# random pos with fixed rotation
register(
        id='HandBlockFixedRotationPos-v0',
        entry_point='envs.gym_robotics.hand.manipulate:HandBlockEnv',
        kwargs=_merge({'target_position': 'random', 'target_rotation': 'fixed'}, kwargs),
        max_episode_steps=100,
    )


