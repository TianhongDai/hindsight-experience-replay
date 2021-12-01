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

register(
        id='HandManipulateBlockPos-v0',
        entry_point='envs.gym_robotics.hand.manipulate:HandBlockEnv',
        kwargs=_merge({'target_position': 'random', 'target_rotation': 'ignore'}, kwargs),
        max_episode_steps=100,
    )

