PORT = 8080

CMD_LEN = 4

# All of these messages sent from the client are prefaced with the total message length. 

# Make new enviroment. 
# Message: MAKE.len(env_name).env_name
# Response: done
MAKE_ENV_CMD = "MAKE"

# Resets the enviroment. Returns a map of some sort. 
# Message: RSET
# Response: pickle encoded version of env.reset()
RESET_ENV_CMD = "RSET"

# Returns env._max_episode_steps
# Message: EPST
# Response: pickle encoded _max_episode_steps
GET_MAX_EPISODE_STEPS_CMD = "EPST"

# Step through the sim. Returns new observations, reward, and info. 
# Message: STEP.pickle(action)
# Response: pickle encoded env.step(action)
STEP_CMD = "STEP"

# Render the enviroment
# Message: RNDR
# Response: done
RENDER_CMD = "RNDR"

# Get action space
# Message: ACSP
# Response: pickle encoded env.action_space
GET_ACTION_SPACE_CMD = "ACSP"

# Terminates the server. No response. 
BYE_CMD = "BYE!"

# Gets env.compute_reward
# Message: RWRD
# Response: pickle encoded reward value
COMPUTE_REWARD_CMD = "RWRD"

# Seed the gym
# Message: SEED.seed_value
# Response: done
SEED_CMD = "SEED"