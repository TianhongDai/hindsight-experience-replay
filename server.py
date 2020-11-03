import socket
import gym
import pickle
from communication_consts import BYE_CMD, GET_ACTION_SPACE_CMD, GET_MAX_EPISODE_STEPS_CMD, MAKE_ENV_CMD, PORT, RENDER_CMD, RESET_ENV_CMD, STEP_CMD

ADDR = socket.gethostname()

print(f"Server hostname: {socket.gethostname()}")


s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind((ADDR, PORT))

s.listen(1)
conn, address = s.accept()
print("Accepted connection")

env = None

def pickle_response(data):
  res = pickle.dumps(data)
  conn.send(res)

def done_response():
  conn.send(bytes("done", 'utf-8'))

while True:
  data = conn.recv(1024)
  if len(data) >= 4:
    cmd = data[0:4].decode('utf-8')
    if (cmd == MAKE_ENV_CMD):
      env_name_len = int.from_bytes(data[4:8], byteorder="big")
      env_name = data[8:8+env_name_len].decode('utf-8')
      print(f"Creating enviroment: {env_name}")
      env = gym.make(env_name)
      conn.send(bytes("done", 'utf-8'))
    elif (cmd == RESET_ENV_CMD):
      print("Reseting enviroment")
      pickle_response(env.reset())
    elif (cmd == GET_MAX_EPISODE_STEPS_CMD):
      print("Getting max episode steps")
      pickle_response(env._max_episode_steps)
    elif (cmd == STEP_CMD):
      print("Enviroment step")
      action = pickle.loads(data[4:])
      pickle_response(env.step(action))
    elif (cmd == RENDER_CMD):
      print("Rendering")
      env.render()
      done_response()
    elif (cmd == GET_ACTION_SPACE_CMD):
      print("Action space")
      pickle_response(env.action_space)
    elif (cmd == BYE_CMD):
      exit()
    else:
      print(f"Unrecognized command: {cmd}")
