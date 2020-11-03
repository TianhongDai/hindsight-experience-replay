
import pickle
import socket
from communication_consts import BYE_CMD, GET_ACTION_SPACE_CMD, GET_MAX_EPISODE_STEPS_CMD, MAKE_ENV_CMD, PORT, RENDER_CMD, RESET_ENV_CMD, STEP_CMD

class Client:
  def __init__(self, hostname) -> None:
    self.s = socket.socket()
    self.s.connect((hostname, PORT))

  def _get_response(self):
    while True:
      res = self.s.recv(2048)
      if (len(res) > 0):
        return res

  def _send_byte_message(self, msg):
    self.s.send(msg)
    return self._get_response()

  def _send_cmd(self, cmd):
    msg = bytes(cmd, 'utf-8')
    return self._send_byte_message(msg)

  def _send_pickle_res_msg(self, msg):
    return pickle.loads(self._send_byte_message(msg))

  def _basic_pickle_res_cmd(self, cmd):
    msg = bytes(cmd, 'utf-8')
    return self._send_pickle_res_msg(msg)



  def create_env(self, env_name):
    msg = bytes(MAKE_ENV_CMD, 'utf-8')
    msg += len(env_name).to_bytes(4, byteorder="big")
    msg += bytes(env_name, 'utf-8')
    self._send_byte_message(msg)

  def reset_env(self):
    return self._basic_pickle_res_cmd(RESET_ENV_CMD)

  def get_max_episode_steps(self):
    return self._basic_pickle_res_cmd(GET_MAX_EPISODE_STEPS_CMD)

  def step_enviroment(self, action):
    msg = bytes(STEP_CMD, 'utf-8')
    msg += pickle.dumps(action)
    return self._send_pickle_res_msg(msg)

  def render_enviroment(self):
    self._send_cmd(RENDER_CMD)

  def get_action_space(self):
    return self._basic_pickle_res_cmd(GET_ACTION_SPACE_CMD)

  def close(self):
    self.s.send(bytes(BYE_CMD, 'utf-8'))