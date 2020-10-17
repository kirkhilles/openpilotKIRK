import math
import numpy as np
import cereal.messaging as messaging
from common.realtime import sec_since_boot
from selfdrive.controls.lib.drive_helpers import MPC_COST_LONG
from common.op_params import opParams
from common.numpy_fast import interp, clip
from selfdrive.config import Conversions as CV
from cereal.messaging import SubMaster

from selfdrive.controls.lib.dynamic_follow.auto_df import predict
from selfdrive.controls.lib.dynamic_follow.df_manager import dfManager
from selfdrive.controls.lib.dynamic_follow.support import LeadData, CarData, dfData, dfProfiles
from common.data_collector import DataCollector
travis = False


class DynamicFollow:
  def __init__(self, mpc_id):
    self.mpc_id = mpc_id
    self.op_params = opParams()
    self.df_profiles = dfProfiles()
    self.df_manager = dfManager(self.op_params)

    if not travis and mpc_id == 1:
      self.pm = messaging.PubMaster(['dynamicFollowData'])
    else:
      self.pm = None

    # Model variables
    mpc_rate = 1 / 20.
    self.model_scales = {'v_ego': [-0.06112159043550491, 37.96522521972656], 'a_lead': [-3.109330892562866, 3.3612186908721924], 'v_lead': [0.0, 35.27671432495117], 'x_lead': [2.4600000381469727, 141.44000244140625]}
    self.predict_rate = 1 / 4.
    self.skip_every = round(0.25 / mpc_rate)
    self.model_input_len = round(45 / mpc_rate)

    # Dynamic follow variables
    self.default_TR = 1.8
    self.TR = 1.8
    self.v_ego_retention = 2.5
    self.v_rel_retention = 1.75

    self.sng_TR = 1.8  # reacceleration stop and go TR
    self.sng_speed = 18.0 * CV.MPH_TO_MS

    self._setup_collector()
    self._setup_changing_variables()

  def _setup_collector(self):
    self.sm_collector = SubMaster(['liveTracks', 'laneSpeed'])
    self.log_auto_df = self.op_params.get('log_auto_df')
    if not isinstance(self.log_auto_df, bool):
      self.log_auto_df = False
    self.data_collector = DataCollector(file_path='/data/df_data', keys=['v_ego', 'a_ego', 'a_lead', 'v_lead', 'x_lead', 'left_lane_speeds', 'middle_lane_speeds', 'right_lane_speeds', 'left_lane_distances', 'middle_lane_distances', 'right_lane_distances', 'profile', 'time'], log_data=self.log_auto_df)

  def _setup_changing_variables(self):
    self.TR = self.default_TR
    self.user_profile = self.df_profiles.relaxed  # just a starting point
    self.model_profile = self.df_profiles.relaxed

    self.last_effective_profile = self.user_profile
    self.profile_change_time = 0

    self.sng = False
    self.car_data = CarData()
    self.lead_data = LeadData()
    self.df_data = dfData()  # dynamic follow data

    self.last_cost = 0.0
    self.last_predict_time = 0.0
    self.auto_df_model_data = []
    self._get_live_params()  # so they're defined just in case

  def update(self, CS, libmpc):
    self._get_live_params()
    self._update_car(CS)
    self._get_profiles()

    if self.mpc_id == 1 and self.log_auto_df:
      self._gather_data()

    if not self.lead_data.status:
      self.TR = self.default_TR
    else:
      self._store_df_data()
      self.TR = self._get_TR()

    if not travis:
      self._change_cost(libmpc)
      self._send_cur_state()

    return self.TR

  def _get_profiles(self):
    """This receives profile change updates from dfManager and runs the auto-df prediction if auto mode"""
    df_out = self.df_manager.update()
    self.user_profile = df_out.user_profile
    if df_out.is_auto:  # todo: find some way to share prediction between the two mpcs to reduce processing overhead
      self._get_pred()  # sets self.model_profile, all other checks are inside function

  def _gather_data(self):
    self.sm_collector.update(0)
    # live_tracks = [[i.dRel, i.vRel, i.aRel, i.yRel] for i in self.sm_collector['liveTracks']]
    if self.car_data.cruise_enabled:
      self.data_collector.update([self.car_data.v_ego,
                                  self.car_data.a_ego,
                                  self.lead_data.a_lead,
                                  self.lead_data.v_lead,
                                  self.lead_data.x_lead,
                                  list(self.sm_collector['laneSpeed'].leftLaneSpeeds),
                                  list(self.sm_collector['laneSpeed'].middleLaneSpeeds),
                                  list(self.sm_collector['laneSpeed'].rightLaneSpeeds),

                                  list(self.sm_collector['laneSpeed'].leftLaneDistances),
                                  list(self.sm_collector['laneSpeed'].middleLaneDistances),
                                  list(self.sm_collector['laneSpeed'].rightLaneDistances),
                                  self.user_profile,
                                  sec_since_boot()])

  def _norm(self, x, name):
    self.x = x
    return interp(x, self.model_scales[name], [0, 1])

  def _send_cur_state(self):
    if self.mpc_id == 1 and self.pm is not None:
      dat = messaging.new_message()
      dat.init('dynamicFollowData')
      dat.dynamicFollowData.mpcTR = self.TR
      dat.dynamicFollowData.profilePred = self.model_profile
      self.pm.send('dynamicFollowData', dat)

  def _change_cost(self, libmpc):
    TRs = [0.9, 1.8, 2.7]
    costs = [1.15, 0.15, 0.05]
    cost = interp(self.TR, TRs, costs)

    change_time = sec_since_boot() - self.profile_change_time
    change_time_x = [0, 0.5, 4]  # for three seconds after effective profile has changed
    change_mod_y = [3, 6, 1]  # multiply cost by multiplier to quickly change distance
    if change_time < change_time_x[-1]:  # if profile changed in last 3 seconds
      cost_mod = interp(change_time, change_time_x, change_mod_y)
      cost_mod_speeds = [0, 20 * CV.MPH_TO_MS]  # don't change cost too much under 20 mph
      cost_mods = [cost_mod * 0.01, cost_mod]
      cost *= interp(cost_mod, cost_mod_speeds, cost_mods)

    if self.last_cost != cost:
      libmpc.change_costs(MPC_COST_LONG.TTC, cost, MPC_COST_LONG.ACCELERATION, MPC_COST_LONG.JERK)  # todo: jerk is the derivative of acceleration, could tune that
      self.last_cost = cost

  def _store_df_data(self):
    cur_time = sec_since_boot()
    # Store custom relative accel over time
    if self.lead_data.status:
      if self.lead_data.new_lead:
        self.df_data.v_rels = []  # reset when new lead
      else:
        self.df_data.v_rels = self._remove_old_entries(self.df_data.v_rels, cur_time, self.v_rel_retention)
      self.df_data.v_rels.append({'v_ego': self.car_data.v_ego, 'v_lead': self.lead_data.v_lead, 'time': cur_time})

    # Store our velocity for better sng
    self.df_data.v_egos = self._remove_old_entries(self.df_data.v_egos, cur_time, self.v_ego_retention)
    self.df_data.v_egos.append({'v_ego': self.car_data.v_ego, 'time': cur_time})

    # Store data for auto-df model
    self.auto_df_model_data.append([self._norm(self.car_data.v_ego, 'v_ego'),
                                    self._norm(self.lead_data.v_lead, 'v_lead'),
                                    self._norm(self.lead_data.a_lead, 'a_lead'),
                                    self._norm(self.lead_data.x_lead, 'x_lead')])
    while len(self.auto_df_model_data) > self.model_input_len:
      del self.auto_df_model_data[0]

  def _get_pred(self):
    cur_time = sec_since_boot()
    if self.car_data.cruise_enabled and self.lead_data.status:
      if cur_time - self.last_predict_time > self.predict_rate:
        if len(self.auto_df_model_data) == self.model_input_len:
          pred = predict(np.array(self.auto_df_model_data[::self.skip_every], dtype=np.float32).flatten())
          self.last_predict_time = cur_time
          self.model_profile = int(np.argmax(pred))

  def _remove_old_entries(self, lst, cur_time, retention):
    return [sample for sample in lst if cur_time - sample['time'] <= retention]

  def _relative_accel_mod(self):
    """
    Returns relative acceleration mod calculated from list of lead and ego velocities over time (longer than 1s)
    If min_consider_time has not been reached, uses lead accel and ego accel from openpilot (kalman filtered)
    """
    a_ego = self.car_data.a_ego
    a_lead = self.lead_data.a_lead
    min_consider_time = 0.75  # minimum amount of time required to consider calculation
    if len(self.df_data.v_rels) > 0:  # if not empty
      elapsed_time = self.df_data.v_rels[-1]['time'] - self.df_data.v_rels[0]['time']
      if elapsed_time > min_consider_time:
        a_ego = (self.df_data.v_rels[-1]['v_ego'] - self.df_data.v_rels[0]['v_ego']) / elapsed_time
        a_lead = (self.df_data.v_rels[-1]['v_lead'] - self.df_data.v_rels[0]['v_lead']) / elapsed_time

    mods_x = [-1.5, -.75, 0]
    mods_y = [1, 1.25, 1.3]
    if a_lead < 0:  # more weight to slight lead decel
      a_lead *= interp(a_lead, mods_x, mods_y)

    if a_lead - a_ego > 0:  # return only if adding distance
      return 0

    rel_x = [-2.6822, -1.7882, -0.8941, -0.447, -0.2235, 0.0, 0.2235, 0.447, 0.8941, 1.7882, 2.6822]
    mod_y = [0.3245 * 1.1, 0.277 * 1.08, 0.11075 * 1.06, 0.08106 * 1.045, 0.06325 * 1.035, 0.0, -0.09, -0.09375, -0.125, -0.3, -0.35]
    return interp(a_lead - a_ego, rel_x, mod_y)

  def global_profile_mod(self, profile_mod_x, profile_mod_pos, profile_mod_neg, x_vel, y_dist):
    """
    This function modifies the y_dist list used by dynamic follow in accordance with global_df_mod
    It also intelligently adjusts the profile mods at each breakpoint based on the change in TR
    """
    if self.global_df_mod == 1.:
      return profile_mod_pos, profile_mod_neg, y_dist
    global_df_mod = 1 - self.global_df_mod

    # Calculate new TRs
    speeds = [0, self.sng_speed, 18, x_vel[-1]]  # [0, 18 mph, ~40 mph, highest profile mod speed (~78 mph)]
    mods = [0, 0.1, 0.7, 1]  # how much to limit global_df_mod at each speed, 1 is full effect
    y_dist_new = [y - (y * global_df_mod * interp(x, speeds, mods)) for x, y in zip(x_vel, y_dist)]

    # Calculate how to change profile mods based on change in TR
    # eg. if df mod is 0.7, then increase positive mod and decrease negative mod
    calc_profile_mods = [(interp(mod_x, x_vel, y_dist) - interp(mod_x, x_vel, y_dist_new) + 1) for mod_x in profile_mod_x]
    profile_mod_pos = [mod_pos * mod for mod_pos, mod in zip(profile_mod_pos, calc_profile_mods)]
    profile_mod_neg = [mod_neg * ((1 - mod) + 1) for mod_neg, mod in zip(profile_mod_neg, calc_profile_mods)]

    return profile_mod_pos, profile_mod_neg, y_dist_new

  def _get_TR(self):
    x_vel = [0.0, 1.8627, 3.7253, 5.588, 7.4507, 9.3133, 11.5598, 13.645, 22.352, 31.2928, 33.528, 35.7632, 40.2336]  # velocities
    profile_mod_x = [5 * CV.MPH_TO_MS, 30 * CV.MPH_TO_MS, 55 * CV.MPH_TO_MS, 80 * CV.MPH_TO_MS]  # profile mod speeds

    if self.df_manager.is_auto:  # decide which profile to use, model profile will be updated before this
      df_profile = self.model_profile
    else:
      df_profile = self.user_profile

    if df_profile != self.last_effective_profile:
      self.profile_change_time = sec_since_boot()
    self.last_effective_profile = df_profile

    if df_profile == self.df_profiles.roadtrip:
      y_dist = [1.5486, 1.556, 1.5655, 1.5773, 1.5964, 1.6246, 1.6715, 1.7057, 1.7859, 1.8542, 1.8697, 1.8833, 1.8961]  # TRs
      profile_mod_pos = [0.5, 0.35, 0.1, 0.03]
      profile_mod_neg = [1.3, 1.4, 1.8, 2.0]
    elif df_profile == self.df_profiles.traffic:  # for in congested traffic
      x_vel = [0.0, 1.892, 3.7432, 5.8632, 8.0727, 10.7301, 14.343, 17.6275, 22.4049, 28.6752, 34.8858, 40.35]
      y_dist = [1.3781, 1.3791, 1.3457, 1.3134, 1.3145, 1.318, 1.3485, 1.257, 1.144, 0.979, 0.9461, 0.9156]
      profile_mod_pos = [1.075, 1.55, 2.6, 3.75]
      profile_mod_neg = [0.95, .275, 0.1, 0.05]
    elif df_profile == self.df_profiles.relaxed:  # default to relaxed/stock
      y_dist = [1.385, 1.394, 1.406, 1.421, 1.444, 1.474, 1.521, 1.544, 1.568, 1.588, 1.599, 1.613, 1.634]
      profile_mod_pos = [1.0, 0.955, 0.898, 0.905]
      profile_mod_neg = [1.0, 1.0825, 1.1877, 1.174]
    else:
      raise Exception('Unknown profile type: {}'.format(df_profile))

    # Global df mod
    profile_mod_pos, profile_mod_neg, y_dist = self.global_profile_mod(profile_mod_x, profile_mod_pos, profile_mod_neg, x_vel, y_dist)

    # Profile modifications - Designed so that each profile reacts similarly to changing lead dynamics
    profile_mod_pos = interp(self.car_data.v_ego, profile_mod_x, profile_mod_pos)
    profile_mod_neg = interp(self.car_data.v_ego, profile_mod_x, profile_mod_neg)

    if self.car_data.v_ego > self.sng_speed:  # keep sng distance until we're above sng speed again
      self.sng = False

    if (self.car_data.v_ego >= self.sng_speed or self.df_data.v_egos[0]['v_ego'] >= self.car_data.v_ego) and not self.sng:
      # if above 15 mph OR we're decelerating to a stop, keep shorter TR. when we reaccelerate, use sng_TR and slowly decrease
      TR = interp(self.car_data.v_ego, x_vel, y_dist)
    else:  # this allows us to get closer to the lead car when stopping, while being able to have smooth stop and go when reaccelerating
      self.sng = True
      x = [self.sng_speed * 0.7, self.sng_speed]  # decrease TR between 12.6 and 18 mph from 1.8s to defined TR above at 18mph while accelerating
      y = [self.sng_TR, interp(self.sng_speed, x_vel, y_dist)]
      TR = interp(self.car_data.v_ego, x, y)

    TR_mods = []
    # Dynamic follow modifications (the secret sauce)
    x = [-26.8224, -20.0288, -15.6871, -11.1965, -7.8645, -4.9472, -3.0541, -2.2244, -1.5045, -0.7908, -0.3196, 0.0, 0.5588, 1.3682, 1.898, 2.7316, 4.4704]  # relative velocity values
    y = [.76, 0.62323, 0.49488, 0.40656, 0.32227, 0.23914*1.025, 0.12269*1.05, 0.10483*1.075, 0.08074*1.15, 0.04886*1.25, 0.0072*1.075, 0.0, -0.05648, -0.0792, -0.15675, -0.23289, -0.315]  # modification values
    TR_mods.append(interp(self.lead_data.v_lead - self.car_data.v_ego, x, y))

    x = [-4.4795, -2.8122, -1.5727, -1.1129, -0.6611, -0.2692, 0.0, 0.1466, 0.5144, 0.6903, 0.9302]  # lead acceleration values
    y = [0.24, 0.16, 0.092, 0.0515, 0.0305, 0.022, 0.0, -0.0153, -0.042, -0.053, -0.059]  # modification values
    TR_mods.append(interp(self.lead_data.a_lead, x, y))

    # deadzone = self.car_data.v_ego / 3  # 10 mph at 30 mph  # todo: tune pedal to react similarly to without before adding/testing this
    # if self.lead_data.v_lead - deadzone > self.car_data.v_ego:
    #   TR_mods.append(self._relative_accel_mod())

    x = [self.sng_speed, self.sng_speed / 5.0]  # as we approach 0, apply x% more distance
    y = [1.0, 1.05]
    profile_mod_pos *= interp(self.car_data.v_ego, x, y)  # but only for currently positive mods

    TR_mod = sum([mod * profile_mod_neg if mod < 0 else mod * profile_mod_pos for mod in TR_mods])  # alter TR modification according to profile
    TR += TR_mod

    if (self.car_data.left_blinker or self.car_data.right_blinker) and df_profile != self.df_profiles.traffic:
      x = [8.9408, 22.352, 31.2928]  # 20, 50, 70 mph
      y = [1.0, .75, .65]
      TR *= interp(self.car_data.v_ego, x, y)  # reduce TR when changing lanes

    return float(clip(TR, self.min_TR, 2.7))

  def update_lead(self, v_lead=None, a_lead=None, x_lead=None, status=False, new_lead=False):
    self.lead_data.v_lead = v_lead
    self.lead_data.a_lead = a_lead
    self.lead_data.x_lead = x_lead

    self.lead_data.status = status
    self.lead_data.new_lead = new_lead

  def _update_car(self, CS):
    self.car_data.v_ego = CS.vEgo
    self.car_data.a_ego = CS.aEgo

    self.car_data.left_blinker = CS.leftBlinker
    self.car_data.right_blinker = CS.rightBlinker
    self.car_data.cruise_enabled = CS.cruiseState.enabled

  def _get_live_params(self):
    self.global_df_mod = self.op_params.get('global_df_mod')
    if self.global_df_mod != 1.:
      self.global_df_mod = clip(self.global_df_mod, 0.85, 1.2)

    self.min_TR = self.op_params.get('min_TR')
    if self.min_TR != 1.:
      self.min_TR = clip(self.min_TR, 0.85, 1.6)
