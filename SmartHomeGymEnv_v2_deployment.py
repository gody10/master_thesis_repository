import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np
from math import *
import matplotlib.pyplot as plt

class SmartHomeGymEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        data_file,
        priority_n_devices,
        priority_device_power_consumption,
        priority_user_probabilities,
        deferrable_n_devices=0, # Create an env for deferrable 0
        deferrable_device_power_consumption=np.array([]),
        deferrable_device_on_duration=np.array([]), # in hours ??? 
        episode_horizon=365, # the horizon of each episode in days 
        deferrable_user_probabilities=np.array([]),
        time_step_duration=1 # Time step duration in hours (default value is 1, i.e. each time step represents one hour)
    ):

        super(SmartHomeGymEnv,self).__init__()

        # Setting the power consumption of each device that is registered
        # It should have the form of : np.array([1.0,1.0,1.0,1.0,1.0]) meaning 5 devices with 1 kW consumption
        # We are also checking if the dimensions match with what was give as number of devices

        #assert(priority_n_devices>0 and deferrable_n_devices>0), "Number of devices must be greater than 0"

        assert(priority_device_power_consumption.shape[0]==priority_n_devices), "Wrong dimensions in Priority Device Power Consumption array"

        #assert(deferrable_device_power_consumption.shape[0]==deferrable_n_devices), "Wrong dimensions in Deferrable Device Power Consumption array"

        assert(100%(time_step_duration*100)==0 and time_step_duration<=1), "Cannot set this time step duration"
 
        # Setting the consumption of each priority device that is registered
        self.priority_device_power_consumption = priority_device_power_consumption

        # Setting the consumption of each deferrable device that is registered
        self.deferrable_device_power_consumption = deferrable_device_power_consumption

        # Loading the csv file
        self.data = pd.read_csv(data_file)
        
        # Representing each moment as an hour of the day
        time = []
        for i in range(len(self.data)):
            time.append(i % (24//time_step_duration))

        # Setting the number of moments that we want to keep the same price based on the time step duration that the user has given
        self.moments_to_keep_same_price = 1//time_step_duration

        # Assigning the time array to a variable
        self.time = time

        # Assigning the time step duration to a variable
        self.time_step_duration = time_step_duration

        # Getting data from the csv file that are needed for the environment
        energy_price = self.data.loc[:,'Price (cents per kWh)']

        # Repeating the same price for the moments that we want to keep the same price
        self.energy_price = pd.DataFrame(np.repeat(energy_price.values, self.moments_to_keep_same_price, axis=0))
        
        # Setting the number of priority devices that are going to exist in this Smart Home
        self.priority_n_devices = priority_n_devices

        # Setting the number of deferrable devices that are going to exist in this Smart Home
        self.deferrable_n_devices = deferrable_n_devices

        # Setting the duration of the deferrable devices
        self.deferrable_device_on_duration = deferrable_device_on_duration

        # Changing the duration based on the time step duration
        self.deferrable_device_on_duration = self.deferrable_device_on_duration * (1//time_step_duration)
        
         # Create an array that will countdown the activation of the deferrable devices
        self.deferrable_device_on_countdown = np.zeros(self.deferrable_n_devices, dtype=float)

        # Setting the episode length
        # 24 for one day, 168 for one week and so on (if time_step_duration=1)
        self.max_steps = episode_horizon * 24 * (1//time_step_duration)

        # Setting the priority user transition probabilities
        self.priority_user_transition_probabilites = priority_user_probabilities

        # Setting the deferrable user transition probabilities
        self.deferrable_user_transition_probabilites = deferrable_user_probabilities

        self.deferrable_penalty = 50  # defferable device penalty hyperparameter
        self.standard_penalty = 100  # standard device penalty hyperparameter

        # Setting a variable that will discourage the agent to activate diferrable devices more than a set of times in a day

        if (self.deferrable_n_devices!=0):
            # The action space includes the possible actions, meaning all the possible combinations of on/off devices that the user has registered
            self.action_space = spaces.Dict({
                'priority_devices' : spaces.Box(low=0, high=1, shape=(priority_n_devices,)),
                'deferrable_devices' : spaces.Box(low=0, high=1, shape=(deferrable_n_devices,)),
            })
        
            # The observation space includes the knowledge of which devices are on or off and the energy price of the current timestep 
            self.observation_space = spaces.Dict({
                'priority_devices': spaces.Box(low=0, high=1, shape=(priority_n_devices,)),
                'deferrable_devices': spaces.Box(low=0, high=1, shape=(deferrable_n_devices,)),
                'deferrable_devices_duration_countdown' : spaces.Box(low=0, high=max(self.deferrable_device_on_duration), shape=(deferrable_n_devices,)), # We want the agent to see how much time is left for each deferrable device
                'energy_price' : spaces.Box(low=-10000, high=10000, dtype=float),
                'time' : spaces.Box(low=0, high=24//time_step_duration, dtype=float)
            })

        else:
            # The action space includes the possible actions, meaning all the possible combinations of on/off devices that the user has registered
            self.action_space = spaces.Box(low=0, high=1, shape=(self.priority_n_devices,),dtype=float)
        
            # The observation space includes the knowledge of which devices are on or off and the energy price of the current timestep 
            self.observation_space = spaces.Dict({
                'devices': spaces.Box(low=0, high=1, shape=(self.priority_n_devices,)),
                'energy_price' : spaces.Box(low=-10000, high=10000,dtype=float),
                'time' : spaces.Box(low=0, high=24//time_step_duration, dtype=float)
            })

        # Calling the reset function in order to reset the environment
        self.reset()

        print("Environment successfully initialized")

        return None
    

    def state_transition(self, observation, action, user_transition_probabilites):

        # We round the action to 0 or 1 because we currently accept only On/Off states
        action = np.round(action)

        # We also round the observation
        observation = np.round(observation)

        # Create arrays to represent the different cases of the transition probabilities
        p1 = user_transition_probabilites[:, 0]
        p2 = user_transition_probabilites[:, 1]
        q1 = user_transition_probabilites[:, 2]
        q2 = user_transition_probabilites[:, 3]

        # Create boolean masks for the different cases of the transition probabilities
        on_to_on_mask = (action == 1) & (observation == 1)
        on_to_off_mask = (action == 0) & (observation == 1)
        off_to_off_mask = (action == 0) & (observation == 0)
        off_to_on_mask = (action == 1) & (observation == 0)

        # Initialize the new_states array
        new_states = np.zeros(user_transition_probabilites.shape[0], dtype=int)

        # Calculate the new state values using the transition probabilities and boolean masks
        new_states[on_to_on_mask] = np.random.binomial(1, p1[on_to_on_mask]) # The user accepts an ON request when the device is ON with probability p1 or rejects it 
        # with probability 1-p1
        new_states[on_to_off_mask] = np.random.binomial(1, p2[on_to_off_mask]) # The user accepts an OFF request when the device is ON with probability 1 - p2 or rejects it 
        # with probability p2
        new_states[off_to_off_mask] = np.random.binomial(1, 1 - q1[off_to_off_mask]) # The user accepts an OFF request when the device is OFF with probability q1 or rejects it 
        # with probability 1 - q1
        new_states[off_to_on_mask] = np.random.binomial(1, 1 - q2[off_to_on_mask]) # The user accepts an ON request when the device is OFF with probability 1 - q2 or rejects it 
        # with probability q2

        return new_states
    
    def step(self, action):

        # We advance the time by one timestep
        self.current_time += 1

        if(self.deferrable_n_devices!=0):

            # We save the old state of the deferrable devices
            self.old_deferrable_state = self.observation['deferrable_devices']

            # We calculate the device state, i.e. we see which changes the user accepts
            self.priority_devices_now = self.state_transition(self.observation['priority_devices'], action['priority_devices'], self.priority_user_transition_probabilites)
            self.deferrable_devices_now = self.state_transition(self.observation['deferrable_devices'], action['deferrable_devices'], self.deferrable_user_transition_probabilites)
            
            self.temp_deferrable_countdown = self.deferrable_device_on_countdown.copy()

            # Update the countdown for each deferrable device
            for i in range(len(self.observation['deferrable_devices'])):
                if self.deferrable_devices_now[i] == 1:
                    if self.old_deferrable_state[i] == 0:  # If the device was off and now is on
                        self.temp_deferrable_countdown[i] = self.deferrable_device_on_duration[i]
                    self.temp_deferrable_countdown[i] -= 1  # Decrease the countdown
                    if (self.temp_deferrable_countdown[i]<=0):
                        self.temp_deferrable_countdown[i] = 0
                        self.deferrable_devices_now[i] = 0
                    if self.temp_deferrable_countdown[i] <= 0:
                        self.temp_deferrable_countdown[i] = 0
                        self.deferrable_devices_now[i] = 0
                #elif self.deferrable_devices_now[i] == 0:
                    #self.deferrable_device_on_countdown[i] = 0

            # Get the energy price of the current timestep
            self.energy_price_now = self.energy_price.iloc[self.current_time][0]
            
            deferrable_reward = np.sum( # Do we want the user to decline the suggestion to turn it off when the countdown is not 0? Should it be action['deferrable_devices'] or self.deferrable_devices_now?   
                np.where(
                    (np.round(self.old_deferrable_state) == 1) & (np.round(action['deferrable_devices']) == 0) & (self.observation['deferrable_devices_duration_countdown'] > 0),
                    -self.deferrable_penalty,
                    np.where(
                        np.round(self.deferrable_devices_now) == np.round(action['deferrable_devices']), 
                            -self.energy_price_now * self.deferrable_device_power_consumption * self.deferrable_devices_now * self.time_step_duration,
                            -self.standard_penalty
                    )
                )
            )
            
            priority_reward = np.sum(
                np.where(
                    np.round(self.priority_devices_now) == np.round(action['priority_devices']), 
                        -self.energy_price_now * self.priority_device_power_consumption * self.priority_devices_now * self.time_step_duration, 
                        -self.standard_penalty 
                )
            )
            
            self.reward = priority_reward + deferrable_reward

            

            # Set the new Time
            self.time_now = self.time[self.current_time]
           
            
            self.observation = {
                'priority_devices': np.array(self.priority_devices_now), 
                'deferrable_devices': np.array(self.deferrable_devices_now),
                'deferrable_devices_duration_countdown' :  np.array(self.temp_deferrable_countdown),
                'energy_price' : np.array([self.energy_price_now]),
                'time' : np.array([self.time_now])
            }
            #print(self.time_now)
            #print(self.energy_price_now)

            
            # print(self.old_deferrable_state)
            # print("#########")
            # print(self.observation['deferrable_devices'])
            # print("#########")
            # print(self.observation['deferrable_devices_duration_countdown'])
            # print("Next")
            
            # Calculating infos
            self.priority_device_reward_counter += priority_reward
            self.deferrable_device_reward_counter += deferrable_reward

            priority_kwh_consumption = np.sum(np.where(np.round(self.priority_devices_now)==1,self.priority_device_power_consumption,0))
            deferrable_kwh_consumption = np.sum(np.where(np.round(self.deferrable_devices_now)==1,self.deferrable_device_power_consumption,0))

            self.plot_priority_kwh.append(priority_kwh_consumption)
            self.plot_deferrable_kwh.append(deferrable_kwh_consumption)

            current_kwh_consumption = priority_kwh_consumption + deferrable_kwh_consumption

        else:
            # We calculate the device state, i.e. we see which changes the user accepts
            self.priority_devices_now = self.state_transition(self.observation['devices'], action, self.priority_user_transition_probabilites)

            # Get the energy price of the current timestep
            self.energy_price_now = self.energy_price.iloc[self.current_time][0]

            self.reward = np.sum(np.where(np.round(self.priority_devices_now) == np.round(action), ( -self.energy_price_now * self.priority_device_power_consumption * self.devices_now* self.time_step_duration), -self.standard_penalty ))

            # priority_reward = np.sum(
            #     np.where(
            #         np.round(self.priority_devices_now) == np.round(action['priority_devices']), 
            #             -self.energy_price_now * self.priority_device_power_consumption * self.priority_devices_now * self.time_step_duration, 
            #             -self.standard_penalty 
            #     )
            # )

            current_kwh_consumption = np.sum(np.where(np.round(self.priority_devices_now)==1,self.priority_device_power_consumption*self.time_step_duration,0))

            

            # Set the new Time
            self.time_now = self.time[self.current_time]

            self.observation = {
            'devices': np.array(self.priority_devices_now), 
            'energy_price' : np.array([self.energy_price_now]),
            'time' : np.array([self.time_now])
            }

        #print(self.current_time)
        # For Plot
        
        self.plot_price.append(self.energy_price_now)
        self.plot_kwh.append(current_kwh_consumption)
        self.plot_time.append(self.current_time)

        #print(self.plot_kwh)
        #print(self.observation)

        # If we have reached the end of the episode horizon
        if self.current_time >= self.max_steps:

            self.done = True

            if(self.deferrable_n_devices!=0):
            # We set the done flag as True to indicate that that the episode ended
                info = {"priority_device_reward_total" : self.priority_device_reward_counter,
                    "deferrable_device_reward_total" : self.deferrable_device_reward_counter,
                    "kwh_device_history" : self.plot_kwh,
                    "kwh_priority_device_history" : self.plot_priority_kwh,
                    "kwh_deferrable_device_history" : self.plot_deferrable_kwh,
                    "price_history" : self.plot_price,
                    "time" : self.plot_time}
            else:
                info = {"kwh_device_history" : self.plot_kwh,
                    "price_history" : self.plot_price,
                    "time" : self.plot_time}
        else:

            if(self.deferrable_n_devices!=0):
                info = {"priority_device_reward_total" : self.priority_device_reward_counter,
                    "deferrable_device_reward_total" : self.deferrable_device_reward_counter,
                    "kwh_device_history" : self.plot_kwh,
                    "kwh_priority_device_history" : self.plot_priority_kwh,
                    "kwh_deferrable_device_history" : self.plot_deferrable_kwh,
                    "price_history" : self.plot_price,
                    "time" : self.plot_time}
            else:
                info = {"kwh_device_history" : self.plot_kwh,
                    "price_history" : self.plot_price,
                    "time" : self.plot_time}
        
        truncated = False

        #print(action)
        #print(self.observation)
        
        return self.observation, float(self.reward), self.done, truncated, info
    
    def reset(self, *, seed=None, options=None):
        
        if (self.deferrable_n_devices!=0):
            # Setting time at 0
            self.current_time = 0
            # Setting time at 0
            self.moment = 0
            # Setting reward at 0
            self.reward = 0
            # Setting done as False in order to declare that the environment has not reached the end
            self.done = False

            # Initializing the first state with all devices off
            self.priority_devices_now = np.zeros(self.priority_n_devices)
            self.deferrable_devices_now = np.zeros(self.deferrable_n_devices)

            # Setting the first energy price
            self.energy_price_now = self.energy_price.iloc[self.current_time][0]

            # Setting the current time
            self.time_now = self.time[self.current_time]
            
            # Setting up counters for info dictionary variables
            self.priority_device_reward_counter = 0
            self.deferrable_device_reward_counter = 0

            # Setting up the current observation
            self.observation = {
                'priority_devices': np.array(self.priority_devices_now),
                'deferrable_devices': np.array(self.deferrable_devices_now),
                'deferrable_devices_duration_countdown' :  np.array(self.deferrable_device_on_countdown),
                'energy_price' : np.array([self.energy_price_now]),
                'time' : np.array([self.time_now])
            }
        
        else:
            # Setting time at 0
            self.current_time = 0
            # Setting reward at 0
            self.reward = 0
            # Setting done as False in order to declare that the environment has not reached the end
            self.done = False

            # Setting the current time
            self.time_now = self.time[self.current_time]

            # Initializing the first state with all devices off
            self.devices_now = np.zeros(self.priority_n_devices)

            self.energy_price_now = self.energy_price.iloc[self.current_time][0]
            
            # Setting up the current observation
            self.observation = {
                'devices': np.array(self.devices_now),
                'energy_price' : np.array([self.energy_price_now]),
                'time' : np.array([self.time_now])
            }

        self.plot_price = []
        self.plot_kwh = []
        self.plot_priority_kwh = []
        self.plot_deferrable_kwh = []
        self.plot_time = []
        
        if(self.deferrable_n_devices!=0):
            # We set the done flag as True to indicate that that the episode ended
            info = {"priority_device_reward_total" : self.priority_device_reward_counter,
                "deferrable_device_reward_total" : self.deferrable_device_reward_counter,
                "kwh_device_history" : self.plot_kwh,
                "price_history" : self.plot_price,
                "time" : self.plot_time}
        else:
            info = {"kwh_device_history" : self.plot_kwh,
                "price_history" : self.plot_price,
                "time" : self.plot_time}

        return self.observation, info

def main():

    def_n_device = 1


    if(def_n_device!=0):

        priority_user_prob = np.array([
                np.array([0.7, 0.11, 0.31, 0.61]),        
                np.array([0.7, 0.11, 0.31, 0.61]), 
                np.array([0.7, 0.11, 0.31, 0.61]), 
                np.array([0.7, 0.11, 0.31, 0.61]), 
                np.array([0.7, 0.11, 0.31, 0.61])
            ])
        
        deferrable_user_prob = np.array([
                np.array([0.2, 0.11, 0.31, 0]),        
                np.array([0.7, 0.11, 0.31, 0]), 
            ])

        smart_env2 = SmartHomeGymEnv(data_file='../nyiso_hourly_prices.csv',priority_n_devices=5,priority_device_power_consumption=np.array([1.0,1.0,1.0,1.0,1.0]),
                                    deferrable_n_devices=2,deferrable_device_power_consumption=np.array([1.0,1.0]),
                                    deferrable_device_on_duration=np.array([2.0,1.5]),
                                    episode_horizon=7,priority_user_probabilities=priority_user_prob, deferrable_user_probabilities=deferrable_user_prob,time_step_duration=0.5)
        
        smart_env2.step({'deferrable_devices' : [0.51,0.611], 'priority_devices' : [0.8527126,0.8528148, 0.3818789, 0.4139895, 0.4671028]})
        smart_env2.step({'deferrable_devices' : [0.4,0.211], 'priority_devices' : [0.8527126,0.8528148, 0.3818789, 0.4139895, 0.4671028]})
        smart_env2.step({'deferrable_devices' : [0.4,0.211], 'priority_devices' : [0.8527126,0.8528148, 0.3818789, 0.4139895, 0.4671028]})
        smart_env2.step({'deferrable_devices' : [0.6,0.611], 'priority_devices' : [0.8527126,0.8528148, 0.3818789, 0.4139895, 0.4671028]})
        smart_env2.step({'deferrable_devices' : [0.6,0.611], 'priority_devices' : [0.8527126,0.8528148, 0.3818789, 0.4139895, 0.4671028]})
        smart_env2.step({'deferrable_devices' : [0.6,0.611], 'priority_devices' : [0.8527126,0.8528148, 0.3818789, 0.4139895, 0.4671028]})
        smart_env2.step({'deferrable_devices' : [0.51,0.611], 'priority_devices' : [0.8527126,0.8528148, 0.3818789, 0.4139895, 0.4671028]})
        smart_env2.step({'deferrable_devices' : [0.4,0.211], 'priority_devices' : [0.8527126,0.8528148, 0.3818789, 0.4139895, 0.4671028]})
        smart_env2.step({'deferrable_devices' : [0.4,0.211], 'priority_devices' : [0.8527126,0.8528148, 0.3818789, 0.4139895, 0.4671028]})
        smart_env2.step({'deferrable_devices' : [0.6,0.611], 'priority_devices' : [0.8527126,0.8528148, 0.3818789, 0.4139895, 0.4671028]})
        smart_env2.step({'deferrable_devices' : [0.6,0.611], 'priority_devices' : [0.8527126,0.8528148, 0.3818789, 0.4139895, 0.4671028]})
        smart_env2.step({'deferrable_devices' : [0.6,0.611], 'priority_devices' : [0.8527126,0.8528148, 0.3818789, 0.4139895, 0.4671028]})
        smart_env2.step({'deferrable_devices' : [0.51,0.611], 'priority_devices' : [0.8527126,0.8528148, 0.3818789, 0.4139895, 0.4671028]})
        smart_env2.step({'deferrable_devices' : [0.4,0.211], 'priority_devices' : [0.8527126,0.8528148, 0.3818789, 0.4139895, 0.4671028]})
        smart_env2.step({'deferrable_devices' : [0.4,0.211], 'priority_devices' : [0.8527126,0.8528148, 0.3818789, 0.4139895, 0.4671028]})
        smart_env2.step({'deferrable_devices' : [0.6,0.611], 'priority_devices' : [0.8527126,0.8528148, 0.3818789, 0.4139895, 0.4671028]})
        smart_env2.step({'deferrable_devices' : [0.6,0.611], 'priority_devices' : [0.8527126,0.8528148, 0.3818789, 0.4139895, 0.4671028]})
        smart_env2.step({'deferrable_devices' : [0.6,0.611], 'priority_devices' : [0.8527126,0.8528148, 0.3818789, 0.4139895, 0.4671028]})
        smart_env2.step({'deferrable_devices' : [0.51,0.611], 'priority_devices' : [0.8527126,0.8528148, 0.3818789, 0.4139895, 0.4671028]})
        smart_env2.step({'deferrable_devices' : [0.4,0.211], 'priority_devices' : [0.8527126,0.8528148, 0.3818789, 0.4139895, 0.4671028]})
        smart_env2.step({'deferrable_devices' : [0.4,0.211], 'priority_devices' : [0.8527126,0.8528148, 0.3818789, 0.4139895, 0.4671028]})
        smart_env2.step({'deferrable_devices' : [0.6,0.611], 'priority_devices' : [0.8527126,0.8528148, 0.3818789, 0.4139895, 0.4671028]})
        smart_env2.step({'deferrable_devices' : [0.6,0.611], 'priority_devices' : [0.8527126,0.8528148, 0.3818789, 0.4139895, 0.4671028]})
        smart_env2.step({'deferrable_devices' : [0.6,0.611], 'priority_devices' : [0.8527126,0.8528148, 0.3818789, 0.4139895, 0.4671028]})
        smart_env2.step({'deferrable_devices' : [0.6,0.611], 'priority_devices' : [0.8527126,0.8528148, 0.3818789, 0.4139895, 0.4671028]})
        smart_env2.step({'deferrable_devices' : [0.6,0.611], 'priority_devices' : [0.8527126,0.8528148, 0.3818789, 0.4139895, 0.4671028]})
        smart_env2.step({'deferrable_devices' : [0.6,0.611], 'priority_devices' : [0.8527126,0.8528148, 0.3818789, 0.4139895, 0.4671028]})
        smart_env2.step({'deferrable_devices' : [0.6,0.611], 'priority_devices' : [0.8527126,0.8528148, 0.3818789, 0.4139895, 0.4671028]})
        smart_env2.step({'deferrable_devices' : [0.6,0.611], 'priority_devices' : [0.8527126,0.8528148, 0.3818789, 0.4139895, 0.4671028]})
        smart_env2.step({'deferrable_devices' : [0.6,0.611], 'priority_devices' : [0.8527126,0.8528148, 0.3818789, 0.4139895, 0.4671028]})
        smart_env2.step({'deferrable_devices' : [0.6,0.611], 'priority_devices' : [0.8527126,0.8528148, 0.3818789, 0.4139895, 0.4671028]})
        smart_env2.step({'deferrable_devices' : [0.6,0.611], 'priority_devices' : [0.8527126,0.8528148, 0.3818789, 0.4139895, 0.4671028]})
        smart_env2.step({'deferrable_devices' : [0.6,0.611], 'priority_devices' : [0.8527126,0.8528148, 0.3818789, 0.4139895, 0.4671028]})

        
        
        

    else:

        user_prob = np.array([
                np.array([0.7, 0.11, 0.31, 0.61]),        
                np.array([0.7, 0.11, 0.31, 0.61]), 
                np.array([0.7, 0.11, 0.31, 0.61]), 
                np.array([0.7, 0.11, 0.31, 0.61]), 
                np.array([0.7, 0.11, 0.31, 0.61])
            ])
        
        env = SmartHomeGymEnv(data_file='../nyiso_hourly_prices.csv',priority_n_devices=5,priority_device_power_consumption=np.array([1.0,1.0,1.0,1.0,1.0]),
                                    deferrable_n_devices=0,deferrable_device_power_consumption=np.array([]),
                                    deferrable_device_on_duration=np.array([]),
                                    episode_horizon=7,priority_user_probabilities=user_prob, deferrable_user_probabilities=[],time_step_duration=0.5)
        env.step([0.8527126,0.8528148, 0.8818789, 0.8139895, 0.8671028])
        env.step([0.8527126,0.8528148, 0.3818789, 0.4139895, 0.4671028])
        env.step([0.8527126,0.8528148, 0.3818789, 0.4139895, 0.4671028])
        env.step([0.8527126,0.8528148, 0.3818789, 0.4139895, 0.4671028])
        env.step([0.8527126,0.8528148, 0.3818789, 0.4139895, 0.4671028])
        env.step([0.8527126,0.8528148, 0.3818789, 0.4139895, 0.4671028])
        env.step([0.8527126,0.8528148, 0.3818789, 0.4139895, 0.4671028])
        

if __name__ == "__main__":
    main()