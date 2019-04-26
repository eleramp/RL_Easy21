#!/usr/bin/env/python
# -*- coding: utf-8 -*-

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np

class Easy21Env(gym.Env):

    def __init__(self):
        self.min_card_value, self.max_card_value = 1, 10
        self.dealer_ub = 17
        self.game_lb, self.game_ub = 0, 21

        # Action Space:
        #   0 = hits (draws another card)
        #   1 = sticks (no further cards)
        self.action_space = spaces.Discrete(2)
        # Observation Space:
        #   Player's sum [1, 21] and dealer's first card [1, 10]
        self.observation_space = spaces.Tuple((
            spaces.Discrete(self.game_ub+1),
            spaces.Discrete(self.max_card_value+1)))
        self.dim = (self.observation_space.spaces[0].n,
                    self.observation_space.spaces[1].n,
                    self.action_space.n)

        self.player=0
        self.dealer=0

    #draw a card
    def draw(self):
        card_value = np.random.randint(self.min_card_value,self.max_card_value+1)
        if np.random.random_sample() <= 1/3.0:
            return -card_value
        return card_value

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        done = 0
        # Hits?
        if action == 0:
            reward = 0
            self.player += self.draw()
            #Check if the sum is still valid: in [1,21] value range
            if not(self.game_lb < self.player <= self.game_ub):
                reward = -1
                #terminal state
                done = 1
        #Stick?
        elif action == 1:
            #the dealer start playing: hits while its sum is in [1,16] value range
            while self.game_lb < self.dealer < self.dealer_ub:
                self.dealer += self.draw()

                if not(self.game_lb < self.dealer <= self.game_ub) \
                or self.player > self.dealer:
                    reward = +1

                elif self.player == self.dealer:
                    reward = 0

                elif self.player < self.dealer:
                    reward = -1
            #terminal state
            done = 1
        return (self.player,self.dealer),reward,done,{}

    def reset(self):
        self.player = np.random.randint(self.min_card_value,self.max_card_value+1)
        self.dealer = np.random.randint(self.min_card_value,self.max_card_value+1)
        return (self.player, self.dealer)

    def render(self):
        pass

    def close(self):
        pass
