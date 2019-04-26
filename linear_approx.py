#!/usr/bin/env/python
# -*- coding: utf-8 -*-

import gym
import numpy as np
import dill as pickle
import utils
from config import approx_params as cf

## --- epsilon-greedy policy --- ##
def e_greedy_policy(s, theta, actions, epsilon):
    if np.random.random() < epsilon:
        #random action: exploration
        a = actions.sample()
    else:
        #greedy action: exploitation
        a = np.argmax([Q(s,a,theta) for a in range(actions.n)])
    return a
## --- ##

def reset(env_dim):
    # state-action value & counter
    theta = np.random.randn(3*6*2, 1) #cf['dealer_range_f']*cf['player_range_f']*actions.n
    wins = 0
    return theta, wins

def binary_features(s,a):
    f = np.zeros(3*6*2)
    for fi, (lower, upper) in enumerate(zip(range(1,8,3), range(4, 11, 3))):
        f[fi] = (lower <= s[1] <= upper)

    for fi, (lower, upper) in enumerate(zip(range(1,17,3), range(6, 22, 3)), start=3):
        f[fi] = (lower <= s[0] <= upper)

    f[-2] = 1 if a == 0 else 0
    f[-1] = 1 if a == 1 else 0

    return f.reshape(1, -1) #(1*n)-dim

def Q(s,a,theta):
    return np.dot(binary_features(s,a),theta)

def feature_all(env_dim):
    f_all = np.zeros((env_dim[0],env_dim[1],env_dim[2],3*6*2))
    for p in range(1,env_dim[0]):
        for d in range(1,env_dim[1]):
            for a in range(env_dim[2]):
                f_all[p-1,d-1,a] = binary_features((p,d),a)
    return f_all

def Q_all(f_all,theta):
    return np.dot(f_all.reshape(-1, 3*6*2),theta).reshape(-1)

def lin_approx_control(env,trueQ):

    actions = env.action_space
    lambdas = cf['lambdas']
    mselambdas = np.zeros((len(lambdas), cf['n_episodes']))
    finalMSE = np.zeros(len(lambdas))

    f_all = feature_all(env.dim)

    for i_lambda,lambda_decay in enumerate(lambdas):
        theta, wins = reset(env.dim)

        for episode in range(cf['n_episodes']): #terminal state for exploration?
            done = 0
            E = np.zeros_like(theta)

            #initial state and action
            s = env.reset()
            a = e_greedy_policy(s, theta, actions, cf['epsilon'])
            while not done:
                #Forward a step
                s_next,reward,done,_ = env.step(a)
                if done: # Sarsa lambda Update
                    td_error = reward - Q(s,a,theta)
                else: #Pick an action with e-greedy policy
                    a_next = e_greedy_policy(s_next, theta, actions, cf['epsilon'])
                    td_error = reward + cf['r_gamma']*Q(s_next,a_next,theta) - Q(s,a,theta)

                #update
                E = cf['r_gamma'] * lambda_decay * E + binary_features(s,a).reshape(-1,1)
                delta_theta = cf['step_size'] * td_error * E
                theta += delta_theta

                if not done:
                    s = s_next
                    a = a_next

             # bookkeeping
            if reward == 1:
                wins += 1

            mse = np.sum(np.square(Q_all(f_all,theta) - trueQ.ravel())) /  (env.dim[0]*env.dim[1]*env.dim[2])
            mselambdas[i_lambda, episode] = mse

        finalMSE[int(i_lambda)] = mse
        print("Lambda=%.1f Episode %06d, MSE %5.3f, Wins %.3f"%(lambda_decay, episode, mse, wins/(episode+1)))
        print("--------")

    utils.plotMseLambdas(finalMSE, lambdas)
    utils.plotMseEpisodesLambdas(mselambdas)

def main():
    env = gym.make("gym_easy21:easy21-v0")
    trueQ = pickle.load(open('Q.dill', 'rb'))
    lin_approx_control(env, trueQ)

if __name__ == "__main__":
    main()
