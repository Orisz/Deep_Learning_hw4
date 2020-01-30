r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""


# ==============
# Part 1 answers


def part1_pg_hyperparams():
    hp = dict(batch_size=16,
              gamma=0.95,
              beta=0.2,
              learn_rate=1e-3,
              eps=1e-8,
              )
    # TODO: Tweak the hyperparameters if needed.
    #  You can also add new ones if you need them for your model's __init__.
    # ====== YOUR CODE: ======
    #raise NotImplementedError()
    # ========================
    return hp


def part1_aac_hyperparams():
    hp = dict(batch_size=16,
              gamma=0.95,
              beta=0.5,
              delta=1.0,
              learn_rate=1e-3,
              eps=1e-8,
              )
    # TODO: Tweak the hyperparameters. You can also add new ones if you need
    #   them for your model implementation.
    # ====== YOUR CODE: ======
    #raise NotImplementedError()
    # ========================
    return hp


part1_q1 = r"""
We update the policy in the direction of received reward. The policy receive very different rewards for similar behavior. So the policy needs to collect a lot of experience to average over the varying rewards and converge to good behavior.
When we learn a value function of the current state, we can subtract that from the reward. What's left is the advantage, the amount of how much the current action is better than what we'd usually do in that state. The advantage has lower variance since the baseline compensates for the variance introduced by being in different states.
For example consider the following: Scenario 1) trajectory A recieves +5 reward and trajectory B recieves -1 reward. Scenraio 2) trajectory A recieves +6 reward and trajectory B recieves +1 reward.
In both cases we would like the increase probability for trajectoy A and lower the one for trajectroy B. but since in case 2 trajectory B get positive reward this will no be the case. So we gather we need to center the rewards around some 'mean' in order to reduce such variance as the one just given as an example.

"""


part1_q2 = r"""
$v_{\pi}(s)$ is defined as average over the first action according to the policy. So we can think on the state-value function as an average over the actions of the action-value function($q_{\pi}(s,a)$). Now recalling that once obtain the actions scores we sample from them if we sample many times eventually we will get the result over the mean action. I.e. sampling many times from the policy net relates $v_{\pi}(s)$ to $q_{\pi}(s,a)$.

"""


part1_q3 = r"""
1)
There were no significant chages between the graphes on the first experiment. For the loss_p all managed to get to zero loss after some time. To the vpg and epg it took longer so we may suspect the the baseline had something to do with it. bpg and cpg(both using baseline) managed to get to same baseline loss. bpg had some temporary divergence. This affected his mean reward but he managed to recover and got reward similar to the others.
The loss_e graph is the same. epg and cpg managed to minimize the loss to low values of -0.145~. This is wath we want regarding the exploration exploitation theorm. We don't want our agent to be too sure of his actions so he may try some new ones.
Finally accounting for the mean_reward graph. all got prety much the same outcome. The ratio of convergence was also similar(besides the bpg because of the temporary divergence).

2)
We had great difficulty training the aac. It seems that gamma and betta affect him great deal. We think it may stuck on some local minima of 'hovering over' the landing zone. This prevents him from getting the big reward of 'landing' which is significantly bigger than the other rewards in the game. The ways we tried to overcome this is by a) increasing gamma but it seems it had hard time accepting gammas bigger than 0.95. And b) by increasing betta meaning we may try different actions instead of maintaining 'hovering' mode.
Comparing to the other graphes, the loss (loss_p) seems to converge much fatster when using the aac framework. it seems that is also learns to achieve greater rewards much faster then other methods(see the slope of the 'mean_reward' graph). Still we can't explain why it does not doin much better when considering the total reward. A possible explanation is the 'shared' optimization meaning we should optimize the actor and the critic with different optimizers each configured seperatly. 
"""
