r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""


# ==============
# Part 1 answers


def part1_pg_hyperparams():
    hp = dict(batch_size=16,
              gamma=0.99,
              beta=0.5,
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
              gamma=0.99,
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
There was a huge difference between trainig graphes. It seems that using the baseline affects the training significantly, this is probably due to the decrease in the variance as explained in the notebook. bpg and cpg(both using baseline) managed to get to same baseline loss. cpg had some temporary divergence. This affected his mean reward but he managed to recover and got reward similar to the bpg. This has may caused due to the entropy loss since minimizing max-entropy-loss means we want our agent try to explore more and not always go on the safe side.
The loss_e graph shows how and cpg is applying the 'exploration' since there is an entropy loss throughout the process. Since epg didn't learn anything, he managed to get the entropy loss to zero which means he failed(zero loss always means something is wrong).
Finally accounting for the mean_reward graph. epg and vpg did bad. We duduce this is due to high variance in the training process. bpg and cpg did alot better! they managed to get rewards over 100. Still it seems that both of them were somehow 'unstable' we can blame the baseline chosen. b(the average) is some constatnt. It is much better to 'learn' the appropriate offset as seen with the aac...

2)
It seems that in order to train the aac with gamma=0.99 so the agent will have the motivation to do the landing we had to decrease the size of the model in half.
Comparing aac to cpg:
loss_p: aac managed to get the same loss as the cpg but it took him longer. Amore 'gentle' slope
loss_e: aac had largaer loss throughout the entire process. Here we can see how much the acc graph is smoother w.r.t the cpg one. This may has to do with the advantage function. Leading us to a more stabel learning.
mean_reward: This is the grand victory of the advantage function. we can clearly see how though the cpg achieves greater reward faster the aac the aac has significantly smoother graph. This we predict will lead us to greater results in a long-term trainig. As predicted the aac manged to achieve mean reward of 160!
"""
