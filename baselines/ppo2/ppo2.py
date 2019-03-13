import os
import time
import functools
import numpy as np
import os.path as osp
import tensorflow as tf
from baselines import logger
from collections import deque
from baselines.common import explained_variance, set_global_seeds
from baselines.common.policies import build_policy
from baselines.common.runners import AbstractEnvRunner
from baselines.common.tf_util import get_session, save_variables, load_variables
from baselines.common.mpi_adam_optimizer import MpiAdamOptimizer

from mpi4py import MPI
from baselines.common.tf_util import initialize
from baselines.common.mpi_util import sync_from_root
from policies.base_policy import PolicyTrainMode


class Model(object):
    """
    We use this object to :
    __init__:
    - Creates the step_model
    - Creates the train_model

    train():
    - Make the training part (feedforward and retropropagation of gradients)

    save/load():
    - Save load the model
    """
    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                nsteps, ent_coef, vf_coef, max_grad_norm, l2_coef=0):
        sess = get_session()

        with tf.variable_scope('ppo2_model', reuse=tf.AUTO_REUSE):
            # CREATE OUR TWO MODELS
            # act_model that is used for sampling
            act_model = policy(nbatch_act, 1, sess)

            # Train model for training
            train_model = policy(nbatch_train, nsteps, sess)

            bc_model = policy(nbatch=None, nsteps=1, sess=sess)

        # CREATE THE PLACEHOLDERS
        A = train_model.pdtype.sample_placeholder([None])
        ADV = tf.placeholder(tf.float32, [None])
        R = tf.placeholder(tf.float32, [None])
        # Keep track of old actor
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        # Keep track of old critic
        OLDVPRED = tf.placeholder(tf.float32, [None])
        LR = tf.placeholder(tf.float32, [])
        # Cliprange
        CLIPRANGE = tf.placeholder(tf.float32, [])

        BC_ACT = tf.placeholder(bc_model.action.dtype, bc_model.action.shape)

        neglogpac = train_model.pd.neglogp(A)

        # Calculate the entropy
        # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
        entropy = tf.reduce_mean(train_model.pd.entropy())

        # CALCULATE THE LOSS
        # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss

        # Clip the value to reduce variability during Critic training
        # Get the predicted value
        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        # Unclipped value
        vf_losses1 = tf.square(vpred - R)
        # Clipped value
        vf_losses2 = tf.square(vpredclipped - R)

        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

        # Calculate ratio (pi current policy / pi old policy)
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)

        # Defining Loss = - J is equivalent to max J
        pg_losses = -ADV * ratio

        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)

        # Final PG loss
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))

        weights = [v for v in tf.trainable_variables() if '/w:0' in v.name]
        l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in weights])

        # Total loss
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

        if bc_model.action.dtype == tf.float32:
            squared_differences = (bc_model.pi - BC_ACT) ** 2  # pi = mean
            # We assume we're operating on FetchPickAndPlace
            assert squared_differences.shape.as_list() == [None, 4]
            # Give gripper action loss 10 times as much weight as other actions
            squared_differences = tf.concat([squared_differences[:, :3], 10 * squared_differences[:, 3, None]], axis=1)
            assert squared_differences.shape.as_list() == [None, 4]
            squared_norms = tf.reduce_sum(squared_differences, axis=1)
            assert squared_norms.shape.as_list() == [None]
            bc_loss = tf.reduce_mean(squared_norms, axis=0)
            assert bc_loss.shape.as_list() == []
        elif bc_model.action.dtype == tf.int64:
            bc_loss = bc_model.pd.neglogp(BC_ACT)
            assert bc_loss.shape.as_list() == [None]
            bc_loss = tf.reduce_mean(bc_loss)
        else:
            raise Exception(f"Unknown action dtype '{bc_model.action.dtype}'")

        bc_coef = 10

        losses = {
            PolicyTrainMode.R_ONLY: loss + l2_loss * l2_coef,
            PolicyTrainMode.BC_ONLY: bc_loss + l2_loss * l2_coef,
            PolicyTrainMode.R_PLUS_BC: loss + bc_loss * bc_coef + l2_loss * l2_coef
        }

        train_ops = dict()
        # UPDATE THE PARAMETERS USING LOSS
        # 1. Get the model parameters
        params = tf.trainable_variables('ppo2_model')
        for loss in [PolicyTrainMode.R_ONLY, PolicyTrainMode.R_PLUS_BC, PolicyTrainMode.BC_ONLY]:
            # 2. Build our trainer
            trainer = MpiAdamOptimizer(MPI.COMM_WORLD, learning_rate=LR, epsilon=1e-5)
            # 3. Calculate the gradients
            grads_and_var = trainer.compute_gradients(losses[loss], params)
            grads, var = zip(*grads_and_var)

            if max_grad_norm is not None:
                # Clip the gradients (normalize)
                grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
            grads_and_var = list(zip(grads, var))
            # zip aggregate each gradient with parameters associated
            # For instance zip(ABCD, xyza) => Ax, By, Cz, Da

            train_ops[loss] = trainer.apply_gradients(grads_and_var)

        def train(lr, cliprange, obs, returns, masks, actions, values, neglogpacs, states=None):
            # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
            # Returns = R + yV(s')
            advs = returns - values

            # Normalize the advantages
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            td_map = {train_model.X:obs, A:actions, ADV:advs, R:returns, LR:lr,
                    CLIPRANGE:cliprange, OLDNEGLOGPAC:neglogpacs, OLDVPRED:values}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks

            return sess.run(
                [pg_loss, vf_loss, entropy, approxkl, clipfrac, train_ops[PolicyTrainMode.R_ONLY]],
                td_map
            )[:-1]
        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']

        def train_rl_bc(lr, cliprange=None, obs=None, returns=None, masks=None, actions=None,
                        values=None, neglogpacs=None, states=None,
                        train_mode=PolicyTrainMode.R_ONLY, bc_obses=None, bc_actions=None):
            td_map = {LR: lr}
            fetches = {}
            if train_mode in [PolicyTrainMode.R_ONLY, PolicyTrainMode.R_PLUS_BC]:
                advs = returns - values
                advs = (advs - advs.mean()) / (advs.std() + 1e-8)
                td_map.update({train_model.X: obs,
                               A: actions,
                               ADV: advs,
                               R: returns,
                               CLIPRANGE: cliprange,
                               OLDNEGLOGPAC: neglogpacs,
                               OLDVPRED: values})
                if states is not None:
                    td_map[train_model.S] = states
                    td_map[train_model.M] = masks
                fetches.update({'policy_loss': pg_loss, 'value_loss': vf_loss,
                                'policy_entropy': entropy, 'approxkl': approxkl,
                                'clipfrac': clipfrac})
            if train_mode in [PolicyTrainMode.BC_ONLY, PolicyTrainMode.R_PLUS_BC]:
                td_map.update({bc_model.X: bc_obses,
                              BC_ACT: bc_actions})
                fetches.update({'bc_loss': bc_loss})
            train_op = train_ops[train_mode]
            vals = sess.run(list(fetches.values()) + [train_op], td_map)[:-1]
            return {k: v for k, v in zip(fetches.keys(), vals)}

        def train_bc_only(obs, actions, lr):
            feed_dict = {bc_model.X: obs, BC_ACT: actions, LR: lr}
            loss, l2_loss_v, _ = sess.run([bc_loss, l2_loss, train_ops[PolicyTrainMode.BC_ONLY]], feed_dict)
            return loss, l2_loss_v

        self.train = train
        self.train_rl_bc = train_rl_bc
        self.train_bc_only = train_bc_only
        self.train_model = train_model
        self.act_model = act_model
        self.bc_model = bc_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state
        self.losses = losses
        self.BC_ACT = BC_ACT
        self.bc_loss = bc_loss

        self.save = functools.partial(save_variables, sess=sess)
        self.load = functools.partial(load_variables, sess=sess)

        if MPI.COMM_WORLD.Get_rank() == 0:
            initialize()
        global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="")
        sync_from_root(sess, global_variables) #pylint: disable=E1101

class Runner(AbstractEnvRunner):
    """
    We use this object to make a mini batch of experiences
    __init__:
    - Initialize the runner

    run():
    - Make a mini batch
    """
    def __init__(self, *, env, model, nsteps, gamma, lam):
        super().__init__(env=env, model=model, nsteps=nsteps)
        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        # Discount rate
        self.gamma = gamma

    def run(self):
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        mb_states = self.states
        epinfos = []
        # For n in range number of steps
        for _ in range(self.nsteps):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            actions, values, self.states, neglogpacs = self.model.step(self.obs, S=self.states, M=self.dones)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)

            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, S=self.states, M=self.dones)

        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values
        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
            mb_states, epinfos)
# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

def constfn(val):
    def f(_):
        return val
    return f

def learn(*, network, env, total_timesteps, eval_env = None, seed=None, nsteps=2048, ent_coef=0.0, lr=3e-4,
            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
            log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
            save_interval=0, load_path=None, **network_kwargs):
    '''
    Learn policy using PPO algorithm (https://arxiv.org/abs/1707.06347)

    Parameters:
    ----------

    network:                          policy network architecture. Either string (mlp, lstm, lnlstm, cnn_lstm, cnn, cnn_small, conv_only - see baselines.common/models.py for full list)
                                      specifying the standard network architecture, or a function that takes tensorflow tensor as input and returns
                                      tuple (output_tensor, extra_feed) where output tensor is the last network layer output, extra_feed is None for feed-forward
                                      neural nets, and extra_feed is a dictionary describing how to feed state into the network for recurrent neural nets.
                                      See common/models.py/lstm for more details on using recurrent nets in policies

    env: baselines.common.vec_env.VecEnv     environment. Needs to be vectorized for parallel environment simulation.
                                      The environments produced by gym.make can be wrapped using baselines.common.vec_env.DummyVecEnv class.


    nsteps: int                       number of steps of the vectorized environment per update (i.e. batch size is nsteps * nenv where
                                      nenv is number of environment copies simulated in parallel)

    total_timesteps: int              number of timesteps (i.e. number of actions taken in the environment)

    ent_coef: float                   policy entropy coefficient in the optimization objective

    lr: float or function             learning rate, constant or a schedule function [0,1] -> R+ where 1 is beginning of the
                                      training and 0 is the end of the training.

    vf_coef: float                    value function loss coefficient in the optimization objective

    max_grad_norm: float or None      gradient norm clipping coefficient

    gamma: float                      discounting factor

    lam: float                        advantage estimation discounting factor (lambda in the paper)

    log_interval: int                 number of timesteps between logging events

    nminibatches: int                 number of training minibatches per update. For recurrent policies,
                                      should be smaller or equal than number of environments run in parallel.

    noptepochs: int                   number of training epochs per update

    cliprange: float or function      clipping range, constant or schedule function [0,1] -> R+ where 1 is beginning of the training
                                      and 0 is the end of the training

    save_interval: int                number of timesteps between saving events

    load_path: str                    path to load the model from

    **network_kwargs:                 keyword arguments to the policy / network builder. See baselines.common/policies.py/build_policy and arguments to a particular type of network
                                      For instance, 'mlp' network architecture has arguments num_hidden and num_layers.



    '''

    set_global_seeds(seed)

    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    policy = build_policy(env, network, **network_kwargs)

    # Get the nb of env
    nenvs = env.num_envs

    # Get state_space and action_space
    ob_space = env.observation_space
    ac_space = env.action_space

    # Calculate the batch_size
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches

    # Instantiate the model object (that creates act_model and train_model)
    make_model = lambda : Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                    nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm)
    model = make_model()
    if load_path is not None:
        model.load(load_path)
    # Instantiate the runner object
    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)
    if eval_env is not None:
        eval_runner = Runner(env = eval_env, model = model, nsteps = nsteps, gamma = gamma, lam= lam)



    epinfobuf = deque(maxlen=100)
    if eval_env is not None:
        eval_epinfobuf = deque(maxlen=100)

    # Start total timer
    tfirststart = time.time()

    nupdates = total_timesteps//nbatch
    for update in range(1, nupdates+1):
        assert nbatch % nminibatches == 0
        # Start timer
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / nupdates
        # Calculate the learning rate
        lrnow = lr(frac)
        # Calculate the cliprange
        cliprangenow = cliprange(frac)
        # Get minibatch
        obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run() #pylint: disable=E0632
        if eval_env is not None:
            eval_obs, eval_returns, eval_masks, eval_actions, eval_values, eval_neglogpacs, eval_states, eval_epinfos = eval_runner.run() #pylint: disable=E0632

        epinfobuf.extend(epinfos)
        if eval_env is not None:
            eval_epinfobuf.extend(eval_epinfos)

        # Here what we're going to do is for each minibatch calculate the loss and append it.
        mblossvals = []
        if states is None: # nonrecurrent version
            # Index of each element of batch_size
            # Create the indices array
            inds = np.arange(nbatch)
            for _ in range(noptepochs):
                # Randomize the indexes
                np.random.shuffle(inds)
                # 0 to batch_size with batch_train_size step
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices))
        else: # recurrent version
            assert nenvs % nminibatches == 0
            envsperbatch = nenvs // nminibatches
            envinds = np.arange(nenvs)
            flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
            envsperbatch = nbatch_train // nsteps
            for _ in range(noptepochs):
                np.random.shuffle(envinds)
                for start in range(0, nenvs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mbflatinds = flatinds[mbenvinds].ravel()
                    slices = (arr[mbflatinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mbstates = states[mbenvinds]
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices, mbstates))

        # Feedforward --> get losses --> update
        lossvals = np.mean(mblossvals, axis=0)
        # End timer
        tnow = time.time()
        # Calculate the fps (frame per second)
        fps = int(nbatch / (tnow - tstart))
        if update % log_interval == 0 or update == 1:
            # Calculates if value function is a good predicator of the returns (ev > 1)
            # or if it's just worse than predicting nothing (ev =< 0)
            ev = explained_variance(values, returns)
            logger.logkv("serial_timesteps", update*nsteps)
            logger.logkv("nupdates", update)
            logger.logkv("total_timesteps", update*nbatch)
            logger.logkv("fps", fps)
            logger.logkv("explained_variance", float(ev))
            logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
            if eval_env is not None:
                logger.logkv('eval_eprewmean', safemean([epinfo['r'] for epinfo in eval_epinfobuf]) )
                logger.logkv('eval_eplenmean', safemean([epinfo['l'] for epinfo in eval_epinfobuf]) )
            logger.logkv('time_elapsed', tnow - tfirststart)
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv(lossname, lossval)
            if MPI.COMM_WORLD.Get_rank() == 0:
                logger.dumpkvs()
        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir() and MPI.COMM_WORLD.Get_rank() == 0:
            checkdir = osp.join(logger.get_dir(), 'checkpoints')
            os.makedirs(checkdir, exist_ok=True)
            savepath = osp.join(checkdir, '%.5i'%update)
            print('Saving to', savepath)
            model.save(savepath)
    return model
# Avoid division error when calculate the mean (in our case if epinfo is empty returns np.nan, not return an error)
def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)



