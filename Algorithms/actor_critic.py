import tensorflow as tf
import numpy as np

class ActorCriticAgent:
    def __init__(self, 
                 actor,
                 critic,
                 env,
                #  actor_loss_fn,
                 critic_loss_fn = tf.keras.losses.MSE,
                 critic_optimizer = None,
                 actor_optimizer = None,
                 steps = 1):
        
        self.actor = actor
        self.critic = critic
        self.env = env        
        # self.actor_loss_fn = actor_loss_fn
        self.critic_loss_fn = critic_loss_fn
        self.steps = steps
        self.action_space = np.arange(self.env.action_space.n)
        if critic_optimizer == None:
            critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.critic_optimizer = critic_optimizer
        if actor_optimizer == None:
            actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.actor_optimizer = actor_optimizer

        self.history = []
    
    def train(self, episodes=100, gamma=0.99):
        for episode in range(episodes):
            s1 = self.env.reset()[0];
            rewards = []
            actions = []
            values  = []
            states  = []

            done = False
            t = 0
            while done == False :
                Vst = self.critic(np.array([s1]))[0]
                Pst = self.actor(np.array([s1]))[0]
                a = np.random.choice(np.arange(self.env.action_space.n), p=np.array(Pst))
                s2, r, done, trunc, info = self.env.step(a)
                ### reward logic ###
                if trunc:
                    done = True
                    r = -100
                elif done:
                    r = 1000
                else :
                    r = 0
                ####################
                rewards.append(r)
                actions.append(a)
                values.append(Vst[0])
                states.append(s1)
                s1 = s2
                t+=1
            
            rewards = np.array(rewards)[::-1]
            actions = np.array(actions)
            values  = np.array(values)
            values  = values/np.max(values)
            states  = np.array(states)

            # Compute discounted return 
            G = gamma**np.arange(len(rewards)) * rewards
            G = G.cumsum()[::-1]
            values.reshape(len(values), 1)
            assert G.shape == values.shape
            

            with tf.GradientTape() as tape:
                probs = self.actor(states)
                probs = tf.gather_nd(probs , indices=[[i,j] for i,j in enumerate(actions)])
                advantage = -1 * ( G - values )
                losses = tf.multiply(advantage , tf.math.log(probs) )
                actor_loss = tf.reduce_sum(losses)
            actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

            with tf.GradientTape() as tape:
                values = self.critic(states)
                critic_loss = tf.reduce_sum(tf.square(G - values))
            critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

            if episode % 10 == 0:
                print("Episode: {}, Total reward: {}".format(episode, t))    
            
            self.history.append(t)