import tensorflow as tf
import numpy as np
import random

class UAV_fire_extinguish(object):

  n_w = 4
  n_uav = 2
  n_fire = 1
  u_loca = (0,16)
  t_fail = (0.0,0.0)
  t_emit = (1.0,0.5)
  l_fire = [12]
  r_fire = [10.0]
  e_fire = [(1.0,1.0)]
  horizon = 20
  horizon = 5

##### Sampling method #####

def sampling_events(event,prob):

  n_length = len(event)

  x_rand = np.random.random()

  for i in range(n_length):

    x_rand = x_rand - prob[i]

    if x_rand <= 0:

      return event[i]


def mix_distribution(event1,prob1,event2,prob2):

  n_length_1 = len(event1)
  n_length_2 = len(event2)

  new_event = []
  new_prob = []

  for e1 in range(n_length_1):
    for e2 in range(n_length_2):
      e_new = event1[e1] + [event2[e2]]
      new_event.append(e_new)
      p_new = prob1[e1] * prob2[e2]
      new_prob.append(p_new)

  return (new_event,new_prob)

##### check boundary #####

def check_boundary(x,w):

  if x < 0:
    return 0
  elif x > w-1:
    return w-1
  else:
    return x


n_w = UAV_fire_extinguish.n_w
n_grid = n_w * n_w

##################################
##### Mapping between states #####
##################################

###

def two_dim_to_one(l_cor,n_w):

  x = l_cor[0]
  y = l_cor[1]

  l = n_w * y + x

  return l

def one_dim_to_two(l,n_w):

  x = l%n_w
  y = (l-x)/n_w

  return (x,y)

###

############################
##### TRANSITION MODEL #####
############################

### simple movement of one agent due to action

def move_location_single(l_1d,a,n_w):

  if l_1d == n_w * n_w:

    return l_1d


  l = one_dim_to_two(l_1d,n_w)

  x_next = l[0]
  y_next = l[1]

  if a == 0: # up
    y_next = y_next + 1
  elif a == 1: # down
    y_next = y_next - 1
  elif a == 2: # left
    x_next = x_next - 1
  elif a == 3:
    x_next = x_next + 1
  else:
    pass


  x_next = check_boundary(x_next,n_w)
  y_next = check_boundary(y_next,n_w)

  l_next = two_dim_to_one((x_next,y_next),n_w)

  return l_next

######################################################
##### number of uavs at the location of the fire #####
######################################################

def fire_has_uavs(lf,l1,l2):

  num = 0

  if lf == l1:

    num += 1

  if lf == l2:

    num += 1

  return num

######################################################################
##### Obtain all possible sets and the corresponding probability #####
######################################################################

def transition_model(cart_product,a_joint,UAV_fire_extinguish):

  s_fail = UAV_fire_extinguish.n_w * UAV_fire_extinguish.n_w

  ##### Terminal states #####

  initial_state = list((UAV_fire_extinguish.u_loca[0],UAV_fire_extinguish.u_loca[1],1))

  if cart_product[0] == s_fail and cart_product[1] == s_fail:

    return ([initial_state],[1.0])

  fire_sum = 0

  for i in range(UAV_fire_extinguish.n_fire):

    fire_sum += cart_product[UAV_fire_extinguish.n_uav + i]

  if fire_sum == 0:

    return ([initial_state],[1.0])


  ##### Transition of the first UAV #####

  if cart_product[0] == s_fail:

    event_set_0 = [[s_fail]]
    prob_set_0 = [1.0]

  else:

    l0_next = move_location_single(cart_product[0],a_joint[0],UAV_fire_extinguish.n_w)
    event_set_0 = [[l0_next],[s_fail]]
    prob_set_0 =  [1.0 - UAV_fire_extinguish.t_fail[0], UAV_fire_extinguish.t_fail[0]]

  ##### Transition of the second UAV #####

  if cart_product[1] == s_fail:

    event_set_1 = [s_fail]
    prob_set_1 = [1.0]

  else:

    l1_next = move_location_single(cart_product[1],a_joint[1],UAV_fire_extinguish.n_w)
    event_set_1 = [l1_next,s_fail]
    prob_set_1 =  [1.0 - UAV_fire_extinguish.t_fail[1], UAV_fire_extinguish.t_fail[1]]

  (event_product,prob_product) = mix_distribution(event_set_0,prob_set_0,event_set_1,prob_set_1)

  ##### Transition of the fire states #####

  for i_fire in range(UAV_fire_extinguish.n_fire):

    the_fire_state = cart_product[UAV_fire_extinguish.n_uav + i_fire]

    if the_fire_state == 0: # no fire

      (event_product,prob_product) = mix_distribution(event_product,prob_product,[0],[1.0])

    else:

      l_f = UAV_fire_extinguish.l_fire[i_fire]
      l_0 = cart_product[0]
      l_1 = cart_product[1]

      if fire_has_uavs(l_f,l_0,l_1) == 1:

        rate_put_down = UAV_fire_extinguish.e_fire[i_fire][fire_has_uavs(l_f,l_0,l_1) - 1]
        (event_product,prob_product) = mix_distribution(event_product,prob_product,[0,1],[rate_put_down,1.0-rate_put_down])

      elif fire_has_uavs(l_f,l_0,l_1) == 2:

        rate_put_down = UAV_fire_extinguish.e_fire[i_fire][fire_has_uavs(l_f,l_0,l_1) - 1]
        (event_product,prob_product) = mix_distribution(event_product,prob_product,[0,1],[rate_put_down,1.0-rate_put_down])

      else:

        (event_product,prob_product) = mix_distribution(event_product,prob_product,[1],[1.0])



  return (event_product,prob_product)



def transition_sample(current_state, # location is one-dimensional
                      a_joint,
                      last_info_0):

  n_w = UAV_fire_extinguish.n_w

  reward = 0.0

  (event,prob) = transition_model(current_state,a_joint,UAV_fire_extinguish)

  next_state = sampling_events(event,prob)

  (xp0,yp0) = one_dim_to_two(next_state[0],n_w)
  (xp1,yp1) = one_dim_to_two(next_state[1],n_w)

  next_state_coor = [xp0,yp0,xp1,yp1,next_state[2]]

  # Collect rewards

  for i_fire in range(UAV_fire_extinguish.n_fire):

    if current_state[UAV_fire_extinguish.n_uav + i_fire] == 1 and next_state[UAV_fire_extinguish.n_uav + i_fire] == 0:

      reward += UAV_fire_extinguish.r_fire[i_fire]

  # Update info

  p_info_0 = random.random()

  if p_info_0 < UAV_fire_extinguish.t_emit[0] :
    next_info_0 = [xp0,yp0,next_state[2]] + [1]
  else:
    next_info_0 = last_info_0[:]
    next_info_0[-1] += 1


  return [next_state,(next_info_0,reward)]


def samples_by_random_action(n_init_pool,s_init):

  s0_pool = np.zeros((n_init_pool,4),float)
  a0_pool = np.zeros((n_init_pool,5),float)
  r0_pool = np.zeros((n_init_pool,1),float)
  s0p_pool = np.zeros((n_init_pool,4),float)

  s_current = s_init
  (x0_init ,y0_init) = one_dim_to_two(s_init[0],UAV_fire_extinguish.n_w)
  s_current_coor = [x0_init,y0_init,s_init[2]]

  last_info_0 = s_current_coor + [1]

  for i_event in range(n_init_pool):
    #print s_current
    a0 = random.randint(0,4)
    a1 = 4
    outcome = transition_sample(s_current,(a0,a1),last_info_0)

    next_state = outcome[0]
    (next_info_0,reward) = outcome[1]

    s0_pool[i_event,:] = last_info_0
    s0p_pool[i_event,:] = next_info_0
    a0_pool[i_event,a0] = 1.0
    r0_pool[i_event,0] = reward

    last_info_0 = next_info_0
    s_current = next_state



  return (s0_pool,a0_pool,r0_pool,s0p_pool)



def truncate_dataset(data_array,n_keep_size):

  n_size = len(data_array)

  return data_array[(n_size-n_keep_size):,:]


##########################################
############ Neural Network ##############
##########################################

##### functions for nerual network #####


def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    inputs = tf.to_float(inputs)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return (outputs,Weights,biases)

def copy_layer(inputs,Weights,biases,activation_function = None):
    inputs = tf.to_float(inputs)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

def es_greedy(inputs,epsi):

  x_rand = np.random.random()

  if x_rand < epsi:
    return np.random.randint(0,4)
  else:
    return np.argmax(inputs)

def batch_select(inputs,n_total,n_batch,seeds):

  batch_set = np.zeros((n_batch,len(inputs[0])))
  for i in range(n_batch):
    batch_set[i,:] = inputs[seeds[i],:]

  return batch_set



# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

n_hidd = 50
n_init_pool = 5000

outcome = samples_by_random_action(n_init_pool,[0,16,1])

s0_array = outcome[0]
a0_array = outcome[1]
r0_array = outcome[2]
sp0_array = outcome[3]

last_info = tf.placeholder(tf.int32,[None,4])
next_info = tf.placeholder(tf.int32,[None,4])
actions = tf.placeholder(tf.float32,[None,5])
rewards = tf.placeholder(tf.float32,[None,1])

W1_train = tf.placeholder(tf.float32,[4,n_hidd])
b1_train = tf.placeholder(tf.float32,[1,n_hidd])
W2_train = tf.placeholder(tf.float32,[n_hidd,5])
b2_train = tf.placeholder(tf.float32,[1,5])

##### Layers #####

layer_1 = add_layer(last_info,4,n_hidd,activation_function = tf.nn.relu)
layer_out = add_layer(layer_1[0],n_hidd,5,activation_function = None)
Q = layer_out[0]

layer_c1 = copy_layer(next_info,W1_train,b1_train,activation_function = tf.nn.relu)
Q_next = copy_layer(layer_c1,W2_train,b2_train,activation_function = None)

### Loss function ###
best_next_state_action = tf.reduce_max(Q_next,reduction_indices=[1],keep_dims=True)
current_state_action = tf.reduce_sum(tf.mul(Q,actions),reduction_indices=[1],keep_dims=True)
loss = tf.reduce_mean(tf.square(rewards + 0.95 * best_next_state_action - current_state_action))

### train ###
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

### session ###
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

### initialize all information ###
current_state = [0,16,1]
next_state = 0
last_info_0 = [0,0,1,1]

W1_for_feed_train = np.ones((4,n_hidd),float)
b1_for_feed_train = np.ones((1,n_hidd),float)
W2_for_feed_train = np.ones((n_hidd,5),float)
b2_for_feed_train = np.ones((1,5),float)

### batch ###
n_batch_size = 1000

seeds = random.sample(xrange(1,n_init_pool),n_batch_size)

s0_batch = batch_select(s0_array,n_init_pool,n_batch_size,seeds)
r0_batch = batch_select(r0_array,n_init_pool,n_batch_size,seeds)
a0_batch = batch_select(a0_array,n_init_pool,n_batch_size,seeds)
sp0_batch = batch_select(sp0_array,n_init_pool,n_batch_size,seeds)




## +++++++++++++++++++

h_train_step = 10000
h_grad = 100
r_explore = 0.2


for h in range(h_train_step):

  numeric_loss = 0.0

  ##### training the NN #####
  for i in range(h_grad):

    numeric_loss, _ = sess.run([loss,train_step],feed_dict={last_info: s0_batch,
                                                            rewards: r0_batch,
                                                            next_info:sp0_batch,
                                                            actions: a0_batch,
                                                            W1_train:W1_for_feed_train,
                                                            b1_train:b1_for_feed_train,
                                                            W2_train:W2_for_feed_train,
                                                            b2_train:b2_for_feed_train})


  print(numeric_loss)

  action_chosen_0 = es_greedy(sess.run(Q, feed_dict={last_info: [last_info_0]}),r_explore)
  action_chosen_1 = 4

  ##### sample the transition #####
  outcome_transition = transition_sample(current_state,(action_chosen_0,action_chosen_1),last_info_0)

  next_state = outcome_transition[0]
  (next_info_0,reward_immed) = outcome_transition[1]

  print('(iter_t, h    )= ',h)
  print('(current_state)= ',current_state)
  print('(joint action )= ',action_chosen_0)
  print('(next state   )= ',next_state)

  ##### increase sample dataset #####

  s0_array = np.vstack([s0_array, last_info_0])
  sp0_array = np.vstack([sp0_array, next_info_0])
  r0_array = np.vstack([r0_array, reward_immed])

  action_new_0 = np.zeros((1,5),float)
  action_new_0[0,action_chosen_0] = 1.0
  a0_array = np.vstack([a0_array, action_new_0])

  ##### update the train network #####

  if h%50 == 0:
    W1_for_feed_train = sess.run(layer_1[1], feed_dict={last_info: [last_info_0]})
    b1_for_feed_train = sess.run(layer_1[2], feed_dict={last_info: [last_info_0]})
    W2_for_feed_train = sess.run(layer_out[1], feed_dict={last_info: [last_info_0]})
    b2_for_feed_train = sess.run(layer_out[2], feed_dict={last_info: [last_info_0]})

  current_state = next_state
  last_info_0 = next_info_0

  # truncate the size of data samples
  s0_array = truncate_dataset(s0_array,n_init_pool)
  r0_array = truncate_dataset(r0_array,n_init_pool)
  a0_array = truncate_dataset(a0_array,n_init_pool)
  sp0_array = truncate_dataset(sp0_array,n_init_pool)

  # re-sample the batch set
  seeds = random.sample(xrange(1,n_init_pool),n_batch_size)

  s0_batch = batch_select(s0_array,n_init_pool,n_batch_size,seeds)
  r0_batch = batch_select(r0_array,n_init_pool,n_batch_size,seeds)
  a0_batch = batch_select(a0_array,n_init_pool,n_batch_size,seeds)
  sp0_batch = batch_select(sp0_array,n_init_pool,n_batch_size,seeds)
