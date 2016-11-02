from math import exp

def softmax(inputs,T):

  e_input = np.ones(len(inputs))

  for i in range(len(inputs)):

    e_input[i] = exp(inputs[i]/T)

  e_input = e_input/sum(e_input)

  return e_input


def softmax_choose(inputs,T):

  x_rand = np.random.random()

  e_input = np.ones(len(inputs))

  for i in range(len(inputs)):

    e_input[i] = exp(inputs[i]/T)

  e_input = e_input/sum(e_input)

  e_input[-1] += 0.01

  for i in range(len(inputs)):

    if x_rand < e_input[i]:

      return i

    else:

      x_rand = x_rand - e_input[i]


def visualize_scenario_indp_softmax(initial_state,h_print,T,UAV_fire_extinguish):

  size = UAV_fire_extinguish.n_w
  (x0,y0) = one_dim_to_two(initial_state[0],UAV_fire_extinguish.n_w)
  (x1,y1) = one_dim_to_two(initial_state[1],UAV_fire_extinguish.n_w)

  last_info_0 = [x0,y0,x1,y1,1,1,1,1,1,1,1]
  last_info_1 = [x0,y0,x1,y1,1,1,1,1,1,1,1]

  last_info_0_norm = observation_normalization(last_info_0,size)
  last_info_1_norm = observation_normalization(last_info_1,size)


  current_state = initial_state

  print("state  = ",current_state)

  for h in range(h_print):
    action_chosen_1 = softmax_choose(sess.run(Q_a1, feed_dict={last_info_a1: [last_info_1_norm]})[0],T)
    action_chosen_0 = softmax_choose(sess.run(Q_a0, feed_dict={last_info_a0: [last_info_0_norm]})[0],T)

    outcome_transition = transition_sample(current_state,
                                           (action_chosen_0,action_chosen_1),
                                           last_info_0,
                                           last_info_1,
                                           UAV_fire_extinguish)

    next_state = outcome_transition[0]
    (next_info_0,reward_immed) = outcome_transition[1]
    (next_info_1,reward_immed) = outcome_transition[2]

    next_info_0_norm = observation_normalization(next_info_0,size)
    next_info_1_norm = observation_normalization(next_info_1,size)

    #print("action = ",(action_chosen_0,action_chosen_1))

    current_state = next_state
    last_info_0 = next_info_0
    last_info_1 = next_info_1
    last_info_0_norm = next_info_0_norm
    last_info_1_norm = next_info_1_norm

    print("state  = ",current_state)
    #print("delay  = ",(last_info_0[-1],last_info_1[-1]))

