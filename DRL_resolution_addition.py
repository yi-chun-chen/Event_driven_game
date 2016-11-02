for task in range(2):


  UAV_task = UAV_8_by_8
  it_time = 4

  size = UAV_task.n_w
  print("size = ", size)

  UAV_task.e_fire[2][0] = 0.0


  current_state = [0,size**2-1,1,1,1]
  next_state = 0
  last_info_0 = [0,0,size-1,size-1,1,1,1,1]
  last_info_1 = [0,0,size-1,size-1,1,1,1,1]

  last_info_0_norm = observation_normalization(last_info_0,size)
  last_info_1_norm = observation_normalization(last_info_1,size)

  for iteration_times in range(it_time):
    print("iteration times = ", iteration_times)
    print("--- %s seconds ---" % (time.time() - start_time))

    n_batch_size = n_batch_size + 1000
    learning_rate_given = learning_rate_given / 2.0

    print("batch size = ",n_batch_size)
    print("learning rate = ", learning_rate_given)

    if iteration_times + task != 0:
      UAV_task.e_fire[2][0] = 0.0
      print("rate is ", UAV_task.e_fire[2][0])

    ##############################
    ###### Only for agent 1 ######
    ##############################

    ###### Data Set for agent 1 #####
    if iteration_times + task != 0 :

      s0_array_0 = 0

      outcome = samples_by_random_action_fix_one(n_init_pool,[0,size**2 - 1,1,1,1],1,UAV_task)
      #outcome = samples_by_random_action(n_init_pool,[0,4**2-1,1,1,1],UAV_4_by_4)

      (s0_array_0,a0_array_0,r0_array_0,sp0_array_0) = (outcome[0][0],outcome[0][1],outcome[0][2],outcome[0][3])
      (s1_array_0,a1_array_0,r1_array_0,sp1_array_0) = (outcome[1][0],outcome[1][1],outcome[1][2],outcome[1][3])

      (s0_array_1,a0_array_1,r0_array_1,sp0_array_1) = (outcome[0][0],outcome[0][1],outcome[0][2],outcome[0][3])
      (s1_array_1,a1_array_1,r1_array_1,sp1_array_1) = (outcome[1][0],outcome[1][1],outcome[1][2],outcome[1][3])

      seeds = random.sample(xrange(1,n_init_pool),n_batch_size)

      s0_batch_0 = batch_select(s0_array_0,n_batch_size,seeds)
      r0_batch_0 = batch_select(r0_array_0,n_batch_size,seeds)
      a0_batch_0 = batch_select(a0_array_0,n_batch_size,seeds)
      sp0_batch_0 = batch_select(sp0_array_0,n_batch_size,seeds)

      s1_batch_0 = batch_select(s1_array_0,n_batch_size,seeds)
      r1_batch_0 = batch_select(r1_array_0,n_batch_size,seeds)
      a1_batch_0 = batch_select(a1_array_0,n_batch_size,seeds)
      sp1_batch_0 = batch_select(sp1_array_0,n_batch_size,seeds)

      seeds = random.sample(xrange(1,n_init_pool),n_batch_size)

      s0_batch_1 = batch_select(s0_array_1,n_batch_size,seeds)
      r0_batch_1 = batch_select(r0_array_1,n_batch_size,seeds)
      a0_batch_1 = batch_select(a0_array_1,n_batch_size,seeds)
      sp0_batch_1 = batch_select(sp0_array_1,n_batch_size,seeds)

      s1_batch_1 = batch_select(s1_array_1,n_batch_size,seeds)
      r1_batch_1 = batch_select(r1_array_1,n_batch_size,seeds)
      a1_batch_1 = batch_select(a1_array_1,n_batch_size,seeds)
      sp1_batch_1 = batch_select(sp1_array_1,n_batch_size,seeds)

    #################################
    #################################


    for h in range(h_train_step):

      numeric_loss_a0 = 0.0

      ##### training the NN for every 50 steps of self-playing #####
      if h % h_step_for_gradient == 0:

        for i in range(h_grad):

          numeric_loss_a0, _ = sess.run([loss_a0,train_step_a0],feed_dict={last_info_a0: s0_batch_0,
                                                                           rewards_a0: r0_batch_0,
                                                                           next_info_a0: sp0_batch_0,
                                                                           actions_a0: a0_batch_0,
                                                                           W1_train_a0: W1_for_feed_train_a0,
                                                                           b1_train_a0: b1_for_feed_train_a0,
                                                                           W2_train_a0: W2_for_feed_train_a0,
                                                                           b2_train_a0: b2_for_feed_train_a0,
                                                                           learning_rate: learning_rate_given})


      ##### Choose action #####

      # action for agent 0 is chosen from the current NN
      # action for agent 1 is chosen randomly in the first iteration,
      # otherwise it is learned from previous network

      action_chosen_0 = es_greedy(sess.run(Q_a0, feed_dict={last_info_a0: [[size] +last_info_0_norm]}),r_explore)
      if iteration_times + task == 0:
        action_chosen_1 = random.randint(0,4)
      else:
        action_chosen_1 = es_greedy(sess.run(Q_a1, feed_dict={last_info_a1: [[size] +last_info_1_norm]}),0.0)

      ##### sample the transition #####
      outcome_transition = transition_sample(current_state,
                                             (action_chosen_0,action_chosen_1),
                                             last_info_0,
                                             last_info_1,
                                             UAV_task)

      next_state = outcome_transition[0]
      (next_info_0,reward_immed) = outcome_transition[1]
      (next_info_1,reward_immed) = outcome_transition[2]

      next_info_0_norm = observation_normalization(next_info_0,size)
      next_info_1_norm = observation_normalization(next_info_1,size)


      if h%(20*h_step_for_gradient) == 0: print(h,numeric_loss_a0)


      #print(current_state,action_chosen_0,action_chosen_1,reward_immed)


      ##### increase sample dataset #####

      s0_array_0 = np.vstack([s0_array_0, [size] + last_info_0_norm])
      sp0_array_0 = np.vstack([sp0_array_0, [size] + next_info_0_norm])
      r0_array_0 = np.vstack([r0_array_0, reward_immed])

      s1_array_0 = np.vstack([s1_array_0, [size] + last_info_1_norm])
      sp1_array_0 = np.vstack([sp1_array_0, [size] + next_info_1_norm])
      r1_array_0 = np.vstack([r1_array_0, reward_immed])

      action_new_0 = np.zeros((1,5),float)
      action_new_0[0,action_chosen_0] = 1.0
      a0_array_0 = np.vstack([a0_array_0, action_new_0])

      action_new_1 = np.zeros((1,5),float)
      action_new_1[0,action_chosen_1] = 1.0
      a1_array_0 = np.vstack([a1_array_0, action_new_1])

      current_state = next_state
      last_info_0 = next_info_0
      last_info_1 = next_info_1
      last_info_0_norm = next_info_0_norm
      last_info_1_norm = next_info_1_norm

      ##### update the target network #####

      if h% (h_step_for_gradient * 50) == 0:
        W1_for_feed_train_a0 = sess.run(layer_1_a0[1], feed_dict={last_info_a0: [[size] +last_info_0_norm]})
        b1_for_feed_train_a0 = sess.run(layer_1_a0[2], feed_dict={last_info_a0: [[size] +last_info_0_norm]})
        W2_for_feed_train_a0 = sess.run(layer_out_a0[1], feed_dict={last_info_a0: [[size] +last_info_0_norm]})
        b2_for_feed_train_a0 = sess.run(layer_out_a0[2], feed_dict={last_info_a0: [[size] +last_info_0_norm]})


      ##### update the dataset #####
      if h % h_step_for_gradient == 0:

        # truncate the size of data samples
        s0_array_0 = truncate_dataset(s0_array_0,n_upper_size)
        r0_array_0 = truncate_dataset(r0_array_0,n_upper_size)
        a0_array_0 = truncate_dataset(a0_array_0,n_upper_size)
        sp0_array_0 = truncate_dataset(sp0_array_0,n_upper_size)

        s1_array_0 = truncate_dataset(s1_array_0,n_upper_size)
        r1_array_0 = truncate_dataset(r1_array_0,n_upper_size)
        a1_array_0 = truncate_dataset(a1_array_0,n_upper_size)
        sp1_array_0 = truncate_dataset(sp1_array_0,n_upper_size)


        # re-sample the batch set
        seeds = random.sample(xrange(1,len(s0_array_0)),n_batch_size)

        s0_batch_0 = batch_select(s0_array_0,n_batch_size,seeds)
        r0_batch_0 = batch_select(r0_array_0,n_batch_size,seeds)
        a0_batch_0 = batch_select(a0_array_0,n_batch_size,seeds)
        sp0_batch_0 = batch_select(sp0_array_0,n_batch_size,seeds)

        s1_batch_0 = batch_select(s1_array_0,n_batch_size,seeds)
        r1_batch_0 = batch_select(r1_array_0,n_batch_size,seeds)
        a1_batch_0 = batch_select(a1_array_0,n_batch_size,seeds)
        sp1_batch_0 = batch_select(sp1_array_0,n_batch_size,seeds)



    visualize_scenario_indp([0,size**2-1,1,1,1],30,0.0,UAV_task)
    print("============================================")
    visualize_scenario_indp([0,size**2-1,1,1,1],30,0.0,UAV_task)
    print("============================================")
    visualize_scenario_indp([0,size**2-1,1,1,1],30,0.0,UAV_task)
    #############################################
    ##### Done the training for the agent 0 #####
    #############################################


    #####################################
    ##### Training only for agent 1 #####
    #####################################

    UAV_task.e_fire[2][0] = 0.0
    print("rate is ", UAV_task.e_fire[2][0])

    ###### Data Set for agent 1 #####
    if iteration_times + task != -1 :

      s0_array_0 = 0

      outcome = samples_by_random_action_fix_one(n_init_pool,[0,size**2 - 1,1,1,1],0,UAV_task)
      #outcome = samples_by_random_action(n_init_pool,[0,4**2-1,1,1,1],UAV_4_by_4)

      (s0_array_0,a0_array_0,r0_array_0,sp0_array_0) = (outcome[0][0],outcome[0][1],outcome[0][2],outcome[0][3])
      (s1_array_0,a1_array_0,r1_array_0,sp1_array_0) = (outcome[1][0],outcome[1][1],outcome[1][2],outcome[1][3])

      (s0_array_1,a0_array_1,r0_array_1,sp0_array_1) = (outcome[0][0],outcome[0][1],outcome[0][2],outcome[0][3])
      (s1_array_1,a1_array_1,r1_array_1,sp1_array_1) = (outcome[1][0],outcome[1][1],outcome[1][2],outcome[1][3])

      seeds = random.sample(xrange(1,n_init_pool),n_batch_size)

      s0_batch_0 = batch_select(s0_array_0,n_batch_size,seeds)
      r0_batch_0 = batch_select(r0_array_0,n_batch_size,seeds)
      a0_batch_0 = batch_select(a0_array_0,n_batch_size,seeds)
      sp0_batch_0 = batch_select(sp0_array_0,n_batch_size,seeds)

      s1_batch_0 = batch_select(s1_array_0,n_batch_size,seeds)
      r1_batch_0 = batch_select(r1_array_0,n_batch_size,seeds)
      a1_batch_0 = batch_select(a1_array_0,n_batch_size,seeds)
      sp1_batch_0 = batch_select(sp1_array_0,n_batch_size,seeds)

      seeds = random.sample(xrange(1,n_init_pool),n_batch_size)

      s0_batch_1 = batch_select(s0_array_1,n_batch_size,seeds)
      r0_batch_1 = batch_select(r0_array_1,n_batch_size,seeds)
      a0_batch_1 = batch_select(a0_array_1,n_batch_size,seeds)
      sp0_batch_1 = batch_select(sp0_array_1,n_batch_size,seeds)

      s1_batch_1 = batch_select(s1_array_1,n_batch_size,seeds)
      r1_batch_1 = batch_select(r1_array_1,n_batch_size,seeds)
      a1_batch_1 = batch_select(a1_array_1,n_batch_size,seeds)
      sp1_batch_1 = batch_select(sp1_array_1,n_batch_size,seeds)

    #################################

    for h in range(h_train_step):

      numeric_loss_a1 = 0.0

      ##### training the NN #####

      if h % h_step_for_gradient == 0:
        for i in range(h_grad):

          numeric_loss_a1, _ = sess.run([loss_a1,train_step_a1],feed_dict={last_info_a1: s1_batch_1,
                                                                           rewards_a1: r1_batch_1,
                                                                           next_info_a1: sp1_batch_1,
                                                                           actions_a1: a1_batch_1,
                                                                           W1_train_a1: W1_for_feed_train_a1,
                                                                           b1_train_a1: b1_for_feed_train_a1,
                                                                           W2_train_a1: W2_for_feed_train_a1,
                                                                           b2_train_a1: b2_for_feed_train_a1,
                                                                           learning_rate: learning_rate_given})

      ##### Choose action #####

      # action for agent 1 is chosen from the current NN
      # action for agent 0 is chosen from the previous network

      action_chosen_1 = es_greedy(sess.run(Q_a1, feed_dict={last_info_a1: [[size] +last_info_1_norm]}),r_explore)
      action_chosen_0 = es_greedy(sess.run(Q_a0, feed_dict={last_info_a0: [[size] +last_info_0_norm]}),0.0)


      ##### sample the transition #####
      outcome_transition = transition_sample(current_state,
                                             (action_chosen_0,action_chosen_1),
                                             last_info_0,
                                             last_info_1,
                                             UAV_task)

      next_state = outcome_transition[0]
      (next_info_0,reward_immed) = outcome_transition[1]
      (next_info_1,reward_immed) = outcome_transition[2]

      next_info_0_norm = observation_normalization(next_info_0,size)
      next_info_1_norm = observation_normalization(next_info_1,size)

      if h%(20*h_step_for_gradient) == 0: print(h,numeric_loss_a1,last_info_0_norm)


      ##### increase sample dataset #####

      s0_array_1 = np.vstack([s0_array_1, [size] + last_info_0_norm])
      sp0_array_1 = np.vstack([sp0_array_1, [size] + next_info_0_norm])
      r0_array_1 = np.vstack([r0_array_1, reward_immed])

      s1_array_1 = np.vstack([s1_array_1, [size] + last_info_1_norm])
      sp1_array_1 = np.vstack([sp1_array_1, [size] + next_info_1_norm])
      r1_array_1 = np.vstack([r1_array_1, reward_immed])

      action_new_0 = np.zeros((1,5),float)
      action_new_0[0,action_chosen_0] = 1.0
      a0_array_1 = np.vstack([a0_array_1, action_new_0])

      action_new_1 = np.zeros((1,5),float)
      action_new_1[0,action_chosen_1] = 1.0
      a1_array_1 = np.vstack([a1_array_1, action_new_1])

      current_state = next_state
      last_info_0 = next_info_0
      last_info_1 = next_info_1
      last_info_0_norm = next_info_0_norm
      last_info_1_norm = next_info_1_norm

      ##### update the train network #####

      if h% (h_step_for_gradient * 50) == 0:
        W1_for_feed_train_a1 = sess.run(layer_1_a1[1], feed_dict={last_info_a1: [[size] + last_info_1_norm]})
        b1_for_feed_train_a1 = sess.run(layer_1_a1[2], feed_dict={last_info_a1: [[size] + last_info_1_norm]})
        W2_for_feed_train_a1 = sess.run(layer_out_a1[1], feed_dict={last_info_a1: [[size] + last_info_1_norm]})
        b2_for_feed_train_a1 = sess.run(layer_out_a1[2], feed_dict={last_info_a1: [[size] + last_info_1_norm]})


      ##### update the dataset #####
      if h % h_step_for_gradient == 0:

        # truncate the size of data samples
        s0_array_1 = truncate_dataset(s0_array_1,n_upper_size)
        r0_array_1 = truncate_dataset(r0_array_1,n_upper_size)
        a0_array_1 = truncate_dataset(a0_array_1,n_upper_size)
        sp0_array_1 = truncate_dataset(sp0_array_1,n_upper_size)

        s1_array_1 = truncate_dataset(s1_array_1,n_upper_size)
        r1_array_1 = truncate_dataset(r1_array_1,n_upper_size)
        a1_array_1 = truncate_dataset(a1_array_1,n_upper_size)
        sp1_array_1 = truncate_dataset(sp1_array_1,n_upper_size)


        # re-sample the batch set
        seeds = random.sample(xrange(1,len(s0_array_1)),n_batch_size)

        s0_batch_1 = batch_select(s0_array_1,n_batch_size,seeds)
        r0_batch_1 = batch_select(r0_array_1,n_batch_size,seeds)
        a0_batch_1 = batch_select(a0_array_1,n_batch_size,seeds)
        sp0_batch_1 = batch_select(sp0_array_1,n_batch_size,seeds)

        s1_batch_1 = batch_select(s1_array_1,n_batch_size,seeds)
        r1_batch_1 = batch_select(r1_array_1,n_batch_size,seeds)
        a1_batch_1 = batch_select(a1_array_1,n_batch_size,seeds)
        sp1_batch_1 = batch_select(sp1_array_1,n_batch_size,seeds)

    visualize_scenario_indp([0,size**2-1,1,1,1],30,0.0,UAV_task)
    print("============================================")
    visualize_scenario_indp([0,size**2-1,1,1,1],30,0.0,UAV_task)
    print("============================================")
    visualize_scenario_indp([0,size**2-1,1,1,1],30,0.0,UAV_task)

print("--- %s seconds ---" % (time.time() - start_time))
#visualize_scenario([0,15,1,1,1],100,UAV_fire_extinguish)