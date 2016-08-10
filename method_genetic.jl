include("models.jl")

###############################################
############### Sampling ######################
###############################################

# Sampling from events

function sampling_events(event_tuple)

    # event_tuple[1] is the list of events
    # event_tuple[2] is the list of corresponding probability

    x = rand()

    for i_event = 1 : length(event_tuple[1])

        x = x - event_tuple[2][i_event]

        if x < 0

            return tuple(event_tuple[1][i_event]...)

        end

    end

    return tuple(event_tuple[1][end]...)
end

# Sampling event index from a probability distribution

function sampling_index(prob)

    x = rand()

    for i_event = 1 : length(prob)

        x = x - prob[i_event]

        if x < 0

            return i_event

        end

    end

    return i_event

end


###################################################
############### Simulator #########################
###################################################

h = UAV_model.horizon

n_total_s = 17 * 17 * 8


function simulator_two_policy(
    current_state,                      # tuple
    policy_1,
    policy_2,
    UAV_model::UAV_fire_extinguish      # problem info
    )


    label_current_state = mapping_car_to_one[current_state]


    u1_last_time = 0
    u1_last_time_message = label_current_state
    u2_last_time = 0
    u2_last_time_message = label_current_state

    action_seq_1 = Array(Int64,UAV_model.horizon)
    action_seq_2 = Array(Int64,UAV_model.horizon)

    reward_sum = 0

    for t = 1 : UAV_model.horizon

        # For UAV 1

        # one additional step from the last time receiving message
        u1_last_time += 1
        # mapping the observation history to action
        if u1_last_time <= UAV_model.horizon_plan
            a_1 = policy_1[(u1_last_time-1)*n_total_s + u1_last_time_message]
        else
            a_1 = policy_1[(UAV_model.horizon_plan -1)*n_total_s + u1_last_time_message]
        end

        # For UAV 2

        # one additional step from the last time receiving message
        u2_last_time += 1
        # mapping the observation history to action
        if u2_last_time <= UAV_model.horizon_plan
            a_2 = policy_2[(u2_last_time-1)*n_total_s + u2_last_time_message]
        else
            a_2 = policy_2[(UAV_model.horizon_plan -1)*n_total_s + u2_last_time_message]
        end

        # obtain next state
        next_event_tuple = transition_model(current_state,a_1,a_2,UAV_model)
        next_state = sampling_events(next_event_tuple)
        label_next_state = mapping_car_to_one[next_state]

        # check whether each UAV recieve message
        (p1,p2) = (rand(),rand())
        if p1 < UAV_model.t_emit[1]
            u1_last_time = 0
            u1_last_time_message = label_next_state

        end
        if p2 < UAV_model.t_emit[2]
            u2_last_time = 0
            u2_last_time_message = label_next_state
        end

        # add reward
        reward_sum += reward_model(current_state,next_state,UAV_model) * 0.9 ^ (t-1)

        # state to next state
        current_state = next_state

    end

    return reward_sum

end

function simulator_two_policy_trajectory(
    current_state,                      # tuple
    policy_1,
    policy_2,
    UAV_model::UAV_fire_extinguish      # problem info
    )


    label_current_state = mapping_car_to_one[current_state]


    u1_last_time = 0
    u1_last_time_message = label_current_state
    u2_last_time = 0
    u2_last_time_message = label_current_state

    action_seq_1 = Array(Int64,UAV_model.horizon)
    action_seq_2 = Array(Int64,UAV_model.horizon)

    reward_sum = 0
    traject = Array(Any,UAV_model.horizon+1)
    traject[1] = current_state
    println(current_state)

    for t = 1 : UAV_model.horizon

        # For UAV 1

        # one additional step from the last time receiving message
        u1_last_time += 1
        # mapping the observation history to action
        if u1_last_time <= UAV_model.horizon_plan
            a_1 = policy_1[(u1_last_time-1)*n_total_s + u1_last_time_message]
        else
            a_1 = policy_1[(UAV_model.horizon_plan -1)*n_total_s + u1_last_time_message]
        end

        # For UAV 2

        # one additional step from the last time receiving message
        u2_last_time += 1
        # mapping the observation history to action
        if u2_last_time <= UAV_model.horizon_plan
            a_2 = policy_2[(u2_last_time-1)*n_total_s + u2_last_time_message]
        else
            a_2 = policy_2[(UAV_model.horizon_plan -1)*n_total_s + u2_last_time_message]
        end

        # obtain next state
        next_event_tuple = transition_model(current_state,a_1,a_2,UAV_model)
        next_state = sampling_events(next_event_tuple)
        label_next_state = mapping_car_to_one[next_state]

        # check whether each UAV recieve message
        (p1,p2) = (rand(),rand())
        if p1 < UAV_model.t_emit[1]
            u1_last_time = 0
            u1_last_time_message = label_next_state
            println("1 receives message")

        end
        if p2 < UAV_model.t_emit[2]
            u2_last_time = 0
            u2_last_time_message = label_next_state
            println("2 receives message")
        end

        # add reward
        reward_sum += reward_model(current_state,next_state,UAV_model) * 0.9 ^ (t-1)

        # state to next state
        current_state = next_state
        traject[t+1] = current_state
        println(current_state)

    end

    return reward_sum

end

##########################################
########## Fitness function ##############
##########################################


function fitness_function(n_times,current_state,policy_1,policy_2,UAV_model)

    total_reward = 0

    for t_sim = 1 : n_times

        total_reward += simulator_two_policy(current_state,policy_1,policy_2,UAV_model::UAV_fire_extinguish)

    end

    return total_reward / n_times
end

###########################################
########### Genetic Algorithm #############
###########################################

function genetic_process_two_uav(
    pop_size,
    iteration_times,
    current_state,
    UAV_model
    )

    h = UAV_model.horizon

    ###################################
    #### Initialize the population ####
    ###################################

    # initial population for uav 1
    initial_pop_u1 = ones(Int64,pop_size,h*n_total_s)
    for i = 1 : pop_size
        for j = 1 : h*n_total_s
            initial_pop_u1[i,j] = round(Int64,div(5*rand(),1))+1
        end
    end

    # initial population for uav 2
    initial_pop_u2 = ones(Int64,pop_size,h*n_total_s)
    for i = 1 : pop_size
        for j = 1 : h*n_total_s
            initial_pop_u2[i,j] = round(Int64,div(5*rand(),1))+1
        end
    end

    # the initially chosen policy for each uav
    policy_1_fixed = vec(initial_pop_u1[1,:])
    policy_2_fixed = vec(initial_pop_u2[1,:])

    ##############################################
    #### Initialize the socre over population ####
    ##############################################

    # Score for intial population for uav 1
    initial_score_u1 = Array(Float64,pop_size)
    for i = 1 : pop_size
        initial_score_u1[i] = fitness_function(50,current_state,initial_pop_u1[i,:],policy_2_fixed,UAV_model)
    end

    # Score for intial population for uav 2
    initial_score_u2 = Array(Float64,pop_size)
    for i = 1 : pop_size
        initial_score_u2[i] = fitness_function(50,current_state,policy_1_fixed,initial_pop_u2[i,:],UAV_model)
    end

    #####################################################
    ## Probabililty over all policies in the two pools ##
    #####################################################

    # Probability for each policy in the population of uav1
    total_score_u1 = sum(initial_score_u1)
    if total_score_u1 == 0.0
        prob_policy_u1 = ones(Float64,pop_size) / pop_size
    else
        prob_policy_u1 = initial_score_u1 / total_score_u1
    end

    # Probability for each policy in the population of uav2
    total_score_u2 = sum(initial_score_u2)
    if total_score_u2 == 0.0
        prob_policy_u2 = ones(Float64,pop_size) / pop_size
    else
        prob_policy_u2 = initial_score_u2 / total_score_u2
    end

    # Store best policy and its score for each iteration
    best_policy_u1 = 0
    best_policy_u2 = 0

    best_score = Array(Float64,iteration_times)
    best_score_current = -Inf
    best_p1_current = 0
    best_p2_current = 0

    for t = 1 : iteration_times

        ###########################################
        ##### The best two from the two pools #####
        ###########################################

        best_ind_1 = indmax(initial_score_u1)
        best_policy_u1 = initial_pop_u1[best_ind_1,:]
        best_ind_2 = indmax(initial_score_u2)
        best_policy_u2 = initial_pop_u2[best_ind_2,:]

        reward_by_best_two = fitness_function(20,current_state,best_policy_u1,best_policy_u2,UAV_model)

        if reward_by_best_two > best_score_current
            best_score_current = reward_by_best_two
            best_p1_current = best_policy_u1
            best_p2_current = best_policy_u2
        end

        best_score[t] = best_score_current
        println("iteration_times = ",t," reward = ",best_score[t])

        #######################################
        ##### Fix policy 2 and gene the 1 #####
        #######################################


        # new population
        new_pop_u1 = ones(Int64,pop_size,h*n_total_s)

        # cross over
        for i = 1 : round(Int64,pop_size/2)
            first_source_index = sampling_index(prob_policy_u1)
            second_source_index = sampling_index(prob_policy_u1)
            first_string = initial_pop_u1[first_source_index,:]
            second_string = initial_pop_u1[second_source_index,:]

            cut_point = 1 + round(Int64,div(UAV_model.horizon_plan*n_total_s * rand(),1))

            new_pop_u1[2*i - 1,:] = [first_string[1:cut_point]; second_string[cut_point+1:end]]
            new_pop_u1[2*i    ,:] = [second_string[1:cut_point]; first_string[cut_point+1:end]]
        end

        # mutation
        for i = 1 : pop_size
            for j = 1 : h*n_total_s
                if rand() < 0.005
                    new_pop_u1[i,j] = 1 + round(Int64,div(5* rand(),1))
                end
            end
        end

        # Prepare for new iteration for u1
        initial_pop_u1 = copy(new_pop_u1)
        for i = 1 : pop_size
            initial_score_u1[i] = fitness_function(20,current_state,initial_pop_u1[i,:],best_policy_u2,UAV_model)
        end

        total_score_u1 = sum(initial_score_u1)
        if total_score_u1 == 0.0
            prob_policy_u1 = ones(Float64,pop_size) / pop_size
        else
            prob_policy_u1 = initial_score_u1 / total_score_u1
        end

        #######################################
        ##### Fix policy 1 and gene the 2 #####
        #######################################


        # new population
        new_pop_u2 = ones(Int64,pop_size,h*n_total_s)

        # cross over
        for i = 1 : round(Int64,pop_size/2)
            first_source_index = sampling_index(prob_policy_u2)
            second_source_index = sampling_index(prob_policy_u2)
            first_string = initial_pop_u2[first_source_index,:]
            second_string = initial_pop_u2[second_source_index,:]

            cut_point = 1 + round(Int64,div(UAV_model.horizon_plan*n_total_s * rand(),1))

            new_pop_u2[2*i - 1,:] = [first_string[1:cut_point]; second_string[cut_point+1:end]]
            new_pop_u2[2*i    ,:] = [second_string[1:cut_point]; first_string[cut_point+1:end]]
        end

        # mutation
        for i = 1 : pop_size
            for j = 1 : h*n_total_s
                if rand() < 0.001
                    new_pop_u2[i,j] = 1 + round(Int64,div(5* rand(),1))
                end
            end
        end

        # Prepare for new iteration for u2
        initial_pop_u2 = copy(new_pop_u2)
        for i = 1 : pop_size
            initial_score_u2[i] = fitness_function(20,current_state,best_policy_u1,initial_pop_u2[i,:],UAV_model)
        end

        total_score_u2 = sum(initial_score_u2)
        if total_score_u2 == 0.0
            prob_policy_u2 = ones(Float64,pop_size) / pop_size
        else
            prob_policy_u2 = initial_score_u2 / total_score_u2
        end


    end

    return (best_score,best_p1_current,best_p2_current)

end
