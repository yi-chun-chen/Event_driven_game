include("models_two.jl")
include("method_two_genetic.jl")
#######

function cartesian_product(product_set)
        N = length(product_set)
        M = 1
        for i = 1 : N
                M *= length(product_set[i])
        end

        product_returned = [product_set[1]...]

        for class_index = 2 : N
                class_number = length(product_set[class_index])
                current_length = length(product_returned[:,1])

                enlarged_matrix = Array(Int64,current_length*class_number,class_index)

                if class_number == 1
                        enlarged_matrix[:,1:class_index-1] = product_returned
                        for i = 1 : current_length
                                enlarged_matrix[i,class_index] = product_set[class_index][1]
                        end
                else

                        for enlarge_times = 1 : class_number
                                enlarged_matrix[(enlarge_times-1)*current_length+1:enlarge_times*current_length,1:class_index-1] =
                                product_returned
                        end

                        for i = 1 : class_number * current_length
                                item_index = div(i-1,current_length) + 1
                                enlarged_matrix[i,class_index] = product_set[class_index][item_index]
                        end
                end
                product_returned = enlarged_matrix

        end

        return product_returned
end

# Default parameters

function UAV_fire_extinguish_multi_uav_one_fire()

    return UAV_fire_extinguish(
    5,
    2,
    1,
    (0.01,0.01,0.01), # fail rate
    (0.4,0.4,0.4), # message receive rate
    (13),   # fire position
    (1),   # fire reward
    ((0.0,1.0)),
    #((0.0,0.0,1.0)), # fire extinguish rate
    10, # horizon
    5  # planning horizon
    )
end

UAV_model_special = UAV_fire_extinguish_multi_uav_one_fire()

n_w = UAV_model_special.n_w

# Mapping between Cartesian product and integer through dictionary

mapping_car_to_one_special = Dict()
mapping_one_to_car_special = Dict()

car_big = cartesian_product(([1:1:26],[1:1:26]))
#car_big = cartesian_product(([1:1:26],[1:1:26],[1:1:26]))

for i = 1 : 26^(UAV_model_special.n_uav)

    tuple_state = zeros(Int64,UAV_model_special.n_uav+1)

    for i_uav = 1 : UAV_model_special.n_uav

        tuple_state[i_uav] = car_big[i,i_uav]

    end

    tuple_state[UAV_model_special.n_uav + 1] = 1
    tuple_state_1 = copy(tuple_state)
    tuple_state[UAV_model_special.n_uav + 1] = 2
    tuple_state_2 = copy(tuple_state)

    mapping_car_to_one_special[tuple(tuple_state_1...)] = i
    mapping_car_to_one_special[tuple(tuple_state_2...)] = i + 26^(UAV_model_special.n_uav)

    mapping_one_to_car_special[i] = tuple(tuple_state_1...)
    mapping_one_to_car_special[i + 26^(UAV_model_special.n_uav)] = tuple(tuple_state_2...)

end

car_big = 0

function fire_has_on_multi(lf,l_tuple)

    n_on = 0

    for i = 1 : length(l_tuple)

        if lf == l_tuple[i]

            n_on += 1

        end

    end

    return n_on

end


function transition_model_special(
    cart_product, # current state of the system in the form of cartesian product
    a,    # joint action
    UAV_model::UAV_fire_extinguish  # the default parameters for the model
    )

    # transition of the locations
    # no uncertainty while moving

    s_fail =  UAV_model.n_w^2 + 1

    if cart_product[1] == s_fail

        event_set_1 = [s_fail]
        prob_set_1 = [1.0]

    else

        l1_next = move_location(cart_product[1],a[1],UAV_model.n_w)

        event_set_1 = [ l1_next, s_fail]
        prob_set_1 =  [ 1 - UAV_model.t_fail[1], UAV_model.t_fail[1]]

    end

    (event_product,prob_product) = (event_set_1,prob_set_1)

    for i_uav = 2 : UAV_model.n_uav

        l_next = move_location(cart_product[i_uav],a[i_uav],UAV_model.n_w)

        if cart_product[i_uav] == s_fail

            event_set_2 = [s_fail]
            prob_set_2 = [1.0]

        else

            event_set_2 = [ l_next, s_fail]
            prob_set_2 =  [ 1 - UAV_model.t_fail[i_uav], UAV_model.t_fail[i_uav]]

        end

        (event_product,prob_product) = mix_distribution(event_product,prob_product,event_set_2,prob_set_2)

    end

    # transition of fire states
    # if fire exists and there is at least one UAV on the top of the fire, then fire might be extinguished

    for i_fire = 1 : UAV_model.n_fire

        fire_state = cart_product[UAV_model.n_uav + i_fire]

        if fire_state == 1 # no fire

            (event_product,prob_product) = mix_distribution(event_product,prob_product,[1],[1.0])


        else # fire exists

            l_f = UAV_model.l_fire[i_fire]

            n_on = fire_has_on_multi(l_f,cart_product[1:UAV_model.n_uav])


            if n_on == 0

                (event_product,prob_product) = mix_distribution(event_product,prob_product,[2],[1.0])

            else

                rate_put_down = UAV_model.e_fire[n_on]
                (event_product,prob_product) = mix_distribution(event_product,prob_product,[1,2],[rate_put_down,1.0-rate_put_down])

            end

        end

    end

    return(event_product,prob_product)

end

#######################

h = UAV_model_special.horizon

n_total_s_special = 26^(UAV_model_special.n_uav) * 2

function simulator_two_policy_special(
    current_state,                      # tuple
    policy,
    UAV_model::UAV_fire_extinguish      # problem info
    )


    label_current_state = mapping_car_to_one_special[current_state]

    u_last_time = zeros(Int64,UAV_model.n_uav)
    u_last_time_message = ones(Int64,UAV_model.n_uav) * label_current_state

    action_seq = Array(Int64,UAV_model.horizon,UAV_model.n_uav)

    a_joint = ones(Int64,UAV_model.n_uav)

    reward_sum = 0

    for t = 1 : UAV_model.horizon

        for i_uav = 1 : UAV_model.n_uav

            # one additional step from the last time receiving message
            u_last_time[i_uav] += 1

            # mapping the observation history to action
            if u_last_time[i_uav] <= UAV_model.horizon_plan
                a_joint[i_uav] = policy[i_uav][(u_last_time[i_uav]-1)*n_total_s + u_last_time_message[i_uav]]
            else
                a_joint[i_uav] = policy[i_uav][(UAV_model.horizon_plan -1)*n_total_s + u_last_time_message[i_uav]]
            end

        end


        # obtain next state
        next_event_tuple = transition_model_special(current_state,a_joint,UAV_model)
        next_state = sampling_events(next_event_tuple)
        label_next_state = mapping_car_to_one_special[next_state]

        # check whether each UAV recieve
        for i_uav = 1 : UAV_model.n_uav

            p = rand()

            if p < UAV_model.t_emit[i_uav]

                u_last_time[i_uav] = 0
                u_last_time_message[i_uav] = label_next_state

            end

        end

        # add reward
        reward_sum += reward_model(current_state,next_state,UAV_model) * 0.9 ^ (t-1)

        # state to next state
        current_state = next_state

    end

    return reward_sum

end

function fitness_function_special(n_times,current_state,policy,UAV_model)

    total_reward = 0

    for t_sim = 1 : n_times

        total_reward += simulator_two_policy_special(current_state,policy,UAV_model)

    end

    return total_reward / n_times
end

test_policy = Array(Any,UAV_model_special.n_uav)

for i = 1 : UAV_model_special.n_uav

    test_policy[i] = ones(Int64,h*n_total_s_special)

end


function genetic_process_two_uav_special(
    pop_size,
    iteration_times,
    current_state,
    UAV_model
    )

    h = UAV_model.horizon
    n_uav = UAV_model.n_uav

    ###################################
    #### Initialize the population ####
    ###################################

    # initial population for uav 1
    initial_pop = ones(Int64,pop_size,h*n_total_s_special)
    for i = 1 : pop_size
        for j = 1 : h*n_total_s_special
            initial_pop[i,j] = round(Int64,div(5*rand(),1))+1
        end
    end

    # the initially chosen policy for each uav
    policy = Array(Any,UAV_model.n_uav)
    for i_uav = 1 : n_uav
        policy[i_uav] = vec(initial_pop[1,:])
    end

    ##############################################
    #### Initialize the socre over population ####
    ##############################################

    # Score for intial population
    initial_score = Array(Float64,pop_size,UAV_model.n_uav)
    for i_uav = 1 : UAV_model.n_uav
        for i = 1 : pop_size
            policy_fixed_one = copy(policy)
            policy[i_uav] = initial_pop[i,:]
            initial_score[i,i_uav] = fitness_function_special(10,current_state,policy_fixed_one,UAV_model)
        end
    end

    #####################################################
    ## Probabililty over all policies in the two pools ##
    #####################################################

    prob_policy_uav = Array(Float64,pop_size,n_uav)

    # Probability for each policy in the population of each uav

    for i_uav = 1 : n_uav

        total_score = sum(initial_score[:,i_uav])

        if total_score == 0.0

            prob_policy_uav[:,i_uav] = ones(Float64,pop_size) / pop_size

        else

            prob_policy_uav = initial_score[:,i_uav] / total_score

        end

    end


    # Store best policy and its score for each iteration
    best_policy_each = Array(Any,n_uav)

    best_score = Array(Float64,iteration_times)
    best_score_current = -Inf

    best_policy_each_current = Array(Any,n_uav)

    for t = 1 : iteration_times

        ###########################################
        ##### The best two from the two pools #####
        ###########################################

        for i_uav = 1 : UAV_model.n_uav

            best_ind = indmax(initial_score[:,i_uav])
            best_policy_each_current[i_uav] = initial_pop[best_ind,:]

        end

        reward_by_best_two = fitness_function_special(10,current_state,best_policy_each_current,UAV_model)

        if reward_by_best_two > best_score_current
            best_score_current = reward_by_best_two
            best_policy_each = copy(best_policy_each_current)
        end

        best_score[t] = best_score_current
        println("iteration_times = ",t," reward = ",best_score[t])

        new_pop = ones(Int64,pop_size,h*n_total_s_special)

        for i_uav = 1 : UAV_model.n_uav

            for i = 1 : round(Int64,pop_size/2)
                first_source_index = sampling_index(prob_policy_uav[:,i_uav])
                second_source_index = sampling_index(prob_policy_uav[:,i_uav])

                first_string = initial_pop[first_source_index,:]
                second_string = initial_pop[second_source_index,:]

                cut_point = 1 + round(Int64,div(UAV_model.horizon_plan*n_total_s * rand(),1))

                new_pop[2*i - 1,:] = [first_string[1:cut_point]; second_string[cut_point+1:end]]
                new_pop[2*i    ,:] = [second_string[1:cut_point]; first_string[cut_point+1:end]]

            end

            # mutation
            for i = 1 : pop_size
                for j = 1 : h*n_total_s_special
                    if rand() < 0.005
                        new_pop[i,j] = 1 + round(Int64,div(5* rand(),1))
                    end
                end
            end

            # Prepare for new iteration
            initial_pop = copy(new_pop)
            for i = 1 : pop_size
                new_policy = copy(best_policy_each_current)
                new_policy[i_uav] = initial_pop[i,:]
                initial_score[i,i_uav] = fitness_function_special(10,current_state,new_policy,UAV_model)
            end


            total_score = sum(initial_score[:,i_uav])

            if total_score == 0.0

                prob_policy_uav[:,i_uav] = ones(Float64,pop_size) / pop_size

            else

                prob_policy_uav = initial_score[:,i_uav] / total_score

            end

        end



    end

    return (best_score)

end

genetic_process_two_uav_special(10,10,(3,23,11,2),UAV_model_special)