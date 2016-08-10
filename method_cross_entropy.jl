include("models.jl")
include("method_genetic.jl")


#######################################################
##### Sample trajectory from stochastic policies ######
#######################################################

function simulate_trajectory_stochastic(current_state,SP1,SP2,UAV_model)

    label_current_state = mapping_car_to_one[current_state]


    u1_last_time = 0
    u1_last_time_message = label_current_state
    u2_last_time = 0
    u2_last_time_message = label_current_state

    action_seq_1 = Array(Int64,UAV_model.horizon)
    action_seq_2 = Array(Int64,UAV_model.horizon)

    trajectory = Array(Int64,UAV_model.horizon,6)


    for t = 1 : UAV_model.horizon

        trajectory[t,1] = label_current_state
        trajectory[t,2] = u1_last_time
        trajectory[t,3] = u2_last_time

        # For UAV 1

        # one additional step from the last time receiving message
        u1_last_time += 1
        # mapping the observation history to action
        if u1_last_time <= UAV_model.horizon_plan
            a_1 = sampling_index( vec(SP1[u1_last_time][u1_last_time_message,:]) )
        else
            a_1 = sampling_index( vec(SP1[UAV_model.horizon_plan][u1_last_time_message,:]) )
        end

        # For UAV 2

        # one additional step from the last time receiving message
        u2_last_time += 1
        # mapping the observation history to action
        if u2_last_time <= UAV_model.horizon_plan
            a_2 = sampling_index( vec(SP2[u2_last_time][u2_last_time_message,:]) )
        else
            a_2 = sampling_index( vec(SP2[UAV_model.horizon_plan][u2_last_time_message,:]) )
        end

        trajectory[t,4] = a_1
        trajectory[t,5] = a_2

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
        trajectory[t,6] = reward_model(current_state,next_state,UAV_model)


        # state to next state
        current_state = next_state
        label_current_state = label_next_state

    end

    return trajectory

end

SP1 = Array(Any,UAV_model.horizon_plan)
SP2 = Array(Any,UAV_model.horizon_plan)

for tt = 1 : UAV_model.horizon_plan
    SP1[tt] = 0.2 * ones(Float64,n_total_s,5)
    SP2[tt] = 0.2 * ones(Float64,n_total_s,5)
end

simulate_trajectory_stochastic((1,16,2,2,2),SP1,SP2,UAV_model)

##################################################
## Cumulative reward after current time ##########
##################################################



function reward_transform_cumulate(trajectory,UAV_model)

    new_version = copy(trajectory)

    for t = 1 : UAV_model.horizon

        new_version[t,6] = sum(trajectory[t:end,6])

    end

    return new_version

end

#####################################################
##### What each player will see in a trajectory #####
#####################################################


function trajectory_be_seen_by_each_player(trajectory,UAV_model)

    trajectory_1 = Array(Int64,length(trajectory[:,1]),4)
    trajectory_2 = Array(Int64,length(trajectory[:,1]),4)

    for n_event = 1 : length(trajectory[:,1])

        trajectory_1[n_event,1] = trajectory[n_event-trajectory[n_event,2],1]
        trajectory_1[n_event,2] = trajectory[n_event,2]
        trajectory_1[n_event,3] = trajectory[n_event,4]
        trajectory_1[n_event,4] = trajectory[n_event,6]

        trajectory_2[n_event,1] = trajectory[n_event-trajectory[n_event,3],1]
        trajectory_2[n_event,2] = trajectory[n_event,3]
        trajectory_2[n_event,3] = trajectory[n_event,5]
        trajectory_2[n_event,4] = trajectory[n_event,6]

    end

    return (trajectory_1,trajectory_2)

end

#####################################
##### From population to polict #####
#####################################

function population_to_stochastic_policy(population_trajectory_for_player,UAV_model,elite_rate)


    SP = Array(Any,UAV_model.horizon_plan)

    for delay = 1: UAV_model.horizon_plan

        # Initialize stochastic policy

        SP[delay] = 0.2 * ones(Float64,n_total_s,5)

        # Count event and label the position of (state,delay,action)

        event_list = 0
        event_number_collect = Dict() # (event state label) -> (position in the population)

        for n_event = 1 : length(population_trajectory_for_player[:,1])

            if population_trajectory_for_player[n_event,2] == delay-1

                state_label = population_trajectory_for_player[n_event,1]
                event_list = [event_list ; state_label]
                event_number_collect[ state_label ] = [get(event_number_collect,state_label,0); n_event]

            end

        end

        event_list_unique = unique(event_list)

        #if delay == 2; return (event_list_unique,event_number_collect);end

        # list what happened in each (state,delay,action) tuple

        for n_event_tuple = 2 : length(event_list_unique)

            state_in_message = event_list[n_event_tuple]

            number_of_state = length(event_number_collect[state_in_message]) - 1

            #return "yes"

            # list of all information as an array

            events_not_sort = Array(Int64,number_of_state,4)

            for n_event_of_state = 2 : number_of_state+1

                label_in_big_pool = event_number_collect[state_in_message][n_event_of_state]
                events_not_sort[n_event_of_state-1,:] = population_trajectory_for_player[label_in_big_pool,:]

            end


            # sort all events by the reward
            v = sortperm(events_not_sort[:,4],rev=true)
            events_sort = events_not_sort[v,:]

            elite_number = round(Int64,div(length(events_sort[:,1]) * elite_rate,1)) + 1

            action_count = zeros(Int64,5)

            for i_elite = 1 : elite_number

                action_count[events_sort[i_elite,3]] += 1

            end

            for a = 1 : 5

                SP[delay][state_in_message,a] = action_count[a]/elite_number

            end

            #println(state_in_message,SP[delay][state_in_message,:])

        end

    end

    return SP

end

ex_population_trajectory = [
    [2279  0  0  2  5  2];
    [2297  0  1  4  1  2];
    [2279  0  0  4  4  2];
    [2302  2  1  3  1  2];
    [1723  0  0  2  5  0];
    [1723  0  0  2  5  0];
    [1719  1  1  5  2  0]]

#traject_test = simulate_trajectory_stochastic((1,16,2,2,2),SP1,SP2,UAV_model)

#tra_two = trajectory_be_seen_by_each_player(traject_test,UAV_model)

traject_test_1 = [
    [2279  0  1  0];
    [2279  0  2  5];
    [2304  0  2  0];
    [2304  1  5  0];
    [2300  0  3  0];
    [2300  1  1  0];
    [2304  0  3  0]]

ex_outcome = population_to_stochastic_policy(traject_test_1,UAV_model,0.2,1)


######################################################
##### Generate a trajectory pool for each player #####
######################################################

function generate_pool(n_trajectory,current_state,SP1,SP2,UAV_model)

    pool_trajectory = Array(Int64,n_trajectory * UAV_model.horizon,6)
    pool_trajectory_1 = Array(Int64,n_trajectory * UAV_model.horizon,4)
    pool_trajectory_2 = Array(Int64,n_trajectory * UAV_model.horizon,4)

    for i_tra = 1 : n_trajectory

        current_traj = simulate_trajectory_stochastic(current_state,SP1,SP2,UAV_model)
        current_traj_r = reward_transform_cumulate(current_traj,UAV_model)
        traj = trajectory_be_seen_by_each_player(current_traj_r,UAV_model)


        for step = 1 : UAV_model.horizon

            pool_trajectory_1[(i_tra-1)*UAV_model.horizon + step,:] = traj[1][step,:]
            pool_trajectory_2[(i_tra-1)*UAV_model.horizon + step,:] = traj[2][step,:]

        end

    end

    return (pool_trajectory_1,pool_trajectory_2)

end

###################################
####### Cross entropy method ######
###################################


function cross_entropy_times(times_iteration,n_trajectory,current_state,UAV_model,elite_rate)

    # Initialize stochastic policies

    SP1 = Array(Any,UAV_model.horizon_plan)
    SP2 = Array(Any,UAV_model.horizon_plan)

    for tt = 1 : UAV_model.horizon_plan
        SP1[tt] = 0.2 * ones(Float64,n_total_s,5)
        SP2[tt] = 0.2 * ones(Float64,n_total_s,5)
    end

    pool_trajectory = generate_pool(n_trajectory,current_state,SP1,SP2,UAV_model)


    # cross entropy procedure

    for t_iteration = 1 : times_iteration

        println("times of iteration = ",t_iteration)
        test = simulate_trajectory_stochastic(current_state,SP1,SP2,UAV_model)
        println("test reward = ", sum(test[:,6]))

        SP1 = population_to_stochastic_policy(pool_trajectory[1],UAV_model,elite_rate)

        pool_trajectory = generate_pool(n_trajectory,current_state,SP1,SP2,UAV_model)

        SP1 = population_to_stochastic_policy(pool_trajectory[2],UAV_model,elite_rate)

        pool_trajectory = generate_pool(n_trajectory,current_state,SP1,SP2,UAV_model)

    end

    return (SP1,SP2)
end

SP = cross_entropy_times(50,2000,(1,16,2,2,2),UAV_model,0.3);

test = simulate_trajectory_stochastic((1,16,2,2,2),SP[1],SP[2],UAV_model)