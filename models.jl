# Type of the scenario

type UAV_fire_extinguish

    n_w::Int64 # width of the square field
    n_uav::Int64 # number of the UAV
    n_fire::Int64 # number of the fire position
    t_fail # failure rate of each UAV
    t_emit # rate to emit message
    l_fire # fire position of each fire
    r_fire # reward of puting down each fire
    horizon # horizon of the planning problem

end

# Default parameters

function UAV_fire_extinguish()

    return UAV_fire_extinguish(4,2,3,(0.01,0.01),0.5,(4,10,13),(1,2,3),6)
end

UAV_model = UAV_fire_extinguish()

n_w = UAV_model.n_w

# coordinate mapping

function two_d_to_one(x,n_w)
    return (x[2]-1)*n_w + x[1]
end

function one_d_to_two(x,n_w)
    x_1 = div(x-1,n_w) + 1
    x_2 = x % n_w
    if x_2 == 0
        x_2 = 4
    end

    return (x_2,x_1)
end

function check_boundary(x,n_w)

    if x < 1
        return 1
    elseif x > n_w
        return n_w
    else
        return x
    end
end


# cartesian product and joint distribution

function mix_distribution(
    events_1,
    probs_1,
    events_2,
    probs_2
    )

    new_dim = length(events_1) * length(events_2)
    new_event = Array(Any,new_dim)
    new_prob = Array(Float64,new_dim)

    count = 0

    for e_1 = 1 : length(events_1)
        for e_2 = 1 : length(events_2)

            count += 1
            new_event[count] = [events_1[e_1]; events_2[e_2]]
            new_prob[count] = probs_1[e_1] * probs_2[e_2]
        end
    end

    return(new_event,new_prob)
end



# The environment is a 4 by 4 grid world, and there are two agents

n_w = 4
n_grid = n_w^2

n_a = 5

# fire position

l_f_1 = 3
l_f_2 = 10
l_f_3 = 13

# Mapping between Cartesian product and integer through dictionary

mapping_car_to_one = Dict()
mapping_one_to_car = Dict()

counter = 0

for f3 = 1 : 2
    for f2 = 1 : 2
        for f1 = 1 : 2
            for l2 = 1 : n_grid + 1
                for l1 = 1 : n_grid + 1
                    counter += 1
                    mapping_car_to_one[(l1,l2,f1,f2,f3)] = counter
                    mapping_one_to_car[counter] = (l1,l2,f1,f2,f3)
                end
            end
        end
    end
end

###############################################################
###################  TRANSITION MODEL #########################
###############################################################

# Move
# The basic movement stimulated by the action

function move_location(l::Int64,a::Int64,n_w::Int64)

    if l > n_w^2
        return l
    else

        (x1,x2) = one_d_to_two(l,n_w)

        x1_next = 0; x2_next = 0;
        l_next = 0

        if a == 4 # right

                x1_next = x1 + 1
                x2_next = x2

                if x1_next > 4
                    x1_next = 4
                end

        elseif a == 3 # left

                x1_next = x1 - 1
                x2_next = x2

                if x1_next < 1
                    x1_next = 1
                end

        elseif a == 2 # down

                x1_next = x1
                x2_next = x2 - 1

                if x2_next < 1
                    x2_next = 1
                end

        elseif a == 1 # up

                x1_next = x1
                x2_next = x2 + 1

                if x2_next > 4
                    x2_next = 4
                end

        else

                x1_next = x1
                x2_next = x2

        end

        l_next = two_d_to_one((x1_next,x2_next),n_w)

        return l_next

    end

end


# Whether there is a UAV on the top of the fire

function fire_has_on(lf,l1,l2)

    if (lf != l1) * (lf != l2)

        return false

    else

        return true

    end

end


# fire extinguish rate.

function fire_ex_combo_effect(lf,l1,l2)

    if lf == l1 == l2

        return 0.9 # When there are two UAV on the top of fire, the rate is 0.8.

    elseif lf == l1

        return 0.9 # When there is on UAV on the top of fire, the rate is 0.2.

    elseif lf == l2

        return 0.9

    else

        return 0.0 # fire remains

    end

end


# Transition model

function transition_model(
    cart_product, # current state of the system in the form of cartesian product
    a1::Int64,    # action taken by UAV 1
    a2::Int64,    # action taken by UAV 2
    UAV_model::UAV_fire_extinguish  # the default parameters for the model
    )

    # transition of the locations
    # no uncertainty while moving

    if cart_product[1] == 17

        event_set_1 = [17]
        prob_set_1 = [1.0]

    else

    l1_next = move_location(cart_product[1],a1,UAV_model.n_w)

    event_set_1 = [ l1_next, 17]
    prob_set_1 =  [ 1 - UAV_model.t_fail[1], UAV_model.t_fail[1]]

    end

    l2_next = move_location(cart_product[2],a2,UAV_model.n_w)

    if cart_product[2] == 17

        event_set_2 = [17]
        prob_set_2 = [1.0]

    else

    event_set_2 = [ l2_next, 17]
    prob_set_2 =  [ 1 - UAV_model.t_fail[2], UAV_model.t_fail[2]]

    end

    (event_product,prob_product) = mix_distribution(event_set_1,prob_set_1,event_set_2,prob_set_2)

    # transition of fire states
    # if fire exists and there is at least one UAV on the top of the fire, then fire might be extinguished

    for i_fire = 1 : UAV_model.n_fire

        fire_state = cart_product[UAV_model.n_uav + i_fire]

        if fire_state == 1 # no fire

            (event_product,prob_product) = mix_distribution(event_product,prob_product,[1],[1.0])


        else # fire existes

            l_f = UAV_model.l_fire[i_fire]
            l_1 = cart_product[1]
            l_2 = cart_product[2]

            if fire_has_on(l_f,l_1,l_2)

                rate_put_down = fire_ex_combo_effect(l_f,l_1,l_2)
                (event_product,prob_product) = mix_distribution(event_product,prob_product,[1,2],[rate_put_down,1-rate_put_down])

            else

                (event_product,prob_product) = mix_distribution(event_product,prob_product,[2],[1.0])

            end

        end

    end


    return(event_product,prob_product)

end

################################################################
######################  REWARD MODEL ###########################
################################################################

function reward_model(
    cart_product,
    cart_product_next,
    UAV_model::UAV_fire_extinguish  # the default parameters for the model
    )

    rw = 0

    for i_f = 1 : UAV_model.n_fire

        if cart_product[UAV_model.n_uav + i_f] == 2 && cart_product_next[UAV_model.n_uav + i_f] == 1

            rw += UAV_model.r_fire[i_f]

        end

    end

    return rw

end



################################################################
###################  OBSERVATION MODEL #########################
################################################################

################### Old version ################################
################### Do not read ################################


# Noise on fire

function old_obs_fire(
    fi::Int64,   # fire state
    l_fi::Int64, # fire position
    l_j::Int64,  # UAV location
    n_w::Int64   # width
    )

    p_l_fi = one_d_to_two(l_fi,n_w)
    p_l_j = one_d_to_two(l_j,n_w)

    p_wrong = (((p_l_fi[1] - p_l_j[1])^2 + (p_l_fi[2] - p_l_j[2])^2)^0.5)/(n_w * 1.5)

    if fi == 1

        return([1,2],[1-p_wrong,p_wrong])

    else

        return([1,2],[p_wrong,1-p_wrong])

    end

end


# observation exactly reflects the state

function obs_model(
    cart_product,                   # current state
    a1,                             # action by 1
    a2,                             # action by 2
    UAV_model::UAV_fire_extinguish  # the default parameters for the model
    )

    x = rand()

    if x < UAV_model.t_emit

        o_1 = cart_product[1]
        o_2 = cart_product[2]

    else

        o_1 = [cart_product...]
        o_2 = [cart_product...]

    end

    return (o_1,o_2)

end


