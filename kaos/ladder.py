from keras.layers import merge, Lambda

def infer_mu(lp_par):
    l_mu, l_var, p_mu, p_var = lp_par
    q_mu = (l_mu/l_var + p_mu/p_var)/(1/l_var + 1/p_var)
    return q_mu

def infer_mu_shape(input_shape):
    input_shape = input_shape[0]
    assert len(input_shape) == 2
    return input_shape

def infer_mu0(l_par):
    l_mu, l_var = l_par
    q_mu = (l_mu/l_var)/(1/l_var + 1)
    return q_mu

def infer_mu0_shape(input_shape):
    return input_shape[0]

def infer_var(lp_var):
    l_var, p_var = lp_var
    q_var = 1/(1/l_var + 1/p_var)
    return q_var

def infer_var_shape(input_shape):
    input_shape = input_shape[0]
    assert len(input_shape) == 2  # only valid for 2D tensors
    return input_shape

def infer_var0(l_var):
    q_var = 1/(1/l_var + 1)
    return q_var

def infer_var0_shape(input_shape):
    return input_shape

def infer_ladder(l_par, p_par):
    lp_var = l_par[1], p_par[1]
    lp_par = l_par + p_par

    if p_par == (0, 1):
        q_mu = merge(l_par, mode=infer_mu0, output_shape=infer_mu0_shape)
        q_var = Lambda(infer_var0, infer_var0_shape)(l_par[1])
    else:
        q_mu = merge(lp_par, mode=infer_mu, output_shape=infer_mu_shape)
        q_var = merge(lp_var, mode=infer_var, output_shape=infer_var_shape)
    return q_mu, q_var
