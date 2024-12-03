def quadratic_model(s, f0, df, hf):
    s = s.reshape(-1, 1)
    f0 = f0.reshape(-1, 1)
    # print(f0.shape, s.T.shape, df.shape, hf.shape, s.shape)
    m = f0 + s.T @ df + .5 * s.T @ (hf @ s)
    # print(m.shape) #(1, 1)
    return m.flatten()