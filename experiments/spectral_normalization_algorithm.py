#!/usr/bin/env python


import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.autonotebook import tqdm


def alg_convergence(lists, tol=1e-5) -> list[int]:
    return [np.argmax(np.array(l) < tol) for l in lists]


if __name__ == "__main__":
    maxiter = 1000
    m, n = 128, 128
    A = torch.randn(m, n)
    U, S, V = torch.svd(A)
    u_true = U[:, 0]
    v_true = V[:, 0]
    s_true = S[0]
    G_true = torch.outer(u_true, v_true)
    print(f"{s_true=}")
    R = torch.arange(maxiter)

    u0 = torch.randn(m)
    v0 = torch.randn(n)

    u = u0.clone() / u0.norm()
    v = v0.clone() / v0.norm()
    s = A.mv(v).dot(u)

    f_u = []  # u residual
    f_v = []  # v residual
    f_r = []  # right residual
    f_l = []  # left residual
    f_x = []  # diff u
    f_y = []  # diff v
    f_R = []  # alt right residual
    f_L = []  # alt left residual
    s_r = []
    s_l = []
    f_G = []  # gradient residual

    for k in tqdm(R):
        u_old = u
        u = A.mv(v)
        u /= u.norm()
        s = A.mv(v).dot(u)
        s_l.append(abs(s - s_true) / s)
        f_l.append((A.t().mv(u) - s * v).norm() / s)
        f_R.append((A.mv(v) - s * u).norm() / s)
        assert s.sign() == +1
        # s = A.mv(v).dot(u)

        v_old = v
        v = A.T.mv(u)
        v /= v.norm()
        assert s.sign() == +1
        s = A.mv(v).dot(u)
        s_r.append(abs(s - s_true) / s)
        f_r.append((A.mv(v) - s * u).norm() / s)
        f_L.append((A.t().mv(u) - s * v).norm() / s)
        # s = A.mv(v).dot(u)
        g = torch.outer(u, v)
        f_G.append((g - G_true).norm())
        f_x.append((v - v_old).norm())
        f_y.append((u - u_old).norm())
        f_v.append(min((v - v_true).norm(), (v + v_true).norm()))
        f_u.append(min((u - u_true).norm(), (u + u_true).norm()))
        # f_r.append((A.mv(v) - s * u).norm())

    u = u0.clone() / u0.norm()
    v = v0.clone() / v0.norm()
    s = A.mv(v).dot(u)
    g_s = []  # s residual
    g_u = []  # u residual
    g_v = []  # v residual
    g_r = []  # right residual
    g_l = []  # left residual

    for k in tqdm(R):
        u_old = u
        v_old = v
        u = A.mv(v_old)
        u /= u.norm()

        v = A.T.mv(u_old)
        v /= v.norm()

        s = A.mv(v).dot(u)

        g_v.append(min((v - v_true).norm(), (v + v_true).norm()))
        g_u.append(min((u - u_true).norm(), (u + u_true).norm()))
        g_s.append(abs(abs(s) - s_true))
        g_r.append((A.mv(v) - s * u).norm())
        g_l.append((A.t().mv(u) - s * v).norm())

    # f_s,f_u,f_v,f_r,f_l,f_x,f_y

    print(alg_convergence([f_u, f_v, f_r, f_l, f_x, f_y]))
    print(alg_convergence([g_s, g_u, g_v, g_r, g_l]))

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.loglog(
        # fmt: off
        R, f_u, "-r", R, f_v, "-b", R, f_l, "-k", R, f_r, "-y", R, f_L, "--k", R, f_R, "--y",
        R, s_r, "-g", R, s_l, "-g", R, f_G, "-c",  # R, f_x, R, f_y,
        R, g_u, ":r", R, g_v, ":b", R, g_s, ":g", R, g_l, ":k", R, g_r, ":y",
        # fmt: on
    )
    ax.legend([
        "‖u-u⁎‖",
        "‖v-v⁎‖",
        "‖Aᵀu-σv‖",
        "‖Av-σu‖",
        "‖Aᵀu-σv‖/|σ|",
        "‖Av-σu‖/|σ|",
        "|σ-σ⁎|/|σ|",
        "|σ-σ⁎|/|σ|",
        "‖G-G⁎‖",
        "‖G-G⁎‖",
        "‖u-u⁎‖",
        "‖v-v⁎‖",
        "‖σ-σ⁎‖",
        "‖Aᵀu-σv‖",
        "‖Av-σu‖",
    ])
    ax.set_xlabel("k")
    ax.set_ylabel("residual")
    ax.set_title(
        "Takeaway 1: ‖Av-σu‖ and ‖Aᵀu-σv‖ are better convergence proxies than ‖u'-u‖"
        " and ‖v'-v‖.\nTakeaway 2: ‖Av-σu‖ and ‖Aᵀu-σv‖ need to be computed immediately"
        " after updating v and u respectively.\n            otherwise they are not good"
        " proxies for convergence.\nTakeaway 3: If we start with v update we can run in"
        " sign errors for sigma. If we start with u update this does not happen for"
        " some reason."
    )
    fig.suptitle(
        "Spectral norm algorithm convergence (dotted: parallel, solid: alternating)"
    )
    plt.show()
