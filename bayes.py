import torch
import pyro
import pyro.distributions as dist
import pyro.distributions.constraints as constraints

# %% Helper


def summary(x):
    return pd.DataFrame(x.detach().cpu().numpy()).describe()


INTERFACE_LIST = list(data["interface"].unique())
data["i"] = data["interface"].apply(lambda i: INTERFACE_LIST.index(i))


# %% Define Bayesian model


pyro.clear_param_store()


# See Ch. 19 of Doing Bayesian Data Analysis, 2nd Edition
def model(interfaces, times=None):
    # Baseline
    beta_0 = pyro.sample("beta_0", dist.Normal(3, 0.5))
    sigma_y = pyro.sample("sigma_y", dist.HalfNormal(0.25))

    # # Group deflections
    with pyro.plate("J", len(INTERFACE_LIST)):
        # Sharing, more complex model:
        sigma_beta = pyro.sample("sigma_beta", dist.Gamma(10, 25))
        # No sharing, simpler model:
        # sigma_beta = 0.5
        beta_j = pyro.sample("beta_j", dist.Normal(0, sigma_beta))
        mu_j = pyro.deterministic("mu_j", beta_0 + beta_j)

    # # Individual predictions
    with pyro.plate("N", len(interfaces)):
        pyro.sample(
            "y_i",
            dist.LogNormal(mu_j[interfaces], sigma_y),
            obs=times,
        )


interfaces = torch.tensor(correct["i"].values)
times = torch.tensor(correct["taskTime"].values)

pyro.render_model(
    model,
    model_args=(interfaces,),
    filename="model.pdf",
    render_distributions=True,
)

prior = pyro.infer.Predictive(
    model,
    num_samples=10000,
)(interfaces, times=None)

for k in prior:
    print("==========")
    print("Parameter", k)
    print(summary(prior[k]))

# %% Fit Bayesian model

pyro.clear_param_store()

auto_guide = pyro.infer.autoguide.AutoNormal(model)
adam = pyro.optim.Adam({"lr": 0.01})
elbo = pyro.infer.Trace_ELBO()
svi = pyro.infer.SVI(model, auto_guide, adam, elbo)

losses = []
for step in range(10000):
    loss = svi.step(interfaces, times=times)
    losses.append(loss)
    if step % 500 == 0:
        print("Elbo loss: {}".format(loss))

posterior = pyro.infer.Predictive(
    model,
    guide=auto_guide,
    num_samples=10000,
)(interfaces, times=None)

for k in posterior:
    print("==========")
    print("Parameter", k)
    print(summary(posterior[k]))

# %% Compute 95% HDI for time taken

m = posterior["mu_j"]
hdi = pd.DataFrame(
    {
        "avg": m.mean(0).detach().cpu().numpy(),
        "lo": m.kthvalue(int(len(m) * 0.025), dim=0)[0].detach().cpu().numpy(),
        "hi": m.kthvalue(int(len(m) * 0.975), dim=0)[0].detach().cpu().numpy(),
    }
)
hdi["interface"] = hdi.index.map(lambda i: INTERFACE_LIST[i])

order = [
    "Unfamiliar Only",
    "Translation",
    "Probe–Mapping",
    "Probe–Components",
    "Probe–Translation Steps",
    "Probe–NL Translation Explanation",
    "NL",
]

alt.layer(
    alt.Chart()
    .mark_point(color="black", filled=True)
    .encode(
        x="avg:Q",
        y=alt.Y("interface:N", sort=order),
    ),
    alt.Chart()
    .mark_errorbar(extent="ci")
    .encode(
        x="lo:Q",
        x2="hi:Q",
        y=alt.Y("interface:N", sort=order),
    ),
    data=hdi,
).interactive().save("chart3.html")
