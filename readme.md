# Flow Matching

> [Arxiv paper](https://arxiv.org/pdf/2210.02747)
> [Longer form document](https://scontent-iad3-1.xx.fbcdn.net/v/t39.2365-6/469963300_2320719918292896_7950025307614718519_n.pdf?_nc_cat=108&ccb=1-7&_nc_sid=3c67a6&_nc_ohc=-9E3TOQyAosQ7kNvgHKbDJm&_nc_zt=14&_nc_ht=scontent-iad3-1.xx&_nc_gid=AKfHjk5atjiPuU_uAXtA3L5&oh=00_AYD2IDG9Cc314P8-9CS34AFBdvXkM4LMR5Y7IcZWx0tqTg&oe=6788F042)

Applied to generating overhead imagery using (for now) pixel state spaces.

## My Intuition of Flow Matching & Some Notes From the Paper

Basically you train a denoising autoencoder-like model to point from points off the data manifold to points on the data manifold. Like in diffusion / denoising-score-matching, the model is made aware of the current `t` (so that it knows how to control step-size, and can estimate how far current input is from data manifold from where it thinks it should currently be, etc.).

The training process randomly select points according to the proposal distribution `x0 = p(x)` to points in the data distribution `q ~= {x1_i}`. Using the optimal transport scheme, every training example input is formed by an affine combination of a randomly generated `x0` and and a randomly selected true sample `x1` roughly as `x = lerp(x0, x1, t)` (not exactly this expression). The model is given `x` and `t` and asked to point to `x1`.

The "Longer form document" link above helps to build some intution. I think the crazy thing FM accomplishes is that pairing a random `x0` with a randomly selected `x1` works. In low dimensional spaces, and if you wanted to model high frequency signals, I think you'd run into issues where pairing *a wrong* x0 with an x1 would cause the velocity field to be non-smooth and self-intersecting. But I suppose the catch is that with high-dimensional data (images certainly are that) the issue resolves itself because of all the extra dimensions available (as `D` grows, two lines in `R^D` have less and less chance of being near eachother?)


### Similar to Diffusion
 - To train an FM+OT model, we pair a random sample `x0` with a randomly selected true data sample `x1`, then randomly generate a time `t`, and have the model estimate the velocity as the vector between the `t` interpolated two samples.
 - As opposed to diffusion, where you sample a random `x1`, a random `t`, then add `t`-dependent amount of noise to `x1`, and train the model estimate the velocity to push us back from the noisy sample to `x1`.

### As `t -> 0`, you get truly random samples
With diffusion, by making the noise multiplier large enough when `t` approaches zero, you usually have `x1 + noise * schedule(t)` look completely random. But really, it is not.

Whereas with FM, as `t` approaches zero, it really is random because the weight of `x1` goes to zero.
I think they highlight this in the paper but I doubt it matters in practice as long as `schedule(t)` is chosen properly.

### Straighter trajectories
The paper talks about how FM w/ OT results in straigher trajectories than FM w/ Diffusion or Diffusion alone. I certainly understand how OT leads to straighter tracjectories, but I don't completey get why diffusion does not. I think I have a glimpse of the intuition, and it has to do with Diffusion's `schedule` being non-linear while OT is simple linear/affine interpolation.

### ODE Solvers
The paper uses a midpoint solver. I plan to start out with just Euler's method, and am curious to see how bad that is relative to better ODE integration schemes.
