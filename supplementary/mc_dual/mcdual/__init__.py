"""mcdual -- supplementary Monte-Carlo tools for Robbins' problem.  NOT part of the proof.

Modules
-------
grid   : non-uniform grid on [0, 1] (fine near 0).
dp2    : k = 2 ("best-two memory") approximate value function on a grid.
dp3    : k = 3 ("best-three memory") approximate value function on a grid.
dual   : Monte-Carlo information-relaxation dual lower bound and primal upper bound.

All results produced with these tools are floating-point Monte-Carlo estimates with standard errors;
none of them is certified.
"""
