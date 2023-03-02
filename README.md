# Liquidity Optimization in Constant Function Market Makers

This repository hosts supplementary code for the paper ["Finding the Right Curve: Optimal Design of Constant Function Market Makers"]
(https://arxiv.org/abs/2212.03340)
by [Mohak Goyal](https://sites.google.com/view/mohakg/home), 
[Geoffrey Ramseyer](http://www.scs.stanford.edu/~geoff/), 
[Ashish Goel](https://web.stanford.edu/~ashishg/),
and [David Mazières](http://www.scs.stanford.edu/~dm/).

## Usage

- The script `cfmm_lp2.py` computes for any given belief function (on the future valuations of assets)
	optimal CFMM designs and liquidity allocations.
	- Invoke as `python3 cfmm_lp2.py`
	- Custom belief functions can be added in `psi_func_custom(px, py)`.
	- The framework is general; alternative objectives (as in e.g. Section 7 of the paper, when accounting for liquidity provider
	fee revenue and divergence loss) follow from modifications to the objective functions.

- The script `lp_functions.py` provides auxiliary functions and implements various belief functions.

## Dependencies

The optimization toolkit relies on [CVXPY](https://www.cvxpy.org/), which can be installed via
```
python3 -m pip install cvxpy
```

