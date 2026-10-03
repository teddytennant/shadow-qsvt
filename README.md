Classical simulation, in JAX, of shadow QSVT observable estimates from Krylov moments (arXiv:2609.40167).

```
python3 -m pytest
```

What does not match the paper: there are no block-encoding circuits, no Hadamard test (moments are exact dense matvecs), no history-state circuit (the observable-aware procedure is the classical history-Gram reduction of Algorithm 6, not a circuit), and no asymptotic gate counts. The shadow sampler is a classical Pauli channel on a system small enough to simulate, not a fault-tolerant circuit.
