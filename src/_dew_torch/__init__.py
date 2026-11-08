"""Internal to Dew, no public API: what Dew's verification and its reference generators share that needs
torch alone. It sits beside `dew`, not in it, since `import dew` imports JAX (`dew._cache`), and the
generators run in torch-only environments with the checkout's `src` on their path."""
