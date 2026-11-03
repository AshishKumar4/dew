# A racing drone that learns to fly from its camera

A new tutorial, [A racing drone that flies from its camera](../../tutorials/14-camera-only-drone-racing.ipynb), trains a first-person racing drone from scratch. The pilot sees a 64-pixel camera image and the rates, rotor speeds and next gates a real drone senses. It learns from no demonstrations and reads no privileged state while it flies.

The recipe is SkyDreamer's (Verraest et al., [arXiv 2510.14783](https://arxiv.org/abs/2510.14783)), which is DreamerV3 with an informed world model. The agent learns a world model of what its camera sees and how the drone moves, and its latent learns to predict the drone's position, velocity and orientation, which the policy never reads. The policy then learns to fly inside that model's imagination. The race runs in JAX on the GPU and a ray tracer draws its images (HyperLux). One record, `runs/skydreamer/run.json`, describes the whole run, and `dew train` trains it.

On one Colab G4 the run takes 125,000 updates over a million environment steps, which was 110 minutes on the recorded run. The tutorial then flies the trained and the untrained policy from the same starts. From the evaluation starts of SkyDreamer's Table III, the trained pilot passes 44.4 gates in a 2000-step episode and flies all 2000 steps without crashing. From held-out training starts it passes 36.2. The untrained policy passes none and crashes within about 500 steps. The tutorial's last cell films the trained pilot at 256 pixels: four drones, each seen through its own camera.

Run it [in Colab](https://colab.research.google.com/github/AshishKumar4/dew/blob/main/tutorials/14-camera-only-drone-racing.ipynb) on a G4, or read its recorded outputs on the tutorial's page.
