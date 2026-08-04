# Writing Plan

## Central Message

This paper should not be written as an experiment diary. It should be written as
a controlled scientific argument:

1. AutoPhaseNN can be reproduced under a controlled PyTorch pipeline.
2. Replacing only the backbone with UMamba can improve or maintain some
   Fourier-domain metrics.
3. Those metrics do not necessarily imply correct real-space amplitude, phase,
   or support.
4. The key scientific issue is the strength and form of the spatial/physical
   prior, not only optimizer settings or learning rate.

## Paper Scope

### Include in the Main Paper

- AutoPhaseNN reproduction as the reference result.
- UMamba replacement under the same training and evaluation protocol.
- One representative diagnostic ablation if it explains the main failure mode.
- Visual comparison on the same sample IDs.
- A compact metric table with identical metric groups for every reported model.
- A discussion of Fourier-domain ambiguity and spatial support priors.

### Keep Out of the Main Paper

- Learning-rate tuning history.
- Checkpoint-loading and resume-training debugging.
- Gradient probes, direct optimization probes, and console-level diagnostics.
- Every failed threshold, upsampling, or mask variant.
- Long training logs.
- Implementation mistakes that were fixed before the controlled comparison.

These details can remain in lab notes or supplementary material, but they should
not distract from the paper's main argument.

## Suggested Narrative

### Introduction

Start from phase retrieval being ill-posed because only diffraction magnitude is
observed. Then introduce AutoPhaseNN as a neural solution that combines a model
prediction with a physics projection layer. Finally introduce the controlled
question: if the backbone is replaced by UMamba, does the same physical behavior
remain?

### Method

Describe the pipeline as a sequence:

1. diffraction modulus input;
2. network predicts amplitude and phase;
3. support mask is generated from amplitude;
4. complex object is formed and masked;
5. FFT produces predicted diffraction modulus;
6. far-field L1 is used for training;
7. real-space metrics are used for diagnosis.

Make clear that the UMamba experiment keeps steps 1 and 3-7 fixed whenever
possible, changing only step 2.

### Experiments

Use two main experiment groups and one optional diagnostic:

1. AutoPhaseNN reproduction from scratch.
2. UMamba replacement with the same physical post-processing.
3. Optional: one diagnostic ablation, such as soft support or center support,
   only if it clarifies the mechanism.

For small-sample experiments, state that validation is performed on the same
fixed samples to test fitting behavior rather than generalization.

### Results

Present figures before long interpretation:

1. AutoPhaseNN reproduction visualization.
2. UMamba hard-threshold visualization.
3. UMamba soft-threshold visualization.
4. Training curves.

Then present one table with the same metric groups for every model:

- FT: L1, relative L1, MSE/RMSE.
- Amplitude: L1, relative L1, MSE/RMSE.
- Phase: wrapped L1 on support.
- Support: IoU, Dice, predicted support fraction.

### Discussion

Explain the mismatch:

- Far-field magnitude loss is many-to-one with respect to real-space object
  structure.
- Support, amplitude localization, and phase localization are not directly
  supervised by far-field L1.
- AutoPhaseNN may encode a useful prior through decoder topology, upsampling,
  activation ranges, and hard support masking.
- UMamba may need an explicit hard spatial prior or an architecture-level prior
  to avoid physically invalid real-space solutions.

## Fill-in Checklist

- [ ] Add exact dataset size and split details.
- [ ] Add final checkpoint paths.
- [ ] Add metric values for AutoPhaseNN reproduction.
- [ ] Add metric values for UMamba hard threshold.
- [ ] Decide whether the diagnostic ablation should be soft support, center
      support, or another single representative variant.
- [ ] Add final visualization images to `figures/`.
- [ ] Add citations for AutoPhaseNN, phase retrieval, oversampling, Mamba, and
      UMamba.
