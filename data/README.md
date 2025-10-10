
# Data Download
You can download the dataset from [this Google Drive link](https://drive.google.com/drive/folders/1e8Mu5sXwcl6p58eFTiBXgmLM3Pa50myw?usp=drive_link).

# Overview

This dataset is collected from **Kundur's Two‑Area Four‑Machine System**, a standard benchmark model widely used in power‑system stability studies. The data capture detailed electrical measurements under both normal and faulted conditions and are intended to support the development of AI‑based **fault detection and diagnosis** methods.

The system model is located at `matlab/KundurTwoAreaSystemModified.slx`, and the simulation script is `matlab/runSystem.m`.

The dataset includes synchronized voltage and current phasors from PMUs, instantaneous waveforms, generator dynamics, and power‑flow metrics across key buses and machines. It is structured so machine‑learning models can learn patterns, detect anomalies, and classify different types of faults effectively.

# Data Description: Time‑Series & Image Fault Dataset
This repository houses structured simulation data for fault analysis in electrical power systems. The dataset is split into two ZIP archives—one with raw time‑series signals (`.csv`) and another with their corresponding visual plots (`.png`).

# Directory Layout
The dataset is divided into two main archives:

- `timeseries.zip` — raw time‑series signals representing electrical measurements over time.
- `images.zip` — image representations of the same time‑series data, generated after feature selection, normalization/scaling, and other preprocessing steps. See **MH‑WMG: A Multi‑Head Wavelet‑Based MobileNet with Gated Linear Attention for Power Grid Fault Diagnosis** for full details.

```text
data/
├── timeseries.zip
└── images.zip
```

Each archive contains six sub‑folders representing different fault‑area scenarios:

```text
A1/
A2/
A3/
A4/
A5/
A6/
```

## `timeseries.zip`
Inside every `A*` folder you’ll find the following directories:

| Folder | Description |
|--------|-------------|
| **b1** | Bus 1 – Generator bus for Area 1 (power flows toward Area 2). |
| **b2** | Bus 2 – Generator bus for Area 2 (power flows toward Area 1). |
| **current** | Electric‑current data, including phasor (magnitude, angle, frequency) and instantaneous values. |
| **machines** | Synchronous generators (M1–M4): rotor speed, air‑gap power, angle deviation. |
| **power** | Real (active) power measurements, typically in per‑unit (pu). |
| **voltage** | Voltage data including phasor quantities and instantaneous waveforms. |
| **faulted_pmu_i** | Current phasor measurements from PMUs in the faulted area. |
| **faulted_pmu_v** | Voltage phasor measurements from PMUs in the faulted area. |
| **faulted_vi** | Time‑domain voltage and current waveforms related to the faulted area. |

---

Each folder contains **588** files. Column descriptions:

**b1**

- `b1_v_1`, `b1_v_2`, `b1_v_3` — instantaneous phase‑to‑neutral voltages (phases A, B, C) at Bus 1.  
- `b1_i_1`, `b1_i_2`, `b1_i_3` — instantaneous phase‑to‑neutral currents at Bus 1.

**b2**

- `b2_v_1`, `b2_v_2`, `b2_v_3` — instantaneous phase‑to‑neutral voltages at Bus 2.  
- `b2_i_1`, `b2_i_2`, `b2_i_3` — instantaneous phase‑to‑neutral currents at Bus 2.

**current**

- `iabc_g1_magnitude`, `iabc_g1_angle`, `iabc_g1_f` — current‑phasor magnitude, angle, and frequency from PMUs.

**machines**

- `machines_pa_1` – `machines_pa_4` — air‑gap/accelerating power (pu) of Generators M1–M4.  
- `machines_w_1` – `machines_w_4` — rotor speed of generators (1 pu = synchronous speed).  
- `machines_dtheta_1`, `machines_dtheta_2` — rotor‑angle deviation (deg) relative to Machine 1.

**power**

- `p_g1_magnitude_1`, `p_g1_magnitude_2` — per‑unit real‑power outputs for Generators 1 & 2.

**voltage**

- `vabc_g1_magnitude`, `vabc_g1_angle`, `vabc_g1_f` — voltage‑phasor magnitude, angle, and frequency from PMUs.

**faulted_pmu_i**

- `faulted_pmu_i_m`, `faulted_pmu_i_a`, `faulted_pmu_i_f` — current‑phasor magnitude, angle, frequency in the faulted area.

**faulted_pmu_v**

- `faulted_pmu_v_m`, `faulted_pmu_v_a`, `faulted_pmu_v_f` — voltage‑phasor magnitude, angle, frequency in the faulted area.

**faulted_vi**

- `faulted_v_1`, `faulted_v_2`, `faulted_v_3` — three‑phase voltage waveforms in the faulted area.  
- `faulted_i_1`, `faulted_i_2`, `faulted_i_3` — three‑phase current waveforms in the faulted area.


Filename format is:

```
<distance>_<a>_<b>_<c>_<g>.csv
```

- `<distance>`  Fault location (e.g., `0.5` ⇒ 0.5 km along the line)  
- `<a> <b> <c>` Phase‑fault flags (`on` / `off`)  
- `<g>`         Ground‑fault flag (`on` / `off`)

## `images.zip`
Contains image‑based representations of the time‑series data.

Within each `A*` folder you’ll find **588 PNG** files (one per simulation case). Each image is a **5 × 4** grid of sub‑plots, each corresponding to a specific feature.

### Sub‑plot order (left→right, top→bottom)
`b1_i_1`, `b1_v_1`, `b1_v_3`, `b2_v_3`,
`faulted_i_3`, `faulted_pmu_i_a`, `faulted_pmu_i_m`, `faulted_pmu_v_a`,
`faulted_v_1`, `faulted_v_2`, `faulted_v_3`, `machines_dtheta_1`,
`machines_dtheta_2`, `machines_pa_1`, `machines_pa_2`, `machines_w_1`, 
`machines_w_2`, `vabc_g1_angle`, `vabc_g1_f`, `vabc_g1_magnitude`

**Filename format**

```
<distance>_<a>_<b>_<c>_<g>_plot.png
```

# Citation
If you use this dataset, please cite:

> Alkhanafseh, Y., Akinci, T. C., Martínez‑Morales, A. A., Şeker, S., & Ekici, S. (2025). **MH‑WMG: A Multi‑Head Wavelet‑Based MobileNet with Gated Linear Attention for Power Grid Fault Diagnosis**. *IEEE Access*.
