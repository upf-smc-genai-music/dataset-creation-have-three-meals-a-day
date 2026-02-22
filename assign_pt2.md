# Assignment PT2 – Texture Dataset Curation (Reference)

This document summarizes the official assignment requirements for creating
a sound texture dataset for generative modeling.

---

## 1. Goal

Create a dataset of sound textures with frame-level parameter annotations.

The dataset will be used to train models that map parameters → sound.

---

## 2. What is a Sound Texture?

A sound texture is a sound with statistically stable properties over time,
such as:

- Rain
- Wind
- Fire
- Noise-based textures
- Granular or percussive clouds

Not melody, not speech, not speaker identity.

---

## 3. Dataset Size

- Total duration: approximately 10–15 minutes of audio
- Each audio file must have:
  - One corresponding CSV annotation file
  - Same filename (except extension)

Example:

sound_01.wav  
sound_01.csv  

---

## 4. Parameters

Two types of parameters are allowed:

### Continuous
Numerical values, e.g.

- loudness
- brightness
- density
- speed
- roughness

### Categorical (optional)
Discrete classes, e.g.

- soft / medium / hard
- metal / wood / water

Recommended for simplicity:
Use only continuous parameters.

---

## 5. Frame Rate

Parameter annotation must be at:

75 frames per second (fps)

Each row in CSV corresponds to one frame.

---

## 6. CSV Format

First row: parameter names  
Following rows: values

Example:

density,loudness,brightness  
0.30,0.50,0.40  
0.31,0.51,0.42  

---

## 7. parameters.json

Describe each parameter and its range.

Example:

{
  "density": {
    "type": "continuous",
    "min": 0.0,
    "max": 1.0
  },
  "loudness": {
    "type": "continuous",
    "min": 0.0,
    "max": 1.0
  },
  "brightness": {
    "type": "continuous",
    "min": 0.0,
    "max": 1.0
  }
}

---

## 8. Folder Structure (Simple Option)

dataset/
├ parameters.json
├ sound_01.wav
├ sound_01.csv
├ sound_02.wav
├ sound_02.csv

---

## 9. Recommended Approach

- Use synthesizer or programmatic sound generation
- Generate audio and parameters simultaneously
- Avoid manual annotation

---

## 10. Submission

Submit:

- Dataset folder
- parameters.json
- README explaining:
  - Sound type
  - Parameters
  - How data was generated

Deadline: February 23

---

End of reference
