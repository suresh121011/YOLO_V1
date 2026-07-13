# YAML Configuration Templates

## Purpose

Complete YAML configuration templates for data, training, deployment, rules, and feature flags.

## Dependencies

Reads: None (reference document)

Used By:
- python_examples.md
- training_scripts.md

Related:
- ../02_technical_architecture_specification/api_contracts.md
- ../02_technical_architecture_specification/feature_flags.md

---

## 1. Dataset Configuration (`configs/data.yaml`)

```yaml
path: ./data/processed
train: images/train
val: images/val

nc: 23
names:
  0: person
  1: face
  2: medicine_strip
  3: medicine_bottle
  4: water_bottle
  5: knife
  6: stove
  7: gas_cylinder
  8: passport
  9: book
  10: charger
  11: wire
  12: laptop
  13: monitor
  14: cupboard
  15: door
  16: chair
  17: bed
  18: toilet
  19: sink
  20: wet_floor
  21: walking_stick
  22: support_handle
```

## 2. Training Configuration (`configs/training/yolo11n_config.yaml`)

```yaml
model:
  base: yolo11n.pt
  task: detect

training:
  data: configs/data.yaml
  epochs: 150
  imgsz: 640
  batch: 16
  patience: 25
  optimizer: AdamW
  lr0: 0.001
  lrf: 0.01
  momentum: 0.937
  weight_decay: 0.0005
  warmup_epochs: 5
  warmup_bias_lr: 0.1
  close_mosaic: 15

augmentation:
  hsv_h: 0.015
  hsv_s: 0.5
  hsv_v: 0.3
  degrees: 5.0
  translate: 0.1
  scale: 0.4
  flipud: 0.0
  fliplr: 0.5
  mosaic: 0.8
  mixup: 0.1
  copy_paste: 0.1

output:
  project: models
  name: yolo11n
  exist_ok: true
  save: true
  save_period: 10
  val: true
  plots: true
  verbose: true
```

## 3. Risk Rules Configuration (`configs/risk_rules.yaml`)

```yaml
rules:
  - id: knife_near_person
    condition: "detected(knife) AND detected(person)"
    severity: HIGH
    cooldown_seconds: 60
    message_en: "Please be careful, there is a knife nearby."
    message_hi: "कृपया सावधान रहें, पास में चाकू है।"

  - id: stove_unattended
    condition: "detected(stove) AND absent_for(person, 30)"
    severity: CRITICAL
    cooldown_seconds: 30
    message_en: "The stove appears to be on without anyone nearby. Please check."
    message_hi: "चूल्हा बिना किसी के पास चालू लग रहा है। कृपया जांचें।"

  - id: wet_floor_hazard
    condition: "detected(wet_floor)"
    severity: HIGH
    cooldown_seconds: 120
    message_en: "The floor appears to be wet. Please walk carefully."
    message_hi: "फर्श गीला लग रहा है। कृपया सावधानी से चलें।"

  - id: wire_tripping_hazard
    condition: "detected(wire) AND detected(person)"
    severity: HIGH
    cooldown_seconds: 120
    message_en: "There are wires on the floor that could be a tripping hazard."
    message_hi: "फर्श पर तार हैं जिनसे ठोकर लग सकती है।"

  - id: gas_cylinder_check
    condition: "detected(gas_cylinder) AND NOT detected(stove)"
    severity: INFO
    cooldown_seconds: 600
    message_en: "A gas cylinder is visible. Please ensure the regulator is properly connected."
    message_hi: "गैस सिलेंडर दिख रहा है। कृपया सुनिश्चित करें कि रेगुलेटर ठीक से लगा है।"

  - id: medicine_reminder
    condition: "any_of([medicine_strip, medicine_bottle])"
    severity: INFO
    cooldown_seconds: 300
    message_en: "Medicine is visible. Have you taken your medication today?"
    message_hi: "दवाई दिख रही है। क्या आपने आज अपनी दवाई ली है?"
```

## 4. Feature Flags (`configs/feature_flags.yaml`)

```yaml
feature_flags:
  vlm_enabled: false
  hindi_tts: false
  caregiver_sync: false
  thermal_monitoring: true
  active_learning: true
  rule_hot_reload: false
  debug_overlay: false
  performance_logging: true
```

## 5. Per-Class Confidence Thresholds (`configs/class_thresholds.yaml`)

```yaml
class_thresholds:
  knife: 0.20
  stove: 0.25
  gas_cylinder: 0.22
  wire: 0.22
  wet_floor: 0.20
  medicine_strip: 0.25
  medicine_bottle: 0.25
  passport: 0.30
  person: 0.30
  face: 0.35
  # All other classes use default: 0.25
```

---

Previous: None (start here)

Next: [python_examples.md](./python_examples.md)

Related: [../02_technical_architecture_specification/api_contracts.md](../02_technical_architecture_specification/api_contracts.md)
