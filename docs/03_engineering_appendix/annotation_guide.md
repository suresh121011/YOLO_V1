# Annotation Guidelines & Class Definitions

## Purpose

Annotation rules, per-class guidance, capture variations checklist, and formal class definitions for all 23 classes.

## Dependencies

Reads:
- dataset_templates.md

Used By:
- release_checklists.md

Related:
- ../01_executive_implementation_plan/implementation_phases.md

---

## 10.1 General Annotation Rules

1. **Every visible instance** of any of our 23 classes MUST be annotated
2. Bounding boxes must be **tight** (maximum 5px padding around object)
3. **Occluded objects** with ≥ 25% visible area must be annotated
4. Objects **smaller than 20×20 pixels** may be skipped
5. A single image may contain **1 to 15+ annotations**
6. When photographing any object, **annotate ALL other visible classes** in the same frame

---

## 10.2 Per-Class Annotation Guidance

| Class | Annotation Notes |
|:------|:----------------|
| `person` | Full body preferred; partial (torso+head minimum); multiple people = multiple boxes |
| `face` | Annotate face region only; minimum 32×32 pixels |
| `medicine_strip` | Include full strip; annotate each strip separately if multiple |
| `medicine_bottle` | Include cap; do not confuse with water/sauce bottles |
| `water_bottle` | Clear plastic bottles with water; do not include sealed packages |
| `knife` | Include handle + blade; kitchen knives, utility knives; scissors excluded |
| `stove` | Include burner area + knobs; gas stoves mandatory |
| `gas_cylinder` | Full cylinder including regulator attachment area |
| `passport` | Blur personal details in images used for training |
| `wet_floor` | Annotate the WET AREA (shiny/reflective surface), not entire floor |
| `wire` | Annotate visible wire runs; tangled masses = one box around cluster |
| `walking_stick` | Full stick from handle to tip |
| `support_handle` | Wall-mounted grab bars and railings |

---

## 10.3 Capture Variations Checklist

```
For EACH object, capture with variations across:

☐ Lighting:    bright daylight | overcast | tubelight/CFL | dim evening | night with flash
☐ Angle:       top-down 90° | 45° diagonal | eye-level 0° | low angle -30°
☐ Distance:    close-up <30cm | medium 30cm-1m | far 1-3m | very far >3m
☐ Placement:   on table | on floor | on bed | on shelf | in hand | on counter
☐ Context:     alone/clean | cluttered | partially occluded | near similar objects
☐ Background:  plain wall | patterned tile/marble | carpet/rug | kitchen counter
☐ Camera:      landscape | portrait | slight motion blur | sharp focus
```

---

## 10.4 Formal Class Definitions

| ID | Name | Definition | Positive Examples | Negative Examples (do NOT annotate) |
|:---|:-----|:-----------|:-----------------|:-----------------------------------|
| 0 | `person` | Any human figure, full or partial (head+torso minimum) | Standing, sitting, walking elderly person | Mannequin, photo/poster of person, doll |
| 1 | `face` | Human face region, minimum 32×32px | Front-facing, profile, partially occluded | Photo on wall, TV face, mask |
| 2 | `medicine_strip` | Foil/blister pack containing medicine tablets | Indian brand strips, loose strips | Candy wrapper, chewing gum pack |
| 3 | `medicine_bottle` | Pharmaceutical bottle with cap (syrup, pills) | Cough syrup, vitamin bottle, prescription bottle | Water bottle, sauce bottle, shampoo |
| 4 | `water_bottle` | Clear plastic/glass bottle containing water | PET bottle, steel bottle, glass bottle | Medicine bottle, oil bottle, sealed carton |
| 5 | `knife` | Any cutting blade with handle | Kitchen knife, utility knife, bread knife | Scissors, razor, blade without handle |
| 6 | `stove` | Gas burner/cooktop surface | Indian gas stove (Prestige, Pigeon), countertop burner | Induction cooktop (V2), microwave, oven |
| 7 | `gas_cylinder` | Pressurized LPG gas cylinder | HP, Bharat Gas, Indane cylinders | Fire extinguisher, oxygen tank, propane BBQ |
| 8 | `passport` | Indian passport booklet | Open or closed passport | Other documents, ID cards, books |
| 9 | `book` | Any bound book or notebook | Textbook, notebook, diary | Magazine, newspaper, tablet/e-reader |
| 10 | `charger` | Phone/device charger with plug or cable | USB charger, laptop charger, Indian 3-pin plug | Wall socket (without charger), standalone cable |
| 11 | `wire` | Exposed electrical wire or extension cord | Extension cord, tangled wires, charging cables on floor | Behind-wall wiring, properly channeled cables |
| 12 | `laptop` | Open or closed laptop computer | Open laptop on table, closed laptop | Tablet, desktop monitor, keyboard alone |
| 13 | `monitor` | TV screen or desktop monitor | LED TV, desktop monitor, old CRT TV | Laptop screen (annotate as laptop), phone screen |
| 14 | `cupboard` | Freestanding or built-in storage furniture | Indian almirah, Godrej steel cupboard, wooden wardrobe | Open shelf without door, kitchen cabinet (V2) |
| 15 | `door` | Interior house door, open or closed | Wooden door, bathroom door, glass door | Window, gate, curtain partition |
| 16 | `chair` | Any seating with back support | Dining chair, office chair, plastic chair, recliner | Stool (no back), sofa (annotate if ambiguous) |
| 17 | `bed` | Sleeping surface with mattress | Single bed, double bed, cot, diwan | Sofa-cum-bed (annotate as bed when flat) |
| 18 | `toilet` | Toilet fixture | Western commode, Indian squat toilet | Bidet, urinal |
| 19 | `sink` | Water basin fixture | Kitchen sink, bathroom sink/basin | Bathtub, bucket |
| 20 | `wet_floor` | Visibly wet or reflective floor surface | Freshly mopped floor, water spill, bathroom wet floor | Shiny marble floor (when dry), polished wood |
| 21 | `walking_stick` | Mobility aid cane or walking stick | Standard cane, quad cane, hiking stick (if in home) | Umbrella, broom handle |
| 22 | `support_handle` | Wall-mounted grab bar or railing | Bathroom grab bar, staircase railing, bed rail | Door handle, drawer pull, towel rack |

---

## 10.5 Dataset Changelog Template

```markdown
# Dataset Changelog

## [dataset-v1.1.0] — 2026-08-XX
### Added
- 300 new images for `wet_floor` class (bathroom and kitchen)
- 200 new images for `gas_cylinder` class (HP, Bharat Gas brands)

### Changed
- Re-annotated 50 `medicine_strip` images (tighter bounding boxes)

### Fixed
- Removed 12 duplicate images from train set

### QA Results
- Missing labels: 0 | Invalid bboxes: 0 | Duplicates removed: 12
```

---

Previous: [api_reference.md](./api_reference.md)

Next: [release_checklists.md](./release_checklists.md)

Related: [../01_executive_implementation_plan/implementation_phases.md](../01_executive_implementation_plan/implementation_phases.md)
