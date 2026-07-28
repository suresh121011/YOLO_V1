    # Phase E — Before/After Dataset Comparison

    Generated 2026-07-28T03:38:29Z · **smoke (public 190)** → **full (public 7,426)**

    ## 1. Dataset size

    | Metric | smoke (public 190) | full (public 7,426) | Δ |
    |---|---:|---:|---:|
    | total_images | 17,888 | 24,352 | +6,464 |
    | total_labels | 17,888 | 24,352 | +6,464 |
    | total_boxes | 89,127 | 104,289 | +15,162 |

    ## 2. Per-class box counts

    | Class | smoke (public 190) | full (public 7,426) | Δ |
    |---|---:|---:|---:|
    | face | 2,144 | 4,110 | +1,966 |
    | water_bottle | 171 | 1,313 | +1,142 |
    | monitor | 117 | 1,253 | +1,136 |
    | knife | 2,556 | 3,681 | +1,125 |
    | toilet | 647 | 1,748 | +1,101 |
    | door | 1,674 | 2,770 | +1,096 |
    | book | 4,478 | 5,571 | +1,093 |
    | sink | 496 | 1,577 | +1,081 |
    | chair | 1,818 | 2,890 | +1,072 |
    | bed | 336 | 1,402 | +1,066 |
    | cupboard | 3,279 | 4,306 | +1,027 |
    | laptop | 477 | 1,370 | +893 |
    | person | 35,106 | 35,978 | +872 |
    | stove | 267 | 759 | +492 |
    | charger | 110 | 110 | +0 |
    | gas_cylinder | 8,089 | 8,089 | +0 |
    | medicine_bottle | 786 | 786 | +0 |
    | medicine_strip | 2,741 | 2,741 | +0 |
    | passport | 381 | 381 | +0 |
    | support_handle | 1,000 | 1,000 | +0 |
    | walking_stick | 3,562 | 3,562 | +0 |
    | wet_floor | 18,057 | 18,057 | +0 |
    | wire | 835 | 835 | +0 |

    ## 3. Public vs local contribution (accepted images)

    **smoke (public 190)**

    | Source | Accepted | Kind |
    |---|---:|---|
    | local_captures | 17,698 | local |
    | wider_face | 60 | public |
    | openimages | 57 | public |
    | coco | 53 | public |
    | negatives | 20 | public |
    | **public total** | **190** | |
    | **local total** | **17,698** | |

    **full (public 7,426)**

    | Source | Accepted | Kind |
    |---|---:|---|
    | local_captures | 16,926 | local |
    | coco | 4,951 | public |
    | openimages | 1,881 | public |
    | negatives | 497 | public |
    | wider_face | 97 | public |
    | **public total** | **7,426** | |
    | **local total** | **16,926** | |

    ## 4. Class imbalance

    | Metric | smoke (public 190) | full (public 7,426) | Δ |
    |---|---:|---:|---:|
    | gini_coefficient | 0.7183 | 0.6238 | -0.0945 |
    | imbalance_ratio | 319.1 | 327.1 | +8.0000 |
    | max_ratio | 0.3938873741963715 | 0.34498365120003066 | -0.0489 |
    | min_ratio | 0.0012341939030821188 | 0.00105476128834297 | -0.0002 |

    ## 5. QA and leakage

    | Check | Result |
    |---|---|
    | critical issues | 0 |
    | warnings | 505 |
    | train/val leakage | 0 |
    | train/test leakage | 0 |
    | license critical | False |
    | image-quality warnings | 3029 |
    | annotation sweep | 340 |
    | L4/L5 report sweep | 2 |

    ## 6. Completeness

    | Metric | smoke (public 190) | full (public 7,426) |
    |---|---:|---:|
    | images covered | 17888 | 24352 |
    | policies resolved | 5 | 5 |
    | unused policies | 0 | 0 |

    Per-source coverage (`by_policy`):

    | Source | smoke (public 190) | full (public 7,426) |
    |---|---:|---:|
    | coco | 53 | 4,951 |
    | local_captures | 17,698 | 16,926 |
    | negatives | 20 | 497 |
    | openimages | 57 | 1,881 |
    | wider_face | 60 | 97 |

    ## 7. Classes short of the dataset-v1.0.0 floor (200 instances)

    | Class | Count | Short by | Needs Indian-home capture |
    |---|---:|---:|---|
    | charger | 110 | 90 | no |

    Classes marked **yes** cannot be closed by any public source — they are the RG9 Indian-home capture requirement.
