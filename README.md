# \# LesionIQ — AI Dermatology Decision Support System

# 

# Human-AI collaborative preliminary analysis tool for dermatologists.

# Built on ISIC 2019 dataset, EfficientNet-B4 + MLP fusion, MedGemma.

# 

# \---

# 

# \## Machine Setup

# | Machine | Use |

# |---|---|

# | Laptop (RTX 4050) | Development, testing |

# | Desktop (RTX 5070 Ti) | Layer 1–3 building |

# | Threadripper (2x H-series) | Model training |

# 

# \---

# 

# \## Layer Status

# | Layer | Name | Status |

# |---|---|---|

# | Layer 0 | Data Preparation | ✅ Complete |

# | Layer 1 | Generative Augmentation (SD-LoRA) | 🔲 Next |

# | Layer 2 | Model Training (EfficientNet-B4) | 🔲 Pending |

# | Layer 3 | Explainability (Grad-CAM + SHAP) | 🔲 Pending |

# | Layer 4 | Model Serving (TorchServe) | 🔲 Pending |

# | Layer 5 | API (FastAPI) | 🔲 Pending |

# | Layer 6 | Frontend (React Native) | 🔲 Pending |

# 

# \---

# 

# \## Layer 0 — Data Preparation (Complete)

# 

# \### Dataset

# \- ISIC 2019 Training Set — 23,257 dermoscopic images, 8 classes

# \- Source: https://challenge.isic-archive.com/data/#2019

# 

# \### Files in /data

# | File | Description |

# |---|---|

# | ISIC\_2019\_Training\_Metadata.csv | Raw patient metadata (27 columns) |

# | ISIC\_2019\_Training\_GroundTruth.csv | Class labels — MEL, NV, BCC, AK, BKL, DF, VASC, SCC |

# | field\_audit.txt | Full field audit — dtype, nulls, unique values |

# | layer0\_clean.csv | Merged, cleaned — 23,257 × 20 (audit cols kept) |

# | layer0\_model\_ready.csv | Fully encoded, zero nulls — 23,257 × 23 |

# | layer0\_train.csv | 18,707 images (80%) |

# | layer0\_val.csv | 2,244 images (10%) |

# | layer0\_test.csv | 2,306 images (10%) |

# 

# \### Features Selected for MLP Branch

# | Feature | Nulls | Strategy | Weight |

# |---|---|---|---|

# | age\_approx | 1.2% | Imputed with per-class median | HIGH |

# | sex | 0.9% | One-hot: male / female / unknown | MEDIUM |

# | anatom\_site\_general | 7.9% | One-hot: 8 categories + unknown | HIGH |

# | anatom\_site\_1 | 3.8% | Backup fallback for site | MEDIUM |

# 

# \### Fields Dropped (18 total)

# \- patient\_id (100% null), dermoscopic\_type (99.9% null)

# \- anatom\_site\_2/3/4/5/special (53–100% null, redundant)

# \- clin\_size\_long\_diam\_mm, family\_hx\_mm, personal\_hx\_mm (97%+ null)

# \- attribution, copyright\_license, image\_type, UNK (no signal)

# \- diagnosis\_4/5 (97–99% null)

# 

# \### Class Distribution

# | Class | Train | Risk |

# |---|---|---|

# | NV | 9,208 | Dominant — Focal Loss needed |

# | MEL | 3,351 | Moderate |

# | BCC | 2,721 | Moderate |

# | BKL | 1,783 | Moderate |

# | AK | 711 | 🟡 Sparse |

# | SCC | 526 | 🟠 Serious |

# | DF | 206 | 🔴 Critical — SD-LoRA target |

# | VASC | 201 | 🔴 Critical — SD-LoRA target |

# 

# \### Split Strategy

# \- Split by lesion\_id (not image) to prevent patient leakage across splits

# \- 826 images had no lesion\_id — treated as unique lesions

# 

# \### Age Median Validation (Clinically Confirmed)

# AK/BCC/SCC → 70, BKL → 65, MEL → 60, VASC → 55, DF → 50, NV → 45

