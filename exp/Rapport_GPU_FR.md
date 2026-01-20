# Rapport d'Expérimentation sur le Calcul Parallèle GPU

**Nom :** ZHANG Haochen
**Date :** Novembre 2025
**Cours :** Pratique GPU INSA 2025
**Encadrant :** Emmanuel Quémener

---

## Résumé

Cette série d'expériences explore en profondeur les principes, les méthodes de programmation et les caractéristiques de performance du calcul parallèle sur GPU à travers des exercices progressifs. Les expériences couvrent la détection matérielle, la programmation de base OpenCL/CUDA jusqu'à l'analyse de performance de codes applicatifs réels.

**Note :** En raison de contraintes de temps, les sections IV à X n'ont pas pu être réalisées.

---

## I. Investigation et Analyse de l'Environnement Matériel

### 1.1 Objectifs

- Identifier le matériel GPU présent dans le système
- Comprendre les différences architecturales entre GPU et CPU
- Maîtriser les méthodes d'obtention des paramètres de performance GPU

### 1.2 Procédure Expérimentale

#### 1.2.1 Détection du Matériel GPU

Utilisation de plusieurs outils Linux pour détecter le matériel GPU :

```bash
# Détection des périphériques PCI
lspci -nn | egrep '(VGA|3D)'

# Consultation des messages du noyau
dmesg | grep -i nvidia

# Consultation des modules chargés
lsmod | grep nvidia

# Consultation des fichiers de périphériques
ls -l /dev/nvidia*

# Utilisation de l'outil nvidia-smi
nvidia-smi
```

**Résultats :**

GPU détectés :
- **GPU Principal :** NVIDIA GeForce GTX 1080 Ti
  - Cœurs CUDA : 3584
  - Mémoire : 11 Go GDDR5X
  - Fréquence de base : 1480 MHz
  - Fréquence mémoire : 11 GHz
  - Compute Capability : 6.1

- **GPU Secondaire :** NVIDIA Quadro K420
  - Cœurs CUDA : 192
  - Mémoire : 2 Go GDDR3
  - Fréquence de base : 780 MHz
  - Fréquence mémoire : 1,8 GHz
  - Compute Capability : 3.0

#### 1.2.2 Détection de la Plateforme OpenCL

La commande `clinfo -l` a détecté 5 dispositifs OpenCL :

```
Platform #0: AMD Accelerated Parallel Processing
 └── Device #0: Intel Xeon E5-2637 v4 @ 3.50GHz (CPU)

Platform #1: Portable Computing Language
 └── Device #0: pthread-Intel Xeon E5-2637 v4 (CPU)

Platform #2: NVIDIA CUDA
 ├── Device #0: GeForce GTX 1080 Ti (GPU)
 └── Device #1: Quadro K420 (GPU)

Platform #3: Intel OpenCL
 └── Device #0: Intel Xeon E5-2637 v4 (CPU)
```

**Analyse :**
- Le CPU dispose de 3 implémentations OpenCL (AMD, PoCL, Intel), avec des différences de performance significatives
- L'implémentation Intel est généralement la plus rapide sur CPU
- Le GPU est accessible via la plateforme NVIDIA CUDA
- Nombre d'unités de traitement : CPU = 16 (8 cœurs × 2 threads), GTX 1080 Ti = 28 unités SM

### 1.3 Conclusions

1. **Diversité matérielle :** Le système dispose de multiples dispositifs de calcul, permettant des comparaisons de performance
2. **Outils de détection :** Maîtrise des outils essentiels lspci, nvidia-smi, clinfo
3. **Compréhension architecturale :** Le GPU possède bien plus d'unités de calcul parallèle que le CPU (3584 vs 16)

---

## II. Programmation de Base Python/OpenCL

### 2.1 Objectifs

- Maîtriser le modèle de programmation de base de PyOpenCL
- Comprendre le mécanisme de transfert de données hôte-dispositif
- Analyser les caractéristiques de performance d'une simple addition vectorielle

## II-III. Expériences de Programmation (Sections 2.2 - 3.2)

Le code source, les résultats d'exécution et les analyses des expériences suivantes se trouvent dans les sous-dossiers correspondants du dossier `code/` :


## IV - X. Expériences Non Réalisées

En raison de contraintes de temps, les expériences des sections IV à X n'ont pas pu être réalisées.

---

## Annexe A : Configuration de l'Environnement Expérimental

### A.1 Configuration Matérielle

```
Nom d'hôte : opencluster2.cbp.ens-lyon.fr
CPU : 2 × Intel Xeon E5-2637 v4 @ 3.50GHz
    - Nombre de cœurs : 8 cœurs physiques (16 cœurs logiques, hyperthreading)
    - Cache : 15 Mo L3
    - Mémoire : 64 Go DDR4-2400 ECC

GPU1 : NVIDIA GeForce GTX 1080 Ti
    - Cœurs CUDA : 3584
    - Architecture : Pascal (GP102)
    - Compute Capability : 6.1
    - Mémoire : 11 Go GDDR5X @ 11 GHz
    - Bande passante mémoire : 484 Go/s
    - TDP : 250 W

GPU2 : NVIDIA Quadro K420
    - Cœurs CUDA : 192
    - Architecture : Kepler (GK107)
    - Compute Capability : 3.0
    - Mémoire : 2 Go GDDR3 @ 1,8 GHz
    - Bande passante mémoire : 28,8 Go/s
    - TDP : 41 W
```

### A.2 Environnement Logiciel

```
Système d'exploitation : Debian GNU/Linux 10 (Buster)
Noyau : Linux 4.19.0-18-amd64

Pilotes :
- NVIDIA Driver : 384.130
- CUDA Toolkit : 9.0
- OpenCL : 1.2

Environnement Python :
- Python : 3.7.3
- NumPy : 1.16.2
- PyOpenCL : 2019.1.2
- PyCUDA : 2019.1.2

Compilateurs :
- GCC : 8.3.0
- nvcc : 9.0.176
```

