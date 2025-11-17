# INSA 2025 : le GPU par la pratique

Cette session de travaux pratiques se compose de séances de 4h. En préparation de cette dernière session pour l'INSA de Lyon, il existe deux cours présentés par Emmanuel Quémener les 17 octobre et 10 novembre 2025.

- Cours sur les CPU et les architectures parallélisées
- Cours sur les GPU, la technologie disruptive du 21e siècle

Feuilleter ces cours permet de se familiariser avec certains concepts lesquels seront développés durant les séances.

## CQQCOQP : Comment ? Qui ? Quand ? Combien ? Où ? Quoi ? Pourquoi ?

- **Pourquoi ?** Faire un tour d'horizon des GPUs et appréhender des méthodes d'investigation
- **Quoi ?** Programmer, tester et comparer les GPU sur des exemples simples
- **Quand ?** A partir du lundi 10 novembre 2025
- **Combien ?** Mesurer la performance que les GPUs offrent en comparaison des autres machines
- **Où ?** Sur des stations de travail, des noeuds de cluster, des portables (bien configurés), dans des terminaux
- **Qui ?** Pour les édudiants, enseignants, chercheurs, personnels techniques curieux
- **Comment ?** En appliquant quelques commandes simples, généralement dans des terminaux.

## But de la session

C'est de prendre en main les GPU dans les machines, d'appréhender la programmation en OpenCL et CUDA, de comparer les performances avec des CPU classiques par l'intermédiaire de quelques exemples simples et des codes de production.

## Déroulement des sessions pratiques

Le programme est volontairement touffu mais les explications données et les corrigés devraient permettre de poursuivre l'apprentissage par la pratique hors des deux séances de travaux pratiques.

1. Prise en main de l'environnement à distance du Centre Blaise Pascal à l'ENS-Lyon
2. Découverte du matériel, autant CPU que GPU
3. Exploration progressive en OpenCL avec l'exemple de base de la documentation Python/OpenCL
4. Un intermède Python/CUDA pour tester l'autre implémentation sur GPU
5. La réalisation et le portage d'une transformée de Fourier discrète
6. Choix du périphérique en Python et sa programmation
7. Utilisation des librairies externes : exemple avec xGEMM
8. Intégration de "codes métier" : TensorFlow, GENESIS et Gromacs
9. Expoitation de codes Matrices pour la métrologie

De manière à disposer d'une trace de votre travail et de pouvoir l'évaluer, il est demandé de rédiger un "livre de bord" sur la base des questions posées. Faites des copies d'écran et intégrez-les dans votre document, ainsi que les codes que vous aurez produits.

## Démarrage de la session

### Prérequis en matériel, logiciel et humain

De manière à proposer un environnement pleinement fonctionnel, le Centre Blaise Pascal fournit le matériel, les logiciels et un OS correctement intégré. Les personnes qui veulent réaliser cette session sur leur laptop doivent disposer d'un "vrai" système d'exploitation de type Unix, équipé de tout l'environnement adéquat.

### Prérequis pour le matériel

- Si vous n'utilisez PAS le CBP, une machine relativement récente avec une GPU intégrée avec circuit Nvidia
- Si vous utilisez le CBP, un laptop disposant d'un écran assez confortable pour afficher une fenêtre de 1024×768, une connexion réseau la plus stable possible et la capacité d'y installer un logiciel adapté.

### Prérequis pour le logiciel

- Si vous n'utilisez pas le CBP, un OS GNU/Linux correctement configuré pour la GPU embarquée avec tous les composants Nvidia, OpenCL, PyOpenCL, PyCUDA.
  - Un `apt install time pciutils clinfo nvidia-opencl-icd nvidia-smi pocl-opencl-icd python3-pyopencl python-pyopencl-doc python-pycuda-doc python3-pycuda` devrait être suffisant comme prérequis pour une machine avec un circuit Nvidia pas trop ancien
  - Pour les implémentations OpenCL sur CPU sur Debian ou Ubuntu, essayez d'installer celle d'Intel et celle d'AMD. A votre environnement s'ajoute un navigateur pour voir cette page ainsi qu'un traitement de texte pour rédiger le compte-rendu de ces séances.
- Si vous utilisez le CBP, il faut avoir installé le logiciel x2goclient suivant les recommandations de la documentation du CBP. Il est recommandé d'exploiter le traitement de texte et le navigateur dans la session distante.

### Prérequis pour l'humain

- Une allergie à la commande en ligne peut dramatiquement réduire la portée de cette session pratique
- Une pratique des scripts shell sera un avantage, sinon vous avez cette session pour parfaire vos connaissances.

## Investiguer le matériel GPU

### Qu'y a-t-il dans ma machine ?

Le matériel en Informatique Scientifique est défini par l'architecture de Von Neumann:

- CPU (Unité Centrale de Traitement) avec CU (Unité de Contrôle) et ALU (Unité Arithmétique & Logique)
- MU (Unité de Mémoire)
- Input and Output Devices : Périphériques d'Entrée et Sortie

Les GPU sont généralement considérés comme des périphériques d'Entrée/Sortie. Comme la plupart des périphériques installés dans les machines, elles exploitent un bus d'interconnexion PCI ou PCI Express.

Pour récupérer la liste des périphériques PCI, utilisez la commande `lspci -nn`. A l'intérieur d'une longue liste apparaissent quelques périphériques VGA ou 3D. Ce sont les périphériques GPU ou GPGPU.

Voici une sortie de la commande `lspci -nn | egrep '(VGA|3D)'` :

```
3b:00.0 VGA compatible controller [0300]: NVIDIA Corporation GP102 [GeForce GTX 1080 Ti] [10de:1b06] (rev a1)
a1:00.0 VGA compatible controller [0300]: NVIDIA Corporation GK107GL [Quadro K420] [10de:0ff3] (rev a1)
```

### Exercice #1.1: récuperez la liste des périphériques (GP)GPU

- Combien de périphériques VGA sont listés ?
- Combien de périphériques 3D sont listés ?
- Récupérez le modèle du circuit de GPU, dans son nom étendu.
- Récupérez sur le web les informations suivantes pour chaque GPU :
  - le nombre d'unités de calcul (les "cuda cores" ou les "stream processors")
  - la fréquence de base des coeurs de calcul
  - la fréquence de la mémoire

### Informations système avec dmesg

Dans les systèmes Posix (Unix dans le langage courant), tout est fichier. Les informations sur les circuits Nvidia et leur découverte par le système d'exploitation peuvent être récupérées avec un grep dans la commande dmesg.

### Exercice #1.2 : récupérez les informations de votre machine avec dmesg | grep -i nvidia

- Quelle est la version de pilote chargée par le noyau ?
- Que représente, s'il existe, le périphérique `input: HDA NVidia` ?
- Est-ce un périphérique graphique ?

### Modules kernel avec lsmod

Le lsmod offre la liste des modules chargés par le noyau. Ces modules sont de petits programmes dédiés au support d'une fontion très spécifique du noyau, le moteur du système d'exploitation. Le support d'un périphérique nécessite souvent plusieurs modules.

### Exercice #1.3 : récupérez les informations de l'hôte par la commande lsmod | grep nvidia

- Les informations sont-elles identiques à celles ci-dessus ? Caractère par caractère ?

### Périphériques dans /dev

Le périphérique apparaît également dans le dossier `/dev` (pour device), le dossier parent pour tous les périphériques.

Un `ls -l /dev/nvidia*` offre ce genre d'informations :

```
crw-rw-rw- 1 root root 195,   0 Jun 30 18:17 /dev/nvidia0
crw-rw-rw- 1 root root 195, 255 Jun 30 18:17 /dev/nvidiactl
crw-rw-rw- 1 root root 195, 254 Jun 30 18:17 /dev/nvidia-modeset
crw-rw-rw- 1 root root 243,   0 Jul  4 19:17 /dev/nvidia-uvm
crw-rw-rw- 1 root root 243,   1 Jul  4 19:17 /dev/nvidia-uvm-tools
```

### Exercice #1.4 : récupérez les informations de votre machine avec ls -l /dev/* | grep -i nvidia

- Combien de `/dev/nvidia<number>` avez-vous ?
- Cette information est-elle cohérente avec les 3 précédentes ?

### nvidia-smi : System Management Interface

Nvidia présente des informations sur l'usage instantané de ses circuits avec la commande `nvidia-smi`. Cette commande peut aussi être exploitée pour régler certains paramètres de la GPU.

### Exercice #1.5 : récupérez les informations avec la commande nvidia-smi

- Identifiez les caractéristiques et comparer les éléments
- Combien de processus sont-ils listés ?

### OpenCL avec clinfo

Sur les stations du CBP, la majorité des implémentations de OpenCL sont disponibles, autant sur CPU que sur GPU.

La commande `clinfo` récupère des informations liées à tous les périphériques OpenCL disponibles. Pour récupérer une sortie compacte, utilisez `clinfo -l`.

Tous les périphériques OpenCL sont présentés suivant une hiérarchie plateforme/périphérique (Platform/Device).

Voici une sortie de `clinfo -l` pour une des stations de travail :

```
Platform #0: AMD Accelerated Parallel Processing
 `-- Device #0: Intel(R) Xeon(R) CPU E5-2637 v4 @ 3.50GHz
Platform #1: Portable Computing Language
 `-- Device #0: pthread-Intel(R) Xeon(R) CPU E5-2637 v4 @ 3.50GHz
Platform #2: NVIDIA CUDA
 +-- Device #0: GeForce GTX 1080 Ti
 `-- Device #1: Quadro K420
Platform #3: Intel(R) OpenCL
 `-- Device #0: Intel(R) Xeon(R) CPU E5-2637 v4 @ 3.50GHz
```

### Exercice #1.6 : récupérez les informations avec la commande clinfo -l

- Identifiez et comparez votre sortie avec la liste ci-dessus
- De combien de périphériques graphiques disposez-vous ?

### Exercice #1.7 : récupérez les informations détaillées avec clinfo

- Comparez les informations entre les implémentations CPU. Pourquoi ces différences ?
- Comparez le nombre d'unités de traitement des CPU avec celles du Web : Ark d'Intel
- Comparez le nombre d'unités de traitement des GPU avec celles du Web
- Comparez les fréquences identifiées avec celles trouvées sur le Web.
- Retrouvez-vous une cohérence entre le nombre de Compute Units et le nombre de cuda cores ?
- Combien de cuda cores contient chaque Compute Unit ?

### CUDA_VISIBLE_DEVICES

Il est aussi possible de choisir quelle GPU Nvidia exploiter avec la variable d'environnement `CUDA_VISIBLE_DEVICES`.

### Exercice #1.8 : récupérez les informations avec clinfo -l préfixée de CUDA_VISIBLE_DEVICES

- Mettez `CUDA_VISIBLE_DEVICES=0 clinfo -l` et observez la sortie
- Mettez `CUDA_VISIBLE_DEVICES=1 clinfo -l` et observez la sortie
- Mettez `CUDA_VISIBLE_DEVICES=0,1 clinfo -l` et observez la sortie
- Mettez `CUDA_VISIBLE_DEVICES='' clinfo -l` et observez la sortie
- Avez-vous constaté la sélection des différents périphériques ?

### Exercice #1.9 : lancez les commandes de monitoring

- Ouvrez un terminal, tapez `dstat -cim` et observez la sortie
- Détaillez à quoi servent les paramètres de sortie c, i et m
- Ouvrez un terminal, tapez `nvidia-smi dmon` et observez la sortie
- Détaillez à quoi sert l'option dmon
- Arrêtez l'exécution de la précédente avec `<Ctrl><C>`
- Relancez la commande précédente avec `-i 0` ou `-i 1`
- Détaillez à quoi sert l'option `-i` suivie d'un entier

## Récupération des sources

La (presque) totalité des outils exploités par le CBP pour comparer les CPU et les GPU se trouve dans le projet bench4xpu.

La récupération des sources est libre et se réalise par l'outil git :

```bash
git clone https://github.com/numa65536/bench4xpu
```

## Première exploration de l'association Python et OpenCL

Basons-nous pour ce premier programme sur celui présenté sur la documentation officielle de PyOpenCL. Il se propose d'ajouter deux vecteurs `a_np` et `b_np` en un vecteur `res_np`.

```python
#!/usr/bin/env python

import numpy as np
import pyopencl as cl

a_np = np.random.rand(50000).astype(np.float32)
b_np = np.random.rand(50000).astype(np.float32)

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags
a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)

prg = cl.Program(ctx, """
__kernel void sum(
    __global const float *a_g, __global const float *b_g, __global float *res_g)
{
  int gid = get_global_id(0);
  res_g[gid] = a_g[gid] + b_g[gid];
}
""").build()

res_g = cl.Buffer(ctx, mf.WRITE_ONLY, a_np.nbytes)
knl = prg.sum  # Use this Kernel object for repeated calls
knl(queue, a_np.shape, None, a_g, b_g, res_g)

res_np = np.empty_like(a_np)
cl.enqueue_copy(queue, res_np, res_g)

# Check on CPU with Numpy:
print(res_np - (a_np + b_np))
print(np.linalg.norm(res_np - (a_np + b_np)))
assert np.allclose(res_np, a_np + b_np)
```

### Exercice #2.1 : première exécution

- Exploitez un éditeur (par exemple gedit)
- Copiez/Coller le contenu du programme source précédent
- Enregistrez le source avec le nom MySteps.py
- Lancez le avec et jugez de l'exécution : `python MySteps.py`
- Lancez le avec et jugez de l'exécution : `python3 MySteps.py`
- Changez les droits d'exécution de MySteps.py
- Lancez le directement avec `./MySteps.py`
- En cas d'échec de lancement, modifiez MySteps.py
- Préfixez le lancement avec TOUTES les combinaisons de PYOPENCL_CTX
- Redirigez les sorties standards dans des fichiers MySteps_XY.out

### Exercice #2.2 : modifier sans changer la sortie

- Modifiez MySteps_0.py suivant les spécifications
- Exécutez le programme pour plusieurs périphériques
- Sauvez pour chaque exécution la sortie standard
- Comparez avec la commande diff les sorties des exercices 2.1 et 2.2

### Exercice #2.3 : instrumentation minimale du code

- Modifiez MySteps_1.py suivant les spécifications
- Exécutez le programme pour des tailles de vecteurs de 2^15 à 2^30
- Analysez dans quelles situations des problèmes de produisent
- Raccordez ces difficultés aux spécifications matérielles
- Complétez un tableau avec ces résultats
- Concluez sur l'efficacité de OpenCL dans ce cas d'exploitation

## Conclusion

Des résultats, il est possible de voir que, sur une opération aussi simple qu'une addition, dans aucune situation l'implémentation OpenCL n'apporte le moindre intérêt. L'exécution native en Python est toujours plus rapide d'un facteur 4 sur CPU et d'un facteur 6 sur GPU.

Pour qu'une exécution OpenCL soit efficace, il faudra veiller à :
1. Un nombre d'éléments conséquent (plusieurs milliers à plusieurs millions)
2. Un nombre d'opérations élémentaires d'une densité arithmétique "suffisante" (supérieures à la dizaine)

---

*Source: Centre Blaise Pascal, ENS de Lyon*
*Emmanuel Quémener 2022/07/20*
