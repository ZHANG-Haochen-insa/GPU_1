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

---
Un intermède CUDA et son implémentation PyCUDA

Nvidia a ressenti tôt la nécessité d'offrir une abstraction de programmation simple pour ses GPU. Elle a même sorti cg-toolkit dès 2002. Il faudra attendre l'été 2007 pour un langage complet, seulement limité à quelques GPU de sa gamme.

Aujourd'hui, CUDA est omniprésent dans les librairies du constructeur mais aussi dans l'immense majorité des autres développements. Cependant, son problème vient de l'adhérence au constructeur : CUDA ne sert QUE pour Nvidia. Nous verrons que CUDA a aussi d'autres inconvénient, mais à l'usage.

L'impressionnant Andreas Kloeckner a aussi développé, en plus de PyOpenCL, PyCUDA pour exploiter CUDA à travers Python avec des approches : c'est PyCUDA.

L'exemple de la page précédente ressemble fortement à celui que nous modifions depuis le début de nos travaux pratiques. Nous allons l'exploiter pour intégrer cette implémentation CUDA dans notre programme MySteps_3.py (copie de MySteps_2.py).

Les modifications du programme MySteps_3.py sont les suivantes :

    créer une fonction Python CudaAddition
    intégrer les lignes de l'exemple de PyCUDA notamment
        l'appel des librairies Python
        le noyau CUDA où la multiplication a été remplacée par l'addition
        la création du vecteur destination
        l'appel de l'addition
    entourer avec une exception le allclose
        cette précaution permet d'empêcher un plantage
    dupliquer et adapter à CUDA les éléments de contrôle de cohérence des résultats

Exercice #3.1 :

    Modifiez MySteps_3.py suivant les 3 spécifications ci-dessus
    Exécutez le programme pour des tailles de vecteurs de 32 à 32768
    Analysez dans quelles situations des problèmes de produisent
    Raccordez ces difficultés aux spécifications matérielles
    Complétez un tableau avec ces résultats
    Concluez sur l'efficacité de CUDA dans ce cas d'exploitation

Size NativeRate OpenCL Rate CUDA Rate
3229826168424
64559240519670
12812485370404138
25621913098789270
512456911411652535
10248421504531531143
20481561806286097
409628633115314923
819248393997725544
1638469413612849892
32768947854851101677

Normalement, si l'implémentation a été correcte, la partie CUDA fonctionne pour les tailles de vecteurs inférieures ou égales à 1024… Cette limitation est en fait dûe à une mauvaise utilisation de CUDA. En effet, CUDA (et dans une moindre mesure OpenCL) comporte 2 étages de parallélisation. Sous OpenCL, ces étages sont les Work Items et les Threads. Sous CUDA, ces étages sont les Blocks et les Threads. Hors, dans les deux approches OpenCL et CUDA, l'étage de parallélisation Threads est l'étage le plus fin, destiné à paralléliser des exécutions éponymes de la programmation parallèle. Mais, comme dans leurs implémentations sur processeurs, la parallélisation par Threads exige une “synchronisation”. Sous les implémentations CUDA et OpenCL, le nombre de threads maximal sollicitable dans un appel est seulement 1024 !

Cette limitation de 1024 Threads entre en contradiction avec le cadre d'utilisation présenté sur les GPU qui veut que le nombre de tâches équivalentes à exécuter est de l'ordre d'au moins plusieurs dizaines de milliers. Donc, il ne faut pas, dans un premier temps, exploiter les Threads en CUDA mais les Blocks.

Il faudra donc modifier le programme MySteps_4.py (copie de MySteps_3.py fonctionnel mais inefficace) pour exploiter les Blocks. Les modifications sont les suivantes :

    remplacer threadIdx par blockIdx dans le noyau CUDA
    remplacer dans l'appel de sum : block=(a_np.size,1,1) par block=(1,1,1)
    remplacer dans l'appel de sum : grid=(1,1) par grid=(a_np.size)

Exercice #3.2 :

    Modifiez MySteps_4.py suivant les 3 spécifications ci-dessus
    Exécutez le programme pour des tailles de vecteurs de 32768 à 268435456
    Analysez dans quelles situations des problèmes de produisent
    Raccordez ces difficultés aux spécifications matérielles
    Complétez un tableau avec ces résultats
    Concluez sur l'efficacité de CUDA dans ce cas d'exploitation

Size NativeRate OpenCL Rate CUDA Rate
327689101917449308131182
65536115011676519975071033
1310721221679586455109165674
2621441337605386793248280624
52428813979804541572131570096
1048576106982401130607921116513
209715277532772358317612246784
4194304517143454118818354384631
8388608642015438242174678813252
167772166299685243984549817001502
335544326455551965771560729982747
671088646502469008083049350612097
1342177286544202329900313675783432
26843545665653126311185899291297615

Nous constatons normalement, avec la sollicitation des blocks et plus des threads, l'implémentation CUDA fonctionne quelle que soit la taille sollicitée. L'implémentation CUDA rattrape l'OpenCL sans jamais la dépasser mais elle reste indigente en comparaison avec la méthode native, mais nous avons déjà vu pourquoi : problème de complexité arithmétique.

Nous allons donc, comme pour OpenCL, augmenter l'intensité arithmétique du traitement en rajoutant l'implémentation CUDA de notre fonction MySillyFunction ajoutée à chacun des termes des vecteurs avant leur addition.

Pour il convient de modifier le code MySteps_5.py (copie de MySteps_4.py) de la manière suivante :

    copier l'implémentation PyCUDA CUDAAddition en CUDASillyAddition
        cette nouvelle fonction Python sera à modifier pour la suite
    rajouter la fonction interne MySillyFunction dans le noyau CUDA
        une fonction interne doit être préfixée par device
    rajouter la fonction sillysum appelée par Python dans le noyau CUDA
    rajouter la synthèse de la fonction sillysum comparable à sum
    modifier l'appel de la fonction PyCUDA de sum en sillysum
    intrumenter temporellement chaque ligne de CUDASillyAddition
    modifier les appels de fonction Addition en SillyAddition
        pour les 3 implémentations Native, OpenCL et CUDA

Exercice #3.3 :

    Modifiez MySteps_5.py suivant les 7 spécifications ci-dessus
    Exécutez le programme pour des tailles de vecteurs de 32768 à 268435456
    Complétez un tableau avec ces résultats
    Concluez sur l'efficacité de CUDA dans ce cas d'exploitation

Size NativeRate OpenCL Rate CUDA Rate OpenCL ratio CUDA ratio
327681220822104351292760.0854760.023981
655361220648209305692710.1714700.056749
13107212304763931871402550.3195410.113984
26214412486958841812980470.7080840.238687
524288144790517907265742881.2367700.396634
10485761444680340192211182882.3547930.774073
20971521484030698843020565604.7090891.385794
419430415255601320846736060818.6581102.363775
8388608147851422047721510622014.9120813.453616
16777216148411937736167722871725.4266454.870713
33554432148458154005921929168136.3778886.258790
671088641484264752647941055240150.7084957.109518
1342177281486942852220661135268757.3136457.634923
26843545614856321025639441214932869.0372478.177885

Les gains sont substantiels en CUDA mais restent quand même bien en dessous de OpenCL. Pour augmenter l'efficacité de CUDA, il conviendra d'augmenter la complexité arithmétique de manière très substantielle. Par exemple, en multipliant par 16 cette complexité (en appelant par exemple 16 fois successivement cette fonction MySillyFunction), le NativeRate se divise par 16 mais le OpenCLRate ne se divise que par 2. L'implémentation CUDA, quand à elle, augmente de 60% !

Pour conclure sue ce petit intermède CUDA se trouvent les programmes MySteps_5b.py et MySteps_5c.py dérivés de MySteps_5.py :

    MySteps_5b.py : intègre une utilisation hybride des Blocks et des Threads
    MySteps_5c.py : augmente la complexité arithmétique d'un facteur 16