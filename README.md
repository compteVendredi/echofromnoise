# echofromnoise

Reprise (et modifications) du dépôt echo from noise : https://github.com/david-stojanovski/echo_from_noise

# Installation 

Sous linux avec conda d'installé et une carte graphique compatible CUDA :
```
conda create --name echofromnoise
conda activate echofromnoise
conda install pip python=3.11
python -m pip install -r echo_from_noise/requirements.txt
conda install opencv
conda install -c conda-forge mpi4py~=3.1.4 openmpi
```

# Préparation des données

Téléchargez le dossier sur ce lien ("download collection") : https://humanheart-project.creatis.insa-lyon.fr/database/#collection/6373703d73e9f0047faa1bc8/folder/6373727d73e9f0047faa1bca

Dézippez-le et mettez-le à la racine du dépôt.

Exécutez le script prepare_camus_public qui va permettre de déplacer les dossiers patients en fonction de leur appartenance aux données test ou training.
```
python prepare_camus_public.py
```

Puis exécutez les scripts suivants dans l'ordre qui vont extraire (mettre au format .png) puis augmenter les données :
```
python echo_from_noise/data_preparation/extract_camus_data.py
python echo_from_noise/data_preparation/augment_camus_labels.py
```


# SDM Training

```
python echo_from_noise/semantic_diffusion_model/image_train.py --datadir CAMUS_augmented --savedir ./output --batch_size_train 1  --is_train True --save_interval 50000 --lr_anneal_steps 50000 --random_flip True --deterministic_train False  --img_size 256
```
Fonctionne seulement si vous avez assez de mémoire.
Avec 8gb de VRAM et un batch_size de 1 il n'est pas possible de faire rentrer le modèle.

# Checkpoints

À défaut de pouvoir réentraîné SDM, téléchargez les checkpoints (*.pt) de SDM sur ce lien :
Placez les dans un dossier "checkpoint" à la racine du dépôt.


# SDM Inference

```
python echo_from_noise/semantic_diffusion_model/image_sample.py --datadir CAMUS_augmented/2CH_ED_augmented --resume_checkpoint checkpoint/ema_0.9999_050000_2ch_ed_256.pt --results_dir ./results_2CH_ED --num_samples 2250 --is_train False --inference_on_train True
```
