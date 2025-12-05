#!/usr/bin/env python3
"""
Wrapper script per allenare una SWIN con MIM su ImageNet

Questo script riusa la pipeline di pretraining BEiT già presente in `MOODv1/run_beit_pretraining.py`.
Esempio d'uso:
  python train_swin_mim.py --data_path /path/to/imagenet --batch_size 256 --epochs 300

Lo script sovrascrive alcuni argomenti di default (modello, dataset) e inoltra il resto
alla funzione `main` di `run_beit_pretraining`.

COME USARE CON PIÙ GPU:
CUDA_VISIBLE_DEVICES=5,6 torchrun --nproc_per_node=2 train_swin_mim.py --config configs/swin_mim.yaml
"""
import os
import sys
from typing import Any, Dict
import argparse
import yaml
ROOT = os.path.dirname(os.path.abspath(__file__))


import run_beit_pretraining as rb


def _load_yaml_config(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"File di configurazione YAML non trovato: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def _apply_yaml_to_rb_args(yaml_cfg: Dict[str, Any], rb_args):
    # Applica tutte le chiavi del YAML come attributi su rb_args
    for k, v in yaml_cfg.items():
        # skip None values to keep defaults where appropriate
        setattr(rb_args, k, v)
    return rb_args


def main():
    # Carica file YAML di configurazione. Nessun argomento CLI: tutto proviene dal YAML.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', '--config_path', '-c', dest='config_path', type=str,
                        default=os.path.join(ROOT, 'configs', 'swin_mim.yaml'),
                        help='Percorso del file YAML di configurazione')
    args, remaining = parser.parse_known_args()
    config_path = args.config_path
    yaml_cfg = _load_yaml_config(config_path)

    original_argv = sys.argv
    sys.argv = [sys.argv[0]] + remaining  # passa solo gli argomenti rimanenti (non --config)
    rb_args = rb.get_args()
    sys.argv = original_argv  # ripristina

    # mappatura alias YAML -> arg names della pipeline
    alias_map = {
        'd_vae_path': 'discrete_vae_weight_path',
        'mask_patches': 'num_mask_patches',
        'data_path': 'data_path',
        'batch_size': 'batch_size',
        'epochs': 'epochs',
        'model': 'model',
        'd_vae_type': 'discrete_vae_type',
        'output_dir': 'output_dir',
        'device': 'device',
        'seed': 'seed',
        'num_workers': 'num_workers',
        'input_size': 'input_size',
        'second_input_size': 'second_input_size',
        'trial': 'trial',
    }

    # Applica i valori del YAML sull'oggetto rb_args
    mapped_cfg = {}
    for k, v in yaml_cfg.items():
        if k in alias_map:
            mapped_cfg[alias_map[k]] = v
        else:
            mapped_cfg[k] = v

    # assicurati che il dataset sia ImageNet a meno che non sia specificato
    if 'data_set' not in mapped_cfg:
        mapped_cfg['data_set'] = 'imagenet1k'

    rb_args = _apply_yaml_to_rb_args(mapped_cfg, rb_args)

    # crea output dir
    if hasattr(rb_args, 'output_dir') and rb_args.output_dir:
        os.makedirs(rb_args.output_dir, exist_ok=True)

    print('Avvio training SWIN con MIM usando YAML:', config_path)
    print('Model:', rb_args.model)
    print('Data path:', rb_args.data_path)
    print('Batch size:', rb_args.batch_size)
    print('Epochs:', rb_args.epochs)
    print('Output dir:', rb_args.output_dir)


    rb.main(rb_args)


if __name__ == '__main__':
    main()
