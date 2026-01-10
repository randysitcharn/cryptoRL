# Server Setup - cryptoRL (vast.ai)

Guide pour initialiser un nouveau serveur de training.

## Pre-requis

- Serveur vast.ai avec GPU NVIDIA
- Python 3.9+ (environnement `main` deja actif)
- 16GB+ RAM, 100GB+ disque

## Procedure

### 1. Mettre a jour CURRENT_SERVER.md

Editer `docs/CURRENT_SERVER.md` avec les infos du nouveau serveur:
```
SSH_HOST=<nouvelle_ip>
SSH_PORT=<nouveau_port>
SSH_USER=root
```

### 2. Connexion au serveur

```bash
ssh -p PORT root@HOST
cd /workspace
```

### 3. Clone du repo

```bash
git clone https://github.com/five-music/cryptoRL.git
cd cryptoRL
```

### 4. Installer dependances

```bash
pip install -r requirements.txt
```

### 5. Copier les donnees (depuis machine locale)

```bash
# Executer depuis la machine locale:
scp -P PORT -r data/raw_historical/ root@HOST:/workspace/cryptoRL/data/
```

Fichiers requis:
- `data/raw_historical/multi_asset_historical.csv` (principal)
- Ou fichiers individuels: `BTC_1h.csv`, `ETH_1h.csv`, etc.

### 6. Lancer le training WFO

```bash
# Training complet (13 segments)
python scripts/run_full_wfo.py --segments 13

# Ou un seul segment pour test
python scripts/run_full_wfo.py --segment 0 --timesteps 150000
```

### 7. Monitoring TensorBoard

Sur le serveur:
```bash
tensorboard --logdir logs/wfo --port 8081 &
```

Tunnel SSH (depuis local):
```bash
ssh -p PORT -L 8081:localhost:8081 root@HOST
# Ouvrir http://localhost:8081
```

### 8. Recuperer les resultats

```bash
# Depuis machine locale:
scp -P PORT root@HOST:/workspace/cryptoRL/results/wfo_results.csv ./
scp -P PORT -r root@HOST:/workspace/cryptoRL/weights/wfo/ ./weights_backup/
```

## Structure des outputs

```
/workspace/cryptoRL/
├── weights/wfo/segment_*/     # Modeles entraines
├── models/wfo/segment_*/      # HMM + Scaler
├── results/wfo_results.csv    # Metriques
└── logs/wfo/                  # TensorBoard
```

## Troubleshooting

**CUDA not found?**
- Le code utilise automatiquement CPU si pas de GPU

**Out of disk space?**
- Supprimer `logs/wfo/` entre les runs
- `rm -rf logs/wfo/*`

**Training interrompu?**
- Relancer avec `--segment N` pour reprendre au segment N
