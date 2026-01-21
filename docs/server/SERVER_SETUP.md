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

### 3. Configurer SSH pour GitHub

Copier la cle SSH (depuis machine locale):
```bash
scp -P PORT ~/.ssh/id_github ~/.ssh/id_github.pub root@HOST:~/.ssh/
```

Configurer SSH sur le serveur:
```bash
ssh -p PORT root@HOST
chmod 600 ~/.ssh/id_github
chmod 700 ~/.ssh

cat > ~/.ssh/config << 'EOF'
Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_github
    IdentitiesOnly yes
EOF
chmod 600 ~/.ssh/config

# Test connexion
ssh -o StrictHostKeyChecking=no git@github.com
```

### 4. Clone du repo

```bash
cd /workspace
git clone git@github.com:randysitcharn/cryptoRL.git
cd cryptoRL
```

### 5. Installer dependances

```bash
pip install -r requirements.txt
```

### 6. Copier les donnees (depuis machine locale)

```bash
# Executer depuis la machine locale:
scp -P PORT -r data/raw_historical/ root@HOST:/workspace/cryptoRL/data/
```

Fichiers requis:
- `data/raw_historical/multi_asset_historical.csv` (principal)
- Ou fichiers individuels: `BTC_1h.csv`, `ETH_1h.csv`, etc.

### 7. Lancer le training WFO

```bash
# Training complet (13 segments)
python3 scripts/run_full_wfo.py --segments 13

# Ou un seul segment pour test
python3 scripts/run_full_wfo.py --segment 0 --timesteps 150000
```

### 8. Monitoring TensorBoard

**IMPORTANT:** vast.ai lance TensorBoard automatiquement sur `/workspace` (port 16006).
Il faut le reconfigurer pour pointer sur les logs WFO.

Sur le serveur:
```bash
# Arrêter le TensorBoard par défaut et relancer avec les bons logs
pkill -f 'tensorboard.*16006'
nohup tensorboard --logdir /workspace/cryptoRL/logs/wfo --port 16006 --bind_all > /dev/null 2>&1 &
```

Tunnel SSH (depuis local):
```bash
ssh -p PORT -L 6006:localhost:16006 root@HOST
# Ouvrir http://localhost:6006
```

### 9. Recuperer les resultats

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
