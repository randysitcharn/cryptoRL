# Serveur actuel

```
SSH_HOST=142.171.48.138
SSH_PORT=24256
SSH_USER=root
```

## Connexion rapide

```bash
ssh -p 24256 root@142.171.48.138
```

## SCP (copier donnees)

```bash
scp -P 24256 -r data/raw_historical/ root@142.171.48.138:/workspace/cryptoRL/data/
```

## Tunnel TensorBoard

```bash
ssh -p 24256 -L 6006:localhost:16006 root@142.171.48.138
# Ouvrir http://localhost:6006
```

## Reconfigurer TensorBoard (si nécessaire)

```bash
# Sur le serveur: pointer sur les logs WFO
pkill -f 'tensorboard.*16006'
nohup tensorboard --logdir /workspace/cryptoRL/logs/wfo --port 16006 --bind_all > /dev/null 2>&1 &
```
