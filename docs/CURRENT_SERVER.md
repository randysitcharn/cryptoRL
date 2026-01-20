# Serveur actuel

```
SSH_HOST=158.51.110.52
SSH_PORT=20941
SSH_USER=root
```

## Connexion rapide

```bash
ssh -p 20941 root@158.51.110.52
```

## SCP (copier donnees)

```bash
scp -P 20941 -r data/raw_historical/ root@158.51.110.52:/workspace/cryptoRL/data/
```

## Tunnel TensorBoard

```bash
ssh -p 20941 -L 8081:localhost:8081 root@158.51.110.52
# Ouvrir http://localhost:8081
```

## Reconfigurer TensorBoard (si nécessaire)

```bash
# Sur le serveur: pointer sur les logs WFO
pkill -f 'tensorboard.*8081'
nohup tensorboard --logdir /workspace/cryptoRL/logs/wfo --port 8081 --bind_all > /dev/null 2>&1 &
```
