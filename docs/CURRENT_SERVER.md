# Serveur actuel

```
SSH_HOST=120.238.149.205
SSH_PORT=28764
SSH_USER=root
```

## Connexion rapide

```bash
ssh -p 28764 root@120.238.149.205
```

## SCP (copier donnees)

```bash
scp -P 28764 -r data/raw_historical/ root@120.238.149.205:/workspace/cryptoRL/data/
```

## Tunnel TensorBoard

```bash
ssh -p 28764 -L 8081:localhost:8081 root@120.238.149.205
```
