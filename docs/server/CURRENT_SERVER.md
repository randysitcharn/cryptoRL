# Serveur actuel

```
SSH_HOST=158.51.110.52
SSH_PORT=20941
SSH_USER=root
```

## IMPORTANT: Vérifier l'environnement local AVANT d'exécuter des commandes

**Toujours vérifier le système d'exploitation et le shell avant d'exécuter des commandes SSH.**

L'environnement local peut être :
- **Windows + PowerShell** (actuellement le cas)
- **Linux/macOS + Bash**

### Si Windows + PowerShell

1. **NE PAS utiliser** la syntaxe bash :
   ```bash
   # ❌ NE FONCTIONNE PAS sur Windows/PowerShell
   eval $(ssh-agent -s) && ssh-add ~/.ssh/id_ed25519 && ssh ...
   ```

2. **Utiliser directement** la commande ssh (la clé est déjà configurée dans l'agent Windows) :
   ```powershell
   # ✅ FONCTIONNE sur Windows/PowerShell
   ssh -p 20941 root@158.51.110.52 "commande"
   ```

3. **Chemin des clés SSH Windows** : `~/.ssh/` se traduit en `C:\Users\<user>\.ssh\`

4. **Guillemets** : Utiliser des guillemets doubles pour les commandes distantes
   ```powershell
   ssh -p 20941 root@158.51.110.52 "cd /workspace/cryptoRL && python3 script.py"
   ```

5. **Pas de `&&` entre commandes locales PowerShell** - utiliser `;` ou des commandes séparées

### Si Linux/macOS + Bash

Si l'agent SSH n'est pas démarré, utiliser :
```bash
eval $(ssh-agent -s) && ssh-add ~/.ssh/id_ed25519 && ssh -p 20941 root@158.51.110.52 "commande"
```

Sinon, directement :
```bash
ssh -p 20941 root@158.51.110.52 "commande"
```

## IMPORTANT: Rate Limiting

vast.ai bloque les connexions SSH trop fréquentes. Pour éviter d'être bloqué:
- **Attendre 20 secondes** entre chaque commande SSH
- Ne pas faire de requêtes SSH en parallèle
- Si "Connection refused", attendre 60 secondes avant de réessayer

### Commandes en background (exit code 255)

**Problème:** Les commandes avec `nohup ... &` peuvent retourner exit code 255 car SSH attend que le processus se termine.

**Solution:** Détacher complètement stdin avec `</dev/null` :

```bash
# ❌ Ne fonctionne pas (exit 255)
ssh host "nohup cmd > /dev/null 2>&1 &"

# ✅ Fonctionne
ssh host "nohup cmd </dev/null >fichier.log 2>&1 &"
```

### Exemple : Lancer TensorBoard (depuis PowerShell Windows)

```powershell
ssh -p 20941 root@158.51.110.52 "cd /workspace/cryptoRL && nohup tensorboard --logdir logs/wfo --port 8081 --bind_all </dev/null >tensorboard.log 2>&1 & sleep 3; pgrep -f 'tensorboard.*8081' && echo 'OK'"
```

## Connexion rapide (PowerShell)

```powershell
ssh -p 20941 root@158.51.110.52
```

## SCP (copier données) - PowerShell

```powershell
# Copier vers le serveur
scp -P 20941 -r data/raw_historical/ root@158.51.110.52:/workspace/cryptoRL/data/

# Copier depuis le serveur
scp -P 20941 root@158.51.110.52:/workspace/cryptoRL/results/wfo_results.csv ./
```

## Tunnel TensorBoard (PowerShell)

```powershell
ssh -p 20941 -L 8081:localhost:8081 root@158.51.110.52
# Ouvrir http://localhost:8081 dans le navigateur
```

## Reconfigurer TensorBoard (si nécessaire)

Exécuter sur le serveur (via connexion interactive ou commande distante) :

```powershell
# Depuis PowerShell local :
ssh -p 20941 root@158.51.110.52 "pkill -f 'tensorboard.*8081'; nohup tensorboard --logdir /workspace/cryptoRL/logs/wfo --port 8081 --bind_all </dev/null >tensorboard.log 2>&1 &"
```
