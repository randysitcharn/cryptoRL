# Serveur actuel

```
SSH_HOST=158.51.110.52
SSH_PORT=20941
SSH_USER=root
```

## IMPORTANT: Convention des logs

**Tous les fichiers de logs doivent être créés dans le dossier `logs/` du projet.**

Sur le serveur : `/workspace/cryptoRL/logs/`

Exemples :
- Logs d'entraînement : `logs/wfo/`, `logs/training/`
- Logs TensorBoard : `logs/tensorboard/`
- Logs de scripts : `logs/script_name.log`

**Ne jamais créer de fichiers `.log` à la racine du projet ou dans d'autres dossiers.**

**Raison:** TensorBoard est configuré pour pointer sur le dossier `logs/`. Si les logs sont créés ailleurs, ils ne seront pas visibles dans TensorBoard.

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

## IMPORTANT: Rate Limiting & SSH Multiplexing (WSL)

vast.ai bloque les connexions SSH trop fréquentes. **Solution : SSH Multiplexing via WSL.**

### SSH Multiplexing via WSL (recommandé)

Le multiplexing permet de réutiliser une connexion SSH existante → **pas de rate limiting**.

**Configuration (déjà faite dans WSL ~/.ssh/config) :**
```
Host vast
    HostName 158.51.110.52
    Port 20941
    User root
    IdentityFile ~/.ssh/id_ed25519
    ControlMaster auto
    ControlPath ~/.ssh/controlmasters/%r@%h:%p
    ControlPersist 10m
```

**Usage simplifié :**
```powershell
wsl bash -c "ssh vast 'commande'"
```

**Exemples :**
```powershell
# Vérifier un processus
wsl bash -c "ssh vast 'pgrep -fa python'"

# Voir les logs
wsl bash -c "ssh vast 'tail -30 /workspace/cryptoRL/logs/training.log'"

# Lancer un script
wsl bash -c "ssh vast 'cd /workspace/cryptoRL && python3 script.py'"
```

**Avantages :**
- Pas de rate limiting (connexion réutilisée)
- Commandes quasi instantanées
- Tunnel reste ouvert 10 minutes après la dernière commande

### Session tmux (alternative)

Pour du travail interactif prolongé :
```powershell
ssh -p 20941 root@158.51.110.52 -t "tmux attach -t ssh_tmux"
```

**Raccourcis tmux :**
| Raccourci | Action |
|-----------|--------|
| `Ctrl+B, D` | Détacher (quitter sans fermer) |
| `Ctrl+B, C` | Nouvelle fenêtre |
| `Ctrl+B, N` | Fenêtre suivante |

### Commandes en background (exit code 255)

```bash
# ❌ Ne fonctionne pas
ssh host "nohup cmd > /dev/null 2>&1 &"

# ✅ Fonctionne (détacher stdin)
ssh host "nohup cmd </dev/null >logs/fichier.log 2>&1 &"
```

## Connexion rapide

```powershell
# Session tmux (recommandé)
ssh -p 20941 root@158.51.110.52 -t "tmux attach -t ssh_tmux"

# Shell simple
ssh -p 20941 root@158.51.110.52
```

## SCP (copier données)

```powershell
# Copier vers le serveur
scp -P 20941 -r data/raw_historical/ root@158.51.110.52:/workspace/cryptoRL/data/

# Copier depuis le serveur
scp -P 20941 root@158.51.110.52:/workspace/cryptoRL/results/wfo_results.csv ./
```

## Tunnel TensorBoard

```powershell
ssh -p 20941 -L 8081:localhost:8081 root@158.51.110.52
# Ouvrir http://localhost:8081 dans le navigateur
```

## Reconfigurer TensorBoard

Dans la session tmux ou via commande ponctuelle :
```bash
pkill -f 'tensorboard.*8081'
nohup tensorboard --logdir /workspace/cryptoRL/logs/wfo --port 8081 --bind_all </dev/null >logs/tensorboard.log 2>&1 &
```
