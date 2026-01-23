# Script d'initialisation du serveur cryptoRL (PowerShell)
# Usage: .\scripts\init_server.ps1 [-SkipSSHSetup] [-SkipDeps] [-SkipClone] [-SkipData] [-SkipTensorBoard] [-OnlyTensorBoard]

param(
    [switch]$SkipSSHSetup,
    [switch]$SkipDeps,
    [switch]$SkipClone,
    [switch]$SkipData,
    [switch]$SkipTensorBoard,
    [switch]$OnlyTensorBoard,
    [string]$ServerHost = "212.93.107.107",
    [string]$ServerPort = "40075",
    [string]$ServerUser = "root"
)

# Si OnlyTensorBoard, skip tout sauf TensorBoard
if ($OnlyTensorBoard) {
    $SkipSSHSetup = $true
    $SkipDeps = $true
    $SkipClone = $true
    $SkipData = $true
    $SkipTensorBoard = $false
}

$SSH_HOST = $ServerHost
$SSH_PORT = $ServerPort
$SSH_USER = $ServerUser
$WORKSPACE = "/workspace"
$KNOWN_HOSTS = "$env:USERPROFILE\.ssh\known_hosts"
$SSH_CONFIG = "$env:USERPROFILE\.ssh\config"
$LOCAL_DATA = "data\raw_historical"
$REMOTE_DATA = "/workspace/cryptoRL/data/raw_historical"
$TB_PORT = 8081

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "Initialisation du serveur cryptoRL" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "Host: $SSH_HOST"
Write-Host "Port: $SSH_PORT"
Write-Host "User: $SSH_USER"
Write-Host ""

# Fonction pour executer une commande sur le serveur
function Invoke-SSHCommand {
    param([string]$Command)
    ssh -p $SSH_PORT "$SSH_USER@$SSH_HOST" $Command
}

# ============================================================================
# ETAPE 0: Configuration SSH (known_hosts + raccourci vast)
# ============================================================================
if (-not $SkipSSHSetup) {
    Write-Host "[0/10] Configuration SSH..." -ForegroundColor Yellow

    # 0a. Nettoyer les anciens serveurs Vast du known_hosts
    Write-Host "  -> Nettoyage des anciens serveurs Vast..." -ForegroundColor Gray
    if (Test-Path $KNOWN_HOSTS) {
        # Supprimer les lignes contenant des IPs Vast typiques (212.93.x.x, 194.68.x.x, etc.)
        $content = Get-Content $KNOWN_HOSTS | Where-Object {
            $_ -notmatch "^\[?212\.93\." -and
            $_ -notmatch "^\[?194\.68\." -and
            $_ -notmatch "^\[?185\.203\." -and
            $_ -notmatch "^\[?64\.71\." -and
            $_ -notmatch "^\[?216\.73\." -and
            $_ -notmatch "vast"
        }
        $content | Set-Content $KNOWN_HOSTS
        Write-Host "  [OK] Anciens serveurs Vast supprimes" -ForegroundColor Green
    }

    # 0b. Scanner et ajouter la cle du nouveau serveur
    Write-Host "  -> Ajout de la cle SSH du serveur..." -ForegroundColor Gray
    $keyScan = ssh-keyscan -p $SSH_PORT $SSH_HOST 2>$null
    if ($keyScan) {
        $keyScan | Add-Content $KNOWN_HOSTS
        Write-Host "  [OK] Cle SSH ajoutee au known_hosts" -ForegroundColor Green
    } else {
        Write-Host "  [ERREUR] Impossible de scanner la cle SSH" -ForegroundColor Red
        exit 1
    }

    # 0c. Creer/Mettre a jour le raccourci "vast" dans ~/.ssh/config
    Write-Host "  -> Configuration du raccourci 'vast'..." -ForegroundColor Gray

    # S'assurer que le fichier config existe
    if (-not (Test-Path $SSH_CONFIG)) {
        New-Item -Path $SSH_CONFIG -ItemType File -Force | Out-Null
    }

    # Lire le contenu actuel
    $configContent = Get-Content $SSH_CONFIG -Raw -ErrorAction SilentlyContinue
    if (-not $configContent) { $configContent = "" }

    # Supprimer l'ancien bloc "Host vast" s'il existe
    $configContent = $configContent -replace "(?ms)Host vast\r?\n.*?(?=\r?\nHost |\z)", ""
    $configContent = $configContent.Trim()

    # Ajouter le nouveau bloc
    $vastConfig = @"

Host vast
    HostName $SSH_HOST
    Port $SSH_PORT
    User $SSH_USER
    StrictHostKeyChecking no
    UserKnownHostsFile ~/.ssh/known_hosts
"@

    $configContent = $configContent + "`n" + $vastConfig
    $configContent | Set-Content $SSH_CONFIG

    Write-Host "  [OK] Raccourci 'vast' configure (ssh vast)" -ForegroundColor Green
    Write-Host ""
} else {
    Write-Host "[0/10] Configuration SSH... SKIP" -ForegroundColor DarkGray
}

# ============================================================================
# ETAPE 1: Test de connexion
# ============================================================================
Write-Host "[1/10] Test de connexion..." -ForegroundColor Yellow
$result = Invoke-SSHCommand "echo Connexion OK"
if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] Connexion reussie" -ForegroundColor Green
} else {
    Write-Host "[ERREUR] Echec de connexion" -ForegroundColor Red
    exit 1
}

# ============================================================================
# ETAPE 2: Verifier Python
# ============================================================================
Write-Host ""
Write-Host "[2/10] Verification de Python..." -ForegroundColor Yellow
$pythonVersion = Invoke-SSHCommand "python3 --version"
Write-Host "Python: $pythonVersion"

# ============================================================================
# ETAPE 3: Verifier GPU
# ============================================================================
Write-Host ""
Write-Host "[3/10] Verification GPU..." -ForegroundColor Yellow
$gpuInfo = Invoke-SSHCommand "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"
Write-Host "GPU: $gpuInfo"

# ============================================================================
# ETAPE 4: Preparer le workspace
# ============================================================================
Write-Host ""
Write-Host "[4/10] Preparation du workspace..." -ForegroundColor Yellow
Invoke-SSHCommand "mkdir -p $WORKSPACE; cd $WORKSPACE; pwd"
Write-Host "[OK] Workspace pret" -ForegroundColor Green

# ============================================================================
# ETAPE 5-6: Repository
# ============================================================================
if (-not $SkipClone) {
    Write-Host ""
    Write-Host "[5/10] Verification du repository..." -ForegroundColor Yellow
    $repoExists = Invoke-SSHCommand 'test -d /workspace/cryptoRL && echo exists || echo not_exists'
    $skipCloneStep = $false

    if ($repoExists -match "exists") {
        Write-Host "[ATTENTION] Le repository existe deja" -ForegroundColor Yellow
        $response = Read-Host "Voulez-vous le supprimer et le cloner a nouveau? (y/N)"
        if ($response -eq "y" -or $response -eq "Y") {
            Invoke-SSHCommand "rm -rf $WORKSPACE/cryptoRL"
            Write-Host "[OK] Ancien repository supprime" -ForegroundColor Green
        } else {
            Write-Host "[INFO] Utilisation du repository existant" -ForegroundColor Cyan
            Write-Host "[6/10] Mise a jour du repository..." -ForegroundColor Yellow
            Invoke-SSHCommand "cd $WORKSPACE/cryptoRL; git pull"
            Write-Host "[OK] Repository mis a jour" -ForegroundColor Green
            $skipCloneStep = $true
        }
    }

    # Clone du repository (si necessaire)
    if (-not $skipCloneStep) {
        Write-Host ""
        Write-Host "[6/10] Clone du repository GitHub..." -ForegroundColor Yellow
        Write-Host "[ATTENTION] Assurez-vous que SSH est configure pour GitHub sur le serveur" -ForegroundColor Yellow
        Invoke-SSHCommand "cd $WORKSPACE; git clone git@github.com:randysitcharn/cryptoRL.git"
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[OK] Repository clone" -ForegroundColor Green
        } else {
            Write-Host "[ERREUR] Erreur lors du clone. Verifiez la configuration SSH GitHub" -ForegroundColor Red
        }
    }
} else {
    Write-Host ""
    Write-Host "[5/10] Verification du repository... SKIP" -ForegroundColor DarkGray
    Write-Host "[6/10] Clone du repository... SKIP" -ForegroundColor DarkGray
}

# ============================================================================
# ETAPE 7: Installation des dependances
# ============================================================================
Write-Host ""
if (-not $SkipDeps) {
    Write-Host "[7/10] Installation des dependances..." -ForegroundColor Yellow
    Write-Host "[INFO] Cela peut prendre plusieurs minutes..." -ForegroundColor Cyan
    Invoke-SSHCommand "cd $WORKSPACE/cryptoRL; pip install -r requirements.txt"
    Write-Host "[OK] Dependances installees" -ForegroundColor Green
} else {
    Write-Host "[7/10] Installation des dependances... SKIP" -ForegroundColor DarkGray
}

# ============================================================================
# ETAPE 8: Verification PyTorch/CUDA
# ============================================================================
Write-Host ""
Write-Host "[8/10] Verification PyTorch/CUDA..." -ForegroundColor Yellow
$torchCheck = Invoke-SSHCommand "python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())'"
Write-Host "  $torchCheck"

# ============================================================================
# ETAPE 9: Copie des donnees historiques
# ============================================================================
Write-Host ""
if (-not $SkipData) {
    Write-Host "[9/10] Synchronisation des donnees historiques..." -ForegroundColor Yellow

    # Verifier si les donnees locales existent
    if (-not (Test-Path $LOCAL_DATA)) {
        Write-Host "  [ERREUR] Dossier local '$LOCAL_DATA' introuvable" -ForegroundColor Red
    } else {
        # Compter les fichiers locaux
        $localFiles = (Get-ChildItem -Path $LOCAL_DATA -File -Recurse).Count
        Write-Host "  -> Fichiers locaux: $localFiles" -ForegroundColor Gray

        # Verifier les donnees sur le serveur
        $remoteCheck = Invoke-SSHCommand "ls $REMOTE_DATA/*.parquet 2>/dev/null | wc -l"
        $remoteFiles = [int]$remoteCheck.Trim()
        Write-Host "  -> Fichiers serveur: $remoteFiles" -ForegroundColor Gray

        if ($remoteFiles -ge $localFiles -and $localFiles -gt 0) {
            Write-Host "  [OK] Donnees deja presentes sur le serveur ($remoteFiles fichiers)" -ForegroundColor Green
        } else {
            Write-Host "  -> Upload des donnees en cours..." -ForegroundColor Cyan
            # Creer le dossier distant si necessaire
            Invoke-SSHCommand "mkdir -p $REMOTE_DATA"
            # Copier les donnees
            # Copier tout le contenu du dossier
            scp -P $SSH_PORT -r "${LOCAL_DATA}/" "${SSH_USER}@${SSH_HOST}:${REMOTE_DATA}/"
            if ($LASTEXITCODE -eq 0) {
                Write-Host "  [OK] Donnees uploadees" -ForegroundColor Green
            } else {
                Write-Host "  [ERREUR] Echec de l'upload" -ForegroundColor Red
            }
        }
    }
} else {
    Write-Host "[9/10] Synchronisation des donnees... SKIP" -ForegroundColor DarkGray
}

# ============================================================================
# ETAPE 10: Configuration TensorBoard
# ============================================================================
Write-Host ""
if (-not $SkipTensorBoard) {
    Write-Host "[10/10] Configuration TensorBoard (port $TB_PORT)..." -ForegroundColor Yellow

    # Arreter toute instance TensorBoard existante
    Write-Host "  -> Arret des instances TensorBoard existantes..." -ForegroundColor Gray
    Invoke-SSHCommand "pkill -f tensorboard 2>/dev/null; sleep 1"

    # Creer le dossier logs si necessaire
    Invoke-SSHCommand "mkdir -p $WORKSPACE/cryptoRL/logs"

    # Lancer TensorBoard en arriere-plan
    Write-Host "  -> Lancement de TensorBoard..." -ForegroundColor Gray
    Invoke-SSHCommand "nohup tensorboard --logdir=$WORKSPACE/cryptoRL/logs --port=$TB_PORT --bind_all > /tmp/tensorboard.log 2>&1 &"

    # Verifier que TensorBoard est lance
    Start-Sleep -Seconds 2
    $tbCheck = Invoke-SSHCommand "pgrep -f tensorboard"
    if ($tbCheck) {
        Write-Host "  [OK] TensorBoard lance sur le port $TB_PORT" -ForegroundColor Green
        Write-Host "  -> URL: http://${SSH_HOST}:${TB_PORT}" -ForegroundColor Cyan
    } else {
        Write-Host "  [ERREUR] TensorBoard n'a pas demarre" -ForegroundColor Red
        Write-Host "  -> Logs: ssh vast 'cat /tmp/tensorboard.log'" -ForegroundColor Gray
    }
} else {
    Write-Host "[10/10] Configuration TensorBoard... SKIP" -ForegroundColor DarkGray
}

# ============================================================================
# Resume
# ============================================================================
Write-Host ""
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "[OK] Initialisation terminee!" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Raccourcis disponibles:" -ForegroundColor Cyan
Write-Host "  ssh vast                              - Connexion rapide"
Write-Host "  scp -r file vast:/workspace/cryptoRL/ - Copier des fichiers"
Write-Host ""
Write-Host "URLs:" -ForegroundColor Cyan
Write-Host "  TensorBoard: http://${SSH_HOST}:${TB_PORT}"
Write-Host ""
Write-Host "Lancer le training:" -ForegroundColor Yellow
Write-Host "  ssh vast 'cd /workspace/cryptoRL; python3 scripts/run_full_wfo.py --segment 0 --timesteps 150000'"
Write-Host ""
