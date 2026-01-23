#!/bin/bash
# Script d'initialisation du serveur cryptoRL
# Usage: ./scripts/init_server.sh

set -e

SSH_HOST="212.93.107.107"
SSH_PORT="40075"
SSH_USER="root"
WORKSPACE="/workspace"

echo "═══════════════════════════════════════════════════════════════"
echo "Initialisation du serveur cryptoRL"
echo "═══════════════════════════════════════════════════════════════"
echo "Host: $SSH_HOST"
echo "Port: $SSH_PORT"
echo "User: $SSH_USER"
echo ""

# Fonction pour exécuter une commande sur le serveur
ssh_cmd() {
    ssh -p $SSH_PORT $SSH_USER@$SSH_HOST "$1"
}

# 1. Test de connexion
echo "[1/8] Test de connexion..."
if ssh_cmd "echo 'Connexion OK'"; then
    echo "✅ Connexion réussie"
else
    echo "❌ Échec de connexion"
    exit 1
fi

# 2. Vérifier Python
echo ""
echo "[2/8] Vérification de Python..."
PYTHON_VERSION=$(ssh_cmd "python3 --version 2>&1 || echo 'Python non trouvé'")
echo "Python: $PYTHON_VERSION"

# 3. Vérifier GPU
echo ""
echo "[3/8] Vérification GPU..."
GPU_INFO=$(ssh_cmd "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 || echo 'GPU non disponible'")
echo "GPU: $GPU_INFO"

# 4. Créer le répertoire workspace si nécessaire
echo ""
echo "[4/8] Préparation du workspace..."
ssh_cmd "mkdir -p $WORKSPACE && cd $WORKSPACE && pwd"
echo "✅ Workspace prêt"

# 5. Vérifier si le repo existe déjà
echo ""
echo "[5/8] Vérification du repository..."
if ssh_cmd "test -d $WORKSPACE/cryptoRL"; then
    echo "⚠️  Le repository existe déjà"
    read -p "Voulez-vous le supprimer et le cloner à nouveau? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        ssh_cmd "rm -rf $WORKSPACE/cryptoRL"
        echo "✅ Ancien repository supprimé"
    else
        echo "ℹ️  Utilisation du repository existant"
        echo "[6/8] Mise à jour du repository..."
        ssh_cmd "cd $WORKSPACE/cryptoRL && git pull"
        echo "✅ Repository mis à jour"
        skip_clone=true
    fi
else
    skip_clone=false
fi

# 6. Clone du repository (si nécessaire)
if [ "$skip_clone" = false ]; then
    echo ""
    echo "[6/8] Clone du repository GitHub..."
    echo "⚠️  Assurez-vous que SSH est configuré pour GitHub sur le serveur"
    ssh_cmd "cd $WORKSPACE && git clone git@github.com:randysitcharn/cryptoRL.git || echo 'Erreur: Vérifiez la configuration SSH GitHub'"
    echo "✅ Repository cloné"
fi

# 7. Installation des dépendances
echo ""
echo "[7/8] Installation des dépendances..."
ssh_cmd "cd $WORKSPACE/cryptoRL && pip install -r requirements.txt"
echo "✅ Dépendances installées"

# 8. Vérification finale
echo ""
echo "[8/8] Vérification finale..."
ssh_cmd "cd $WORKSPACE/cryptoRL && python3 -c 'import torch; print(f\"PyTorch: {torch.__version__}\"); print(f\"CUDA disponible: {torch.cuda.is_available()}\")'"
echo ""

echo "═══════════════════════════════════════════════════════════════"
echo "✅ Initialisation terminée!"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Prochaines étapes:"
echo "1. Copier les données: scp -P $SSH_PORT -r data/raw_historical/ $SSH_USER@$SSH_HOST:$WORKSPACE/cryptoRL/data/"
echo "2. Configurer TensorBoard sur le serveur"
echo "3. Lancer le training: ssh -p $SSH_PORT $SSH_USER@$SSH_HOST 'cd $WORKSPACE/cryptoRL && python3 scripts/run_full_wfo.py --segment 0 --timesteps 150000'"
echo ""
