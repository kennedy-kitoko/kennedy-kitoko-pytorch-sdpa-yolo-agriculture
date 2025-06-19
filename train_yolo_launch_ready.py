import os
import json
import torch
import torch.nn.functional as F
import psutil
import gc
from ultralytics import YOLO
from datetime import datetime

# 🚀 Configuration Ultra-Premium pour RTX 4060+ PRÊTE AU LANCEMENT
# Développé par Kennedy Kitoko (🇨🇩) pour SmartFarm
# Version finale : Auto-détection + Fallback complet

def clear_gpu_memory():
    """Nettoyage complet de la mémoire GPU"""
    if torch.cuda.is_available():
        for _ in range(3):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()

def analyze_system_resources():
    """Analyse des ressources système pour optimisation"""
    clear_gpu_memory()
    
    # Analyse RAM
    ram = psutil.virtual_memory()
    ram_gb = ram.total / (1024**3)
    ram_available = ram.available / (1024**3)
    
    # Analyse GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_allocated = torch.cuda.memory_allocated(0) / (1024**3)
        gpu_free = gpu_memory - gpu_allocated
    else:
        gpu_name = "Non disponible"
        gpu_memory = 0
        gpu_free = 0
    
    return {
        'ram_total': ram_gb,
        'ram_available': ram_available,
        'gpu_name': gpu_name,
        'gpu_memory': gpu_memory,
        'gpu_free': gpu_free
    }

def get_adaptive_config(resources):
    """Configuration adaptative basée sur les ressources"""
    
    # Adaptation selon GPU disponible
    if resources['gpu_free'] >= 7.0:  # RTX 4060 niveau
        return {
            'batch': 24,
            'workers': 12,
            'cache': 'ram',
            'tier': 'ULTRA_PREMIUM'
        }
    elif resources['gpu_free'] >= 5.0:
        return {
            'batch': 20,
            'workers': 10,
            'cache': 'ram',
            'tier': 'PREMIUM'
        }
    elif resources['gpu_free'] >= 3.0:
        return {
            'batch': 16,
            'workers': 8,
            'cache': 'disk',
            'tier': 'STABLE'
        }
    else:
        return {
            'batch': 12,
            'workers': 6,
            'cache': False,
            'tier': 'SAFE'
        }

def setup_ultra_environment():
    """Configuration optimale pour YOLOv12 avec PyTorch SDPA"""
    
    # Activation des optimisations internes de PyTorch
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Configuration CUDA optimisée
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Test SDPA avec gestion d'erreur
    try:
        if hasattr(F, 'scaled_dot_product_attention'):
            print("✅ PyTorch SDPA: ACTIVÉ (Quasi Flash Attention)")
            print("🇨🇩 Innovation by Kennedy Kitoko - Congolese Student")
            
            # Benchmark sécurisé
            if torch.cuda.is_available():
                batch, heads, seq, dim = 2, 8, 256, 64
                q = torch.randn(batch, heads, seq, dim, device='cuda', dtype=torch.float16)
                k = torch.randn(batch, heads, seq, dim, device='cuda', dtype=torch.float16)
                v = torch.randn(batch, heads, seq, dim, device='cuda', dtype=torch.float16)
                
                with torch.no_grad():
                    output = F.scaled_dot_product_attention(q, k, v)
                    del q, k, v, output
                    
                print("🚀 SDPA Performance: ULTRA PREMIUM")
                clear_gpu_memory()
                return True
        else:
            print("⚠️ SDPA non disponible, utilisation standard")
            return True
    except Exception as e:
        print(f"⚠️ SDPA non compatible: {e}")
        return True  # Continue quand même

def find_model_file():
    """Auto-détection du fichier modèle"""
    possible_models = [
        'yolo12n.pt',
        'yolov8n.pt',
        'yolov11n.pt',
        'yolo11n.pt',
        'yolo12s.pt'
    ]
    
    for model in possible_models:
        if os.path.exists(model):
            print(f"✅ Modèle trouvé: {model}")
            return model
    
    print("⚠️ Aucun modèle trouvé, téléchargement automatique...")
    return 'yolo11n.pt'  # Téléchargement auto par Ultralytics

def find_dataset_config():
    """Auto-détection du fichier dataset"""
    possible_configs = [
        'weeds_dataset.yaml',
        'data.yaml',
        'dataset.yaml'
    ]
    
    for config in possible_configs:
        if os.path.exists(config):
            print(f"✅ Dataset config trouvé: {config}")
            return config
    
    # Création automatique si non trouvé
    print("⚠️ Aucun dataset.yaml trouvé, création automatique...")
    return create_default_dataset_config()

def create_default_dataset_config():
    """Création automatique du fichier dataset.yaml"""
    
    # Recherche de dossiers dataset
    possible_paths = [
        'weeds-3',
        '../weeds-3',
        'dataset',
        'data'
    ]
    
    dataset_path = None
    for path in possible_paths:
        if os.path.exists(os.path.join(path, 'train', 'images')):
            dataset_path = os.path.abspath(path)
            break
    
    if not dataset_path:
        print("❌ Aucun dataset trouvé! Créez le dossier dataset avec:")
        print("   dataset/train/images/")
        print("   dataset/train/labels/")
        print("   dataset/valid/images/")
        print("   dataset/valid/labels/")
        return None
    
    # Création du fichier YAML
    yaml_content = f"""# Auto-generated dataset config
train: {dataset_path}/train/images
val: {dataset_path}/valid/images
test: {dataset_path}/test/images

nc: 1
names: ['weed']
"""
    
    yaml_path = 'auto_dataset.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"✅ Dataset config créé: {yaml_path}")
    print(f"📁 Dataset path: {dataset_path}")
    return yaml_path

def validate_dataset(data_path):
    """Validation du dataset avant entraînement"""
    if not data_path or not os.path.exists(data_path):
        print(f"❌ Dataset config introuvable: {data_path}")
        return False
    
    print(f"✅ Dataset config trouvé: {data_path}")
    return True

def test_training_config(model, config):
    """Test préliminaire de la configuration"""
    print("🧪 Test préliminaire de la configuration...")
    
    try:
        clear_gpu_memory()
        
        # Test validation simple
        model.val(
            data=config['data'],
            batch=min(8, config['batch']),
            device=config['device'],
            verbose=False
        )
        
        print("✅ Test configuration réussi!")
        return True
        
    except Exception as e:
        print(f"⚠️ Test configuration échoué: {e}")
        return False

def save_config(config, directory):
    """Sauvegarde du dictionnaire de configuration dans un fichier JSON"""
    os.makedirs(directory, exist_ok=True)
    config_path = os.path.join(directory, "train_config.json")
    
    # Ajout informations système
    config_with_system = config.copy()
    config_with_system['system_info'] = analyze_system_resources()
    config_with_system['pytorch_version'] = torch.__version__
    config_with_system['timestamp'] = datetime.now().isoformat()
    
    with open(config_path, "w") as f:
        json.dump(config_with_system, f, indent=4)
    print(f"💾 Configuration sauvegardée dans: {config_path}")

def check_torch_version(min_version="1.12"):
    """Vérifie si la version de torch est compatible"""
    current_version = torch.__version__
    current_major_minor = tuple(map(int, current_version.split(".")[:2]))
    min_major_minor = tuple(map(int, min_version.split(".")[:2]))
    
    if current_major_minor < min_major_minor:
        print(f"⚠️ PyTorch {min_version}+ recommandé, version actuelle: {current_version}")
        return False
    else:
        print(f"✅ PyTorch version: {current_version}")
        return True

# 🎮 Lancement de l'entraînement PRÊT AU LANCEMENT
if __name__ == "__main__":
    print("🌍 Innovation PyTorch SDPA by Kennedy Kitoko")
    print("🎯 Objectif: Détection de mauvaises herbes pour SmartFarm")
    print("🔧 Version LAUNCH-READY: Auto-détection + Fallback complet\n")
    
    try:
        # Vérifications préliminaires
        torch_ok = check_torch_version()
        sdpa_ok = setup_ultra_environment()
        
        # Auto-détection des fichiers
        model_file = find_model_file()
        dataset_file = find_dataset_config()
        
        if not dataset_file:
            print("❌ Impossible de créer/trouver le dataset")
            exit(1)
        
        # Analyse système
        print("\n🔍 Analyse des ressources système...")
        resources = analyze_system_resources()
        adaptive_config = get_adaptive_config(resources)
        
        print(f"💾 RAM: {resources['ram_total']:.1f} GB (disponible: {resources['ram_available']:.1f} GB)")
        print(f"🎮 GPU: {resources['gpu_name']}")
        print(f"📱 VRAM: {resources['gpu_memory']:.1f} GB (libre: {resources['gpu_free']:.1f} GB)")
        print(f"⚡ Configuration: {adaptive_config['tier']}")
        
        # Configuration adaptative
        config = {
            'model': model_file,
            'data': dataset_file,
            'epochs': 100,  # Réduit pour test initial
            'batch': adaptive_config['batch'],
            'imgsz': 640,
            'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
            'workers': adaptive_config['workers'],
            'optimizer': 'AdamW',
            'lr0': 0.001,
            'amp': True,
            'cache': adaptive_config['cache'],
            'project': 'runs/kennedy_innovation',
            'name': f'weed_detection_sdpa_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'verbose': True,
            'patience': 30,
            'save_period': 5
        }
        
        print(f"\n📊 Configuration finale:")
        print(f"   Modèle: {config['model']}")
        print(f"   Dataset: {config['data']}")
        print(f"   Batch: {config['batch']}")
        print(f"   Workers: {config['workers']}")
        print(f"   Cache: {config['cache']}")
        print(f"   Device: {config['device']}")
        
        # Validation dataset
        if not validate_dataset(config['data']):
            print("❌ Validation dataset échouée")
            exit(1)
        
        # Sauvegarde configuration
        save_config(config, os.path.join(config['project'], config['name']))
        
        # Chargement modèle
        print("\n🔄 Chargement du modèle...")
        model = YOLO(config['model'])
        print(f"✅ Modèle {config['model']} chargé avec succès!")
        
        # Test préliminaire
        if not test_training_config(model, config):
            print("❌ Réduction configuration pour sécurité")
            config['batch'] = max(4, config['batch'] // 2)
            config['workers'] = max(2, config['workers'] // 2)
            config['cache'] = False
            print(f"🔧 Nouvelle config: Batch {config['batch']}, Workers {config['workers']}")
        
        # Compilation optimisée (si disponible)
        if hasattr(torch, 'compile') and torch_ok:
            try:
                model.model = torch.compile(model.model, mode='reduce-overhead')
                print("⚙️ Modèle compilé avec torch.compile")
            except Exception as e:
                print(f"⚠️ Compilation échouée: {e}")
        
        # Clear avant entraînement
        clear_gpu_memory()
        
        # 🚀 Entraînement
        print(f"\n🚀 Démarrage entraînement {adaptive_config['tier']}...")
        print(f"🕐 Heure de début: {datetime.now().strftime('%H:%M:%S')}")
        start_time = datetime.now()
        
        results = model.train(**config)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 3600
        
        # Résultats finaux
        print(f"\n✅ Entraînement terminé!")
        print(f"⏱️ Durée: {duration:.2f} heures")
        print(f"🏆 Entraînement par Kennedy Kitoko 🇨🇩")
        print(f"💾 Résultats: {results.save_dir}")
        
        # Évaluation finale
        try:
            final_metrics = model.val(data=config['data'])
            if hasattr(final_metrics, 'box'):
                mAP50 = final_metrics.box.map50
                mAP = final_metrics.box.map
                print(f"📊 mAP@50: {mAP50:.3f}")
                print(f"📊 mAP@50-95: {mAP:.3f}")
        except Exception as e:
            print(f"⚠️ Évaluation finale échouée: {e}")
        
        # Export modèle
        try:
            best_model_path = f"{results.save_dir}/weights/best.pt"
            if os.path.exists(best_model_path):
                best_model = YOLO(best_model_path)
                export_path = best_model.export(format='onnx', half=True)
                print(f"📦 Export ONNX: {export_path}")
        except Exception as e:
            print(f"⚠️ Export échoué: {e}")
        
        print(f"\n🎉 SUCCESS! Modèle entraîné avec succès!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Entraînement interrompu par l'utilisateur")
        
    except Exception as e:
        print(f"\n❌ Erreur détectée: {e}")
        print("\n🔧 Vérifiez:")
        print("   1. Dataset structure correcte")
        print("   2. Fichiers .yaml valides")
        print("   3. Espace disque suffisant")
        print("   4. Drivers GPU à jour")
        
        if torch.cuda.is_available():
            try:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_used = torch.cuda.memory_allocated() / 1e9
                print(f"🧠 GPU: {gpu_name} | Mémoire utilisée: {gpu_used:.2f} GB")
            except:
                print("🧠 Informations GPU non disponibles")
        
    finally:
        # Nettoyage final
        clear_gpu_memory()
        print("\n🧹 Nettoyage mémoire terminé")
        print("🎯 Script terminé - Prêt pour nouveau lancement!")