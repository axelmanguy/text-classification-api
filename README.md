# text-classification-api 

Ce dépôt contient une **API de classification de texte** basée sur un modèle de Machine Learning
Des propositions sont faites pour des modèles utilsiant du Deep Learning ou des appels a des LLMs sont proposés.
L'API est développée avec **FastAPI**, avec des routes de metrics pour Prometheus et grafana
et déploiement possible via **Docker**.  

---

## **Choix des modèles et justification**  

L'API prend en charge plusieurs modèles de classification de texte. Voici les principaux choix disponibles :

| Modèle | Type             | Avantages                                              | Inconvénients                                                                                                 |
|--------|------------------|--------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| **SVM + TF-IDF** | Machine Learning | Rapide, léger, peu de dépendances                      | Performances limitées sur du texte complexe                                                                   |
| **CamemBERT** | Deep Learning    | Performant sur le français, supporte des phrases longues | Neccessite de la RAM et un GPU                                                                                |
| **LLM (Large Language Models)** | API externe      | Performant sur une grande varietés de taches           | neccesite un deploiement internalisé de LLM ou une API de fournisseur de LLM (OpenAI, Anthropic, Mistral ...) |

### **Compromis**  
- **SVM + TF-IDF** : Facile à déployer, idéal pour des API légères. Permet un passage a l'echelle sur infrastructure CPU
- **CamemBERT** : Spécialisé en français, adapté aux phrases complexes.  Plus complexe a passer la l'echelle sans infrastructure adaptées
- **LLM** : Meilleur sur des tâches avancées, mais plus gourmand en ressources. Requiert des investissement importants pour un bénéfice sur la tache a resoudre encore a demontrer

## ** Methodologie**  

Plusieurs choix ont été envisagés pour resoudre le probleme :

| Modèle | Type             | Avantages                                              | Inconvénients                                                                                                 |
|--------|------------------|--------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| **SVM + TF-IDF** | Machine Learning | Rapide, léger, peu de dépendances                      | Performances limitées sur du texte complexe                                                                   |
| **CamemBERT** | Deep Learning    | Performant sur le français, supporte des phrases longues | Neccessite de la RAM et un GPU                                                                                |
| **LLM (Large Language Models)** | API externe      | Performant sur une grande varietés de taches           | neccesite un deploiement internalisé de LLM ou une API de fournisseur de LLM (OpenAI, Anthropic, Mistral ...) |

### **Compromis**  
- **SVM + TF-IDF** : Facile à déployer, idéal pour des API légères. Permet un passage a l'echelle sur infrastructure CPU
- **CamemBERT** : Spécialisé en français, adapté aux phrases complexes.  Plus complexe a passer la l'echelle sans infrastructure adaptées
- **LLM** : Meilleur sur des tâches avancées, mais plus gourmand en ressources. Requiert des investissement importants pour un bénéfice sur la tache a resoudre encore a demontrer

### **Approche SVM + TF-IDF**

Approche basée sur la litterature recente qui repose sur des methodes de ML par ailleurs epprouvés mais neccesitant un travail appronfondi de feature engineering
Compte des contraintes de temps le feature engineering a été limités aux actions suivantes :
- **Consolidation des données** : resolution des conflicts entre annotateur par vote majoritaire
- **Choix des features TF-IDF** : utilisation des fonctions standard de sklearn -> approche naive mais rapide
Le modèle a par ailleur été choisi en suivant les recommandations de [Wahba, Y., Madhavji, N., & Steinbacher, J. (2022)](https://arxiv.org/abs/2211.02563v1)

Pour ameliorer les resultats plusieurs autres approches sont possibles :

- **Optimization** : Recherche des meilleurs hyperparametres des modèles par grid search
- **Meilleur preprocess des données** : le TfidfVectorizer ne supporte pas d'autres stop words que ceux pour l'anglais, un preprocessing via nltk pour une meilleur tokenization est une piste serieuse

En matiere d'evaluation les actions suivantes ont été ménés pour minimiser les biais :

- **Séparation des datasets** : les datasets ont été coupé en 3 segments, train, test, et validation. les jeux de train et test sont utilisé pour choisir le modèle et les hyperparametres. Le jeu de validation sert a l'evaluation finale du modèle sur une partie du corpus non vue durant l'apprentissage
- **KFold validation** Pour le choix des hyperparametres la validation croisée permet de minimiser les biais et de donner une mesure statistique pertinente pour le modèle

Pistes d'amélioration :

- **Interval de confiance** : l'évaluation gagnerait en rigueur si les resultats sur le jeu de test final
etaient donnés avec un interval de confiance plutot qu'avec une simple moyenne et un F1-score
- **Point de fonctionnement** : le choix du seuil de detection devrait relever d'une recherche de compromis entre faux positif et faux negatif et resultats d'une etude de la courbe ROC a different mesure de seuil

### **Approche Finetuning de CamemBERT**

Approche basée sur la litterature recente de traitement automatique des langues pour le francais,
[CamemBERT 2.0](https://arxiv.org/abs/2411.08868) est une architectue recente et pre-entrainée sur le français.
L'idée et de l'adapter pour une tache de classification binaire (ok/nok) via un finetuning

#### **Limites**

Plusieurs limites rendent cette approche difficile :

- **Taille du dataset** : le volumes de données pour entrainer est extremement limité et il parait difficile d'envisager un finetuning en l'etat
- **Ressources de calcul** : le finetuning (et l'inference en l'abscence d'optimisation pour CPU) requiert 
l'usage d'un GPU, ce qui peut etre une ressource contrainte au sein de certaines infrastructures

Un code d'entrainement adapté d'un projet precedent est tout de meme fourni pour ce probleme ci 

#### **Proposition pour l'augmentation de données**

Pour augmenter le volume de donnée, une approche recente encore trés experimentale consiste en l'usage de Large language model pour creer de la données synthethique.
On propose de creer 3 prompts specifiques pour la génération de données synthethiques :

- **Prompt de génération OK** : Le contexte de génération est enrichi avec N exemples aléatoire (par exemple 3) de textes conformes pris aleatoirement
dans le corpus. Le LLM recoit des instruction pour génèrer du contenu simillaire mais significativement different.
On propose de mesurer la similarité via une fonction d'embeding (E5 ou BGE-M3 par exemple) et de ne selectioner que les textes générè significativmeent different du corpus initial.
Pour minimiser les biais on prendra soin de générer le corpus avec plusieurs LLM distincts (par exemple LLAMA3, Mixtral, Qwen)
- **Prompt de génération NOK** : On reproduit la methode uniquement pour les exemples non conformes
- **Prompt 'LLM as a judge'** : on ecrit un prompt pour l'évaluation de la pertinence des textes générés.
Cette approche repose sur la conception soigneuse d'un prompt d'evaluation precisant l'ensemble des critères (originalité, respect des mots clefs ect) 

En combinant, ces 3 prompts et en utilisant plusieurs LLMs, il est possible d'obtenir un corpus synthetique beaucoup plus important que le corpus original
Cette approche ne remplace pas toutefois une donnée naturelle et comprends de trés nombreux biais liée a la diversité des données du corpus de base et aux biais inherent aux grand modèles de language.

---

## **Utilisation du Makefile**  

Le projet propose un **Makefile** pour simplifier les commandes courantes.
Voici les principales fonctionnalités disponibles :  

### **Lancer l'entraînement du modèle**
```sh
make train
```
Cette commande entraîne le modèle en utilisant `src/training/train_sklearn_pipeline.py`.  
Elle garantit que le modèle est bien mis à jour avant son utilisation.  

---

### **Démarrer l'API FastAPI**
```sh
make run
```
Cette commande démarre l'API avec **Uvicorn**, en mode **reload** (utile en développement).  
L'API sera accessible à l'adresse **http://127.0.0.1:8000**  

---

### **Exécuter les tests End-to-End**
```sh
make e2e-test
```
Cette commande exécute les tests E2E avec **pytest** en utilisant le fichier `test/e2e_test/test_api.py`.  
Elle permet de vérifier que l'API fonctionne correctement avant un déploiement.  

---

### **Construire l'image Docker**
```sh
make docker-build
```
Cette commande construit une image Docker nommée **text-classifier-api**.  
Elle permet de préparer l'application pour une exécution en conteneur.  

---

### **Exécuter le conteneur Docker**
```sh
make docker-run
```
Cette commande exécute l'API dans un conteneur Docker et l'expose sur le port `8000`.  
L'API sera accessible à **http://127.0.0.1:8000**  


## **1. Installation**  

### **1.1 Installation depuis le dépôt**  

#### **Pré-requis**  
- **Python 3.10** ou plus récent  
- **pip**
- **Docker**  

#### **Étapes d'installation**  

1. **Cloner le dépôt**  
   ```sh
   git clone https://github.com/votre-utilisateur/text-classification-api.git
   cd text-classification-api
   ```

2. **Créer un environnement virtuel et installer les dépendances**  
   ```sh
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
---

### **1.2 Installation avec Docker**  

1. **Construire l'image Docker**  
   ```sh
   docker build -t text-classifier-api -f deployment/Dockerfile .
   ```

2. **Lancer l'API**  
   ```sh
   docker run -p 8000:8000 text-classifier-api
   ```

3. **Vérifier le bon fonctionnement**  
   - Ouvrir **Swagger UI** : [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)  
   - Vérifier l'état de l'API :  
     ```sh
     curl -X GET "http://127.0.0.1:8000/health/"
     ```

---

## **2. Lancer l'API**  

Si vous utilisez l'installation **hors Docker**:  
```sh
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

Si vous utilisez Docker :  
```sh
docker run -p 8000:8000 text-classifier-api
```

L'API est ouverte sur l'url **http://127.0.0.1:8000**.

---

## **3. Tests**  

Le projet utilise **pytest** pour tester l'API et les composants internes.

### **3.1 Lancer les tests unitaires et d'intégration**  
```sh
pytest tests/
```

### **3.2 Tester l'API avec cURL**  
1. **Vérifier la disponibilité de l'API**  
   ```sh
   curl -X GET "http://127.0.0.1:8000/health/"
   ```
   
2. **Envoyer une requête de classification de texte**  
   ```sh
   curl -X POST "http://127.0.0.1:8000/predict/" -H "Content-Type: application/json" -d '{"text": "Exemple de phrase"}'
   ```

3. **Vérifier les métriques (Prometheus)**  
   ```sh
   curl -X GET "http://127.0.0.1:8000/metrics/"
   ```

---

## Propositions de travaux futurs

Ce projet fournit une **API** pour la classification de texte.
Avec plus de temps il aurait pu intégrer la possibilité de choisir entre plusieurs types de modèles 
ou de faire un vote majoritaire pour combiner leurs predictions.

## References
- [Antoun, W., Kulumba, F., Touchent, R., de la Clergerie, É., Sagot, B., & Seddah, D. (2024). CamemBERT 2.0: A Smarter French Language Model Aged to Perfection. arXiv preprint arXiv:2411.08868.](https://arxiv.org/abs/2411.08868)
- [Wahba, Y., Madhavji, N., & Steinbacher, J. (2022, September). A comparison of svm against pre-trained language models (plms) for text classification tasks. In International Conference on Machine Learning, Optimization, and Data Science (pp. 304-313). Cham: Springer Nature Switzerland.](https://arxiv.org/abs/2211.02563v1)



