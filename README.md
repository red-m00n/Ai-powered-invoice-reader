# Invoice OCR Project

Ce projet extrait le texte de factures PDF à l’aide de PaddleOCR, puis stocke le résultat dans une base PostgreSQL.

## Structure

- `backend/` : Scripts Python (OCR, DB)
- `uploads/` : Placez vos fichiers PDF ici (ex: `invoice_sample.pdf`)
- `frontend/` : (Optionnel) Interface HTML
- `README.md` : Ce fichier

## Installation

1. Installez Python ≥ 3.8 et PostgreSQL.
2. Créez la base et la table :

   ```sql
   CREATE DATABASE invoice_ocr_db;
   \c invoice_ocr_db
   CREATE TABLE invoices (
       id SERIAL PRIMARY KEY,
       filename VARCHAR(255) NOT NULL,
       ocr_text TEXT NOT NULL,
       created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
   );
   ```

3. Installez les dépendances :

   ```sh
   cd backend
   pip install -r requirements.txt
   ```

4. Placez un PDF dans `uploads/` nommé `invoice_sample.pdf`.

## Utilisation

```sh
python main.py ../uploads/invoice_sample.pdf
```

Le texte extrait s’affichera et sera inséré dans la base PostgreSQL.

## Configuration

Modifiez `db.py` si besoin pour vos identifiants PostgreSQL.

---

**Prêt à l’emploi !**  
N’oublie pas de placer un fichier `invoice_sample.pdf` dans `uploads/` avant de lancer le script. 