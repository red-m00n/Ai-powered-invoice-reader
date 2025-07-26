from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


DB_USER = "postgres"
DB_PASSWORD = "hachad123"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "invoice_ocr_db"

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine) 