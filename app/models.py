from sqlalchemy import Column, Integer, String, Text, DateTime, Numeric, Date, func
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String(100), nullable=False)
    email = Column(String(100), unique=True, nullable=False, index=True)
    role = Column(String(20), nullable=False, default="user")
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Invoice(Base):
    __tablename__ = "invoices"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    invoice_number = Column(String(64))
    invoice_date = Column(Date)
    supplier_name = Column(String(255))
    total_ht = Column(Numeric(12, 2))
    tva = Column(Numeric(12, 2))
    total_ttc = Column(Numeric(12, 2))
    ocr_text = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now()) 

