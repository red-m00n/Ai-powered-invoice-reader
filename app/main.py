# app/main.py
from fastapi import FastAPI, Request, UploadFile, File, Depends, Form, HTTPException, status
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from app.db import SessionLocal
from app.models import Invoice, User
from app.ocr_processor import process_invoice_file
from app.auth import authenticate_user, create_access_token, create_user, verify_token
from datetime import timedelta
import shutil
import os

app = FastAPI()

# CORS middleware - MUST BE FIRST
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=False,  # Set to False when using allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

# Test endpoint to verify CORS is working
@app.get("/test-cors")
async def test_cors():
    return {"message": "CORS is working!", "status": "success"}

# HTML serving routes (commented out since we're using React)
# @app.get("/", response_class=HTMLResponse)
# def serve_upload():
#     with open("frontend/upload.html", "r") as f:
#         return HTMLResponse(f.read())

# @app.get("/dashboard", response_class=HTMLResponse)
# def serve_dashboard():
#     with open("frontend/dashboard.html", "r") as f:
#         return HTMLResponse(f.read())


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Authentication endpoints
@app.post("/auth/login")
async def login(email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    """Login endpoint"""
    user = authenticate_user(db, email, password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=30)
    access_token = create_access_token(
        data={"sub": user.email, "user_id": user.id, "role": user.role}, 
        expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "email": user.email,
            "full_name": user.full_name,
            "role": user.role
        }
    }

@app.post("/auth/register")
async def register(
    full_name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    role: str = Form("user"),
    db: Session = Depends(get_db)
):
    """Register endpoint"""
    try:
        user = create_user(db, full_name, email, password, role)
        return {
            "message": "User created successfully",
            "user": {
                "id": user.id,
                "email": user.email,
                "full_name": user.full_name,
                "role": user.role
            }
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user"
        )

@app.get("/auth/me")
async def get_current_user(token: str, db: Session = Depends(get_db)):
    """Get current user info"""
    payload = verify_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user = db.query(User).filter(User.id == payload.get("user_id")).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return {
        "id": user.id,
        "email": user.email,
        "full_name": user.full_name,
        "role": user.role
    }

# API: Get all users (admin only)
@app.get("/users")
def get_all_users(db: Session = Depends(get_db)):
    users = db.query(User).all()
    return [
        {
            "id": user.id,
            "full_name": user.full_name,
            "email": user.email,
            "role": user.role,
            "created_at": user.created_at.isoformat() if user.created_at else None
        }
        for user in users
    ]

# Upload route (same as before)
@app.post("/upload")
async def upload_invoice(file: UploadFile = File(...), db: Session = Depends(get_db)):
    file_location = f"uploads/{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)


    try:
        process_invoice_file(file_location)
    except Exception as e:
        return {"error": f"Failed to add invoice to database: {str(e)}"}

    return {
        "message": "File uploaded successfully",
        "filename": file.filename,
    }

# API: Get all invoices
@app.get("/invoices")
def get_all_invoices(db: Session = Depends(get_db)):
    invoices = db.query(Invoice).all()
    return [invoice.__dict__ for invoice in invoices]



# API: Delete an invoice
@app.delete("/invoices/{invoice_id}")
def delete_invoice(invoice_id: int, db: Session = Depends(get_db)):
    invoice = db.query(Invoice).filter(Invoice.id == invoice_id).first()
    if not invoice:
        return {"error": "Invoice not found"}

    db.delete(invoice)
    db.commit()
    return {"message": "Invoice deleted"}

# HTML serving routes (commented out since we're using React)
# @app.get("/edit/{invoice_id}", response_class=HTMLResponse)
# def edit_invoice_form(invoice_id: int, db: Session = Depends(get_db)):
#     invoice = db.query(Invoice).filter(Invoice.id == invoice_id).first()
#     if not invoice:
#         return HTMLResponse("<h2>Invoice not found</h2>", status_code=404)
#     with open("frontend/edit_invoice.html", "r") as f:
#         html = f.read()
#     # Fill the form fields with invoice data
#     for field in ["filename", "invoice_number", "invoice_date", "supplier_name", "total_ht", "tva", "total_ttc", "created_at"]:
#         value = getattr(invoice, field)
#         html = html.replace(f"{{{{{field}}}}}", str(value) if value is not None else "")
#     html = html.replace("{{invoice_id}}", str(invoice.id))
#     return HTMLResponse(html)

# @app.post("/edit/{invoice_id}")
# async def edit_invoice_submit(invoice_id: int, request: Request, db: Session = Depends(get_db)):
#     form = await request.form()
#     invoice = db.query(Invoice).filter(Invoice.id == invoice_id).first()
#     if not invoice:
#         return HTMLResponse("<h2>Invoice not found</h2>", status_code=404)
#     for field in ["filename", "invoice_number", "invoice_date", "supplier_name", "total_ht", "tva", "total_ttc"]:
#         if field in form:
#             setattr(invoice, field, form[field])
#     db.commit()
#     return RedirectResponse(url="/dashboard", status_code=303)


