# app/main.py
from fastapi import FastAPI, Request, UploadFile, File, Depends, Form
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from app.db import SessionLocal
from app.models import Invoice
from app.ocr_processor import process_invoice_file
import shutil
import os

app = FastAPI()
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# @app.get("/", response_class=HTMLResponse)
# def serve_upload():
#     with open("frontend/upload.html", "r") as f:
#         return HTMLResponse(f.read())

@app.get("/dashboard", response_class=HTMLResponse)
def serve_dashboard():
    with open("frontend/dashboard.html", "r") as f:
        return HTMLResponse(f.read())


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

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

@app.get("/edit/{invoice_id}", response_class=HTMLResponse)
def edit_invoice_form(invoice_id: int, db: Session = Depends(get_db)):
    invoice = db.query(Invoice).filter(Invoice.id == invoice_id).first()
    if not invoice:
        return HTMLResponse("<h2>Invoice not found</h2>", status_code=404)
    with open("frontend/edit_invoice.html", "r") as f:
        html = f.read()
    # Fill the form fields with invoice data
    for field in ["filename", "invoice_number", "invoice_date", "supplier_name", "total_ht", "tva", "total_ttc", "created_at"]:
        value = getattr(invoice, field)
        html = html.replace(f"{{{{{field}}}}}", str(value) if value is not None else "")
    html = html.replace("{{invoice_id}}", str(invoice.id))
    return HTMLResponse(html)

@app.post("/edit/{invoice_id}")
async def edit_invoice_submit(invoice_id: int, request: Request, db: Session = Depends(get_db)):
    form = await request.form()
    invoice = db.query(Invoice).filter(Invoice.id == invoice_id).first()
    if not invoice:
        return HTMLResponse("<h2>Invoice not found</h2>", status_code=404)
    for field in ["filename", "invoice_number", "invoice_date", "supplier_name", "total_ht", "tva", "total_ttc"]:
        if field in form:
            setattr(invoice, field, form[field])
    db.commit()
    return RedirectResponse(url="/dashboard", status_code=303)


