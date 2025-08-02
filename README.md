# Invoice OCR Project

A full-stack application for processing invoices using OCR technology with user authentication.

## Features

- **Invoice OCR Processing**: Extract text and data from uploaded invoices
- **User Authentication**: Secure login system with JWT tokens
- **User Management**: Create and manage users with different roles
- **Dashboard**: View and manage processed invoices
- **Modern UI**: Beautiful React frontend with Tailwind CSS

## Project Structure

```
invoice-ocr-project/
├── app/                    # Backend (FastAPI)
│   ├── main.py            # Main FastAPI application
│   ├── models.py          # Database models
│   ├── auth.py            # Authentication utilities
│   ├── db.py              # Database configuration
│   ├── ocr_processor.py   # OCR processing logic
│   ├── requirements.txt   # Python dependencies
│   ├── create_tables.py   # Database initialization
│   └── create_admin.py    # Admin user creation
└── Invoice-ocr-frontend/  # Frontend (React + TypeScript)
    ├── src/
    │   ├── pages/         # React pages
    │   ├── components/    # React components
    │   └── ...
    └── ...
```

## Setup Instructions

### Backend Setup

1. **Install Python dependencies**:
   ```bash
   cd app
   pip install -r requirements.txt
   ```

2. **Configure Database**:
   - Make sure PostgreSQL is running
   - Update database credentials in `app/db.py` if needed
   - Create the database: `invoice_ocr_db`

3. **Initialize Database**:
   ```bash
   cd app
   python create_tables.py
   python create_admin.py
   ```

4. **Start the Backend Server**:
   ```bash
   cd app
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

### Frontend Setup

1. **Install Node.js dependencies**:
   ```bash
   cd Invoice-ocr-frontend
   npm install
   ```

2. **Start the Frontend Development Server**:
   ```bash
   cd Invoice-ocr-frontend
   npm run dev
   ```

## Authentication

### Default Admin User
- **Email**: admin@example.com
- **Password**: admin123

### API Endpoints

- `POST /auth/login` - User login
- `POST /auth/register` - User registration
- `GET /auth/me` - Get current user info

### Frontend Routes

- `/login` - Login page
- `/dashboard` - Main dashboard (requires authentication)
- `/users` - User management (admin only)

## Usage

1. Start both backend and frontend servers
2. Navigate to `http://localhost:5173`
3. Login with the default admin credentials
4. Upload invoices and manage users

## Security Notes

- Change the `SECRET_KEY` in `app/auth.py` for production
- Use environment variables for sensitive configuration
- Implement proper password policies
- Add rate limiting for production use

## Technologies Used

### Backend
- FastAPI
- SQLAlchemy
- PostgreSQL
- PaddleOCR
- JWT Authentication

### Frontend
- React
- TypeScript
- Tailwind CSS
- Shadcn/ui Components
- React Router 