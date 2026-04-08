# README.md

# Finance/Invoice Management Backend

## Features
- Secure JWT authentication (email/password)
- Multi-tenant data isolation (single DB, strict tenant checks)
- Professional code structure
- MongoDB Atlas integration
- Data import script for invoices

## Folder Structure
- config/ — Settings and environment
- models/ — Pydantic models
- routes/ — API endpoints
- middleware/ — Auth middleware
- services/ — Business logic
- utils/ — Security helpers
- scripts/ — Data import/seed

## Setup
1. Copy `config/.env.example` to `.env` and fill in your secrets.
2. Install dependencies:
   ```bash
   pip install fastapi uvicorn pymongo bcrypt python-jose pydantic
   ```
3. Import invoices:
   ```bash
   python scripts/import_invoices.py
   ```
4. Create users (see `services/user_service.py` for example).
5. Run the API:
   ```bash
   uvicorn main:app --reload
   ```

## API Endpoints
- POST /auth/login
- GET /auth/me
- GET /invoices
- GET /invoices/{id}

## Security
- Passwords hashed (bcrypt)
- JWT required for all invoice endpoints
- Tenant access strictly enforced

## Notes
- Never trust tenant input from frontend
- All queries filter by allowed_tenants
- Extendable for more tenants, roles, or microservices
