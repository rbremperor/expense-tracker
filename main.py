from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
import openai
import os
import re
from dotenv import load_dotenv
import asyncpg
from datetime import datetime
from typing import Optional, List
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database connection pool
db_pool = None

# OpenAI API key setup
openai.api_key = os.getenv("OPENAI_API_KEY")

# Serve static files (index.html)
app.mount("/static", StaticFiles(directory="templates"), name="static")


@app.on_event("startup")
async def startup():
    global db_pool
    try:
        db_pool = await asyncpg.create_pool(
            dsn=os.getenv("DATABASE_URL"),
            min_size=1,
            max_size=10,
            timeout=30
        )
        # Create expense table if it doesn't exist
        async with db_pool.acquire() as conn:
            await conn.execute('''
                               CREATE TABLE IF NOT EXISTS expenses
                               (
                                   id
                                   SERIAL
                                   PRIMARY
                                   KEY,
                                   title
                                   TEXT
                                   NOT
                                   NULL,
                                   category
                                   TEXT
                                   NOT
                                   NULL,
                                   amount
                                   DECIMAL
                               (
                                   10,
                                   2
                               ) NOT NULL,
                                   created_at TIMESTAMP DEFAULT NOW
                               (
                               )
                                   )
                               ''')
            # Add indexes for better performance
            await conn.execute('''
                               CREATE INDEX IF NOT EXISTS idx_expenses_category ON expenses(category);
                               CREATE INDEX IF NOT EXISTS idx_expenses_created_at ON expenses(created_at);
                               ''')
        logger.info("Database connection established and tables verified")
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        raise


@app.on_event("shutdown")
async def shutdown():
    if db_pool:
        await db_pool.close()
        logger.info("Database connection pool closed")


# Models
class Expense(BaseModel):
    title: str
    category: str
    amount: float

    @validator('amount')
    def amount_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Amount must be positive')
        return v


class ExpenseInput(BaseModel):
    description: str


class ExpenseResponse(BaseModel):
    id: int
    title: str
    category: str
    amount: float
    created_at: datetime


class ExpensesListResponse(BaseModel):
    expenses: List[ExpenseResponse]
    total: int


# Routes
@app.get("/", response_class=HTMLResponse)
async def get_html():
    try:
        with open("templates/index.html", "r") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except Exception as e:
        logger.error(f"Error loading HTML: {e}")
        raise HTTPException(status_code=500, detail="Error loading application")


@app.post("/add_expense/")
async def add_expense(expense_input: ExpenseInput):
    description = expense_input.description.strip()
    if not description:
        raise HTTPException(status_code=400, detail="Description cannot be empty")

    try:
        parsed_data = await parse_expense(description)
        async with db_pool.acquire() as conn:
            record = await conn.fetchrow(
                """INSERT INTO expenses(title, category, amount)
                   VALUES ($1, $2, $3) RETURNING id, title, category, amount, created_at""",
                parsed_data.title, parsed_data.category, parsed_data.amount
            )
        return {
            "message": "Expense added successfully",
            "data": record
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error adding expense: {e}")
        raise HTTPException(status_code=500, detail="Error adding expense")


@app.get("/get_expenses/", response_model=ExpensesListResponse)
async def get_expenses(
        category: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
):
    try:
        async with db_pool.acquire() as conn:
            if category and category.lower() != "all":
                expenses = await conn.fetch(
                    """SELECT id, title, category, amount, created_at
                       FROM expenses
                       WHERE LOWER(category) = $1
                       ORDER BY created_at DESC
                           LIMIT $2
                       OFFSET $3""",
                    category.lower(), limit, offset
                )
                total = await conn.fetchval(
                    "SELECT COUNT(*) FROM expenses WHERE LOWER(category) = $1",
                    category.lower()
                )
            else:
                expenses = await conn.fetch(
                    """SELECT id, title, category, amount, created_at
                       FROM expenses
                       ORDER BY created_at DESC
                           LIMIT $1
                       OFFSET $2""",
                    limit, offset
                )
                total = await conn.fetchval("SELECT COUNT(*) FROM expenses")
        return {"expenses": expenses, "total": total}
    except Exception as e:
        logger.error(f"Error fetching expenses: {e}")
        raise HTTPException(status_code=500, detail="Error fetching expenses")


@app.delete("/expenses/{expense_id}")
async def delete_expense(expense_id: int):
    try:
        async with db_pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM expenses WHERE id = $1",
                expense_id
            )
            if result == "DELETE 0":
                raise HTTPException(status_code=404, detail="Expense not found")
        return {"message": "Expense deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting expense: {e}")
        raise HTTPException(status_code=500, detail="Error deleting expense")


async def parse_expense(description: str) -> Expense:
    # Extract amount
    amount = 0.0
    if (match := re.search(r'(\d+\.?\d*)', description)):
        amount = float(match.group(1))

    try:
        response = await openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Using faster/cheaper model
            messages=[
                {
                    "role": "system",
                    "content": """You categorize expenses. Respond ONLY with:
                    <category>|<title>|<amount>
                    Categories: food, transportation, entertainment, shopping, bills, services, health, travel, other
                    Example: transportation|Oil change|45.00"""
                },
                {
                    "role": "user",
                    "content": f"Categorize: {description} (Amount found: {amount})"
                }
            ],
            temperature=0.0,
            max_tokens=50
        )

        # Parse response
        content = response.choices[0].message.content.strip()
        category, title, amount = content.split("|")

        return Expense(
            title=title.strip(),
            category=category.strip().lower(),
            amount=float(amount.strip())
        )

    except Exception as e:
        print(f"AI failed, using fallback: {e}")
        title = " ".join(description.split()[:-1]) if match else description
        return Expense(title=title, category="other", amount=amount)