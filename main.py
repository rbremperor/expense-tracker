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
    # ... (previous code remains the same until the category validation)

    # Enhanced category validation
    valid_categories = {
        "food", "transportation", "entertainment",
        "shopping", "bills", "services",
        "health", "travel", "other"  # Ensure 'other' is included
    }

    # Force 'other' if not in valid categories
    category = category.lower() if category.lower() in valid_categories else "other"

    # Check for transport keywords first (client-side matching)
    description_lower = description.lower()
    if any(keyword in description_lower for keyword in transport_keywords):
        title = description[:amount_match.start()].strip() if amount_match else description
        return Expense(title=title, category="transportation", amount=amount)

    try:
        prompt = f"""
        Analyze this expense description and categorize it: "{description}"

        Respond STRICTLY in this format:
        title|category|amount

        Categories (MUST USE THESE EXACT NAMES):
        - Food: Groceries, restaurants, coffee, snacks
        - Transportation: Gas, oil, car maintenance, repairs, parking, public transit
        - Entertainment: Movies, games, concerts, streaming
        - Shopping: Physical goods, clothes, electronics
        - Bills: Regular payments, utilities, subscriptions
        - Services: Professional services, repairs
        - Health: Medical, pharmacy, fitness
        - Travel: Hotels, flights, vacation
        - Other: Anything else

        IMPORTANT TRANSPORTATION EXAMPLES:
        "motor oil" → Motor oil|Transportation|12.99
        "oil change" → Oil change|Transportation|45.00
        "car wash" → Car wash|Transportation|15.00
        "tires" → Tires|Transportation|200.00
        "gas station" → Gas|Transportation|35.00
        "fuel pump" → Fuel|Transportation|40.00
        "car maintenance" → Car maintenance|Transportation|120.00
        "bus ticket" → Bus ticket|Transportation|2.50

        Now categorize this: "{description}"
        Amount to use: {amount} (unless description specifies different)
        """

        response = await openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system",
                 "content": "You are a precise expense categorizer. Use ONLY the specified format and categories."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=100
        )

        content = response.choices[0].message.content.strip()
        if "|" in content:
            parts = content.split("|")
            if len(parts) >= 3:
                title = parts[0].strip()
                category = parts[1].strip().lower()

                # Enhanced category validation with transportation priority
                if 'transport' in category or any(kw in description_lower for kw in transport_keywords):
                    category = 'transportation'

                valid_categories = {
                    "food", "transportation", "entertainment",
                    "shopping", "bills", "services",
                    "health", "travel", "other"
                }

                category = category if category in valid_categories else "other"

                try:
                    parsed_amount = float(parts[2].strip()) if parts[2].strip().replace('.', '',
                                                                                        1).isdigit() else amount
                    if parsed_amount <= 0:
                        raise ValueError("Amount must be positive")
                    amount = parsed_amount
                except ValueError:
                    if amount <= 0:
                        raise ValueError("Amount must be positive")

                return Expense(title=title, category=category, amount=amount)

    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        # Fallback with transport keyword check
        title = description[:amount_match.start()].strip() if amount_match else description
        if any(keyword in description_lower for keyword in transport_keywords):
            return Expense(title=title, category="transportation", amount=amount)
        if amount <= 0:
            raise ValueError("Amount must be positive")
        return Expense(title=title, category="other", amount=amount)