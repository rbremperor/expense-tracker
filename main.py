from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import openai
import os
import re
from dotenv import load_dotenv
import asyncpg
from datetime import datetime

# Load environment variables
load_dotenv()

# FastAPI app
app = FastAPI()

# Database connection pool
db_pool = None

# OpenAI API key setup
openai.api_key = os.getenv("OPENAI_API_KEY")

# Serve static files (index.html)
app.mount("/static", StaticFiles(directory="templates"), name="static")


@app.on_event("startup")
async def startup():
    global db_pool
    db_pool = await asyncpg.create_pool(
        dsn=os.getenv("DATABASE_URL"),
        min_size=1,
        max_size=10
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


@app.on_event("shutdown")
async def shutdown():
    await db_pool.close()


# Expense model
class Expense(BaseModel):
    title: str
    category: str
    amount: float


class ExpenseInput(BaseModel):
    description: str


@app.get("/", response_class=HTMLResponse)
async def get_html():
    with open("templates/index.html", "r") as f:
        return HTMLResponse(content=f.read(), status_code=200)


@app.post("/add_expense/")
async def add_expense(expense_input: ExpenseInput):
    description = expense_input.description
    parsed_data = await parse_expense(description)

    try:
        async with db_pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO expenses(title, category, amount) VALUES($1, $2, $3)",
                parsed_data.title, parsed_data.category, parsed_data.amount
            )
        return {"message": "Expense added successfully", "data": parsed_data}
    except asyncpg.PostgresError as e:
        raise HTTPException(status_code=500, detail="Database error")


async def parse_expense(description: str) -> Expense:
    # First extract amount if it exists
    amount = 0
    amount_match = re.search(r'(\d+\.?\d*)', description)
    if amount_match:
        amount = float(amount_match.group(1))

    # Call OpenAI for all categorization to ensure consistency
    prompt = f"""
    Categorize this expense: "{description}"

    Respond ONLY with this exact format:
    title|category|amount

    Where:
    - title: 1-3 word concise title based on the description
    - category: One of these exact categories:
        Food (anything edible: groceries, restaurants, coffee, fruits)
        Transportation (rides, fuel, public transit, parking)
        Entertainment (movies, games, events, streaming)
        Shopping (physical goods, clothes, electronics)
        Bills (utilities, rent, subscriptions)
        Services (repairs, maintenance, professional services)
        Health (medical, pharmacy, fitness)
        Travel (hotels, flights, vacation expenses)
        Other (anything that doesn't fit above categories)
    - amount: numeric value ({amount} if not specified in the description)
    """

    try:
        response = await openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": "You are an expense categorization assistant. Respond ONLY in the specified format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=50
        )

        content = response.choices[0].message.content.strip()
        if "|" in content:
            parts = content.split("|")
            if len(parts) >= 3:
                title = parts[0].strip()
                category = parts[1].strip()
                try:
                    amount = float(parts[2].strip()) if parts[2].strip().replace('.', '', 1).isdigit() else amount
                except ValueError:
                    pass

                # Validate category
                valid_categories = [
                    "Food", "Transportation", "Entertainment", "Shopping",
                    "Bills", "Services", "Health", "Travel", "Other"
                ]
                category = category if category in valid_categories else "Other"

                return Expense(title=title, category=category, amount=amount)

    except Exception as e:
        print(f"OpenAI API error: {e}")

    # Fallback if API fails
    title = description
    if amount_match:
        title = description[:amount_match.start()].strip()
    return Expense(title=title, category="Other", amount=amount)

@app.get("/get_expenses/")
async def get_expenses():
    try:
        async with db_pool.acquire() as conn:
            expenses = await conn.fetch("SELECT title, category, amount FROM expenses ORDER BY created_at DESC")
        return {"expenses": expenses}
    except asyncpg.PostgresError as e:
        raise HTTPException(status_code=500, detail="Database error")


@app.get("/get_expenses/{category}")
async def get_expenses_by_category(category: str):
    if category.lower() == "all":
        return await get_expenses()

    try:
        async with db_pool.acquire() as conn:
            expenses = await conn.fetch(
                "SELECT title, category, amount FROM expenses WHERE LOWER(category) = $1 ORDER BY created_at DESC",
                category.lower()
            )
        return {"expenses": expenses}
    except asyncpg.PostgresError as e:
        raise HTTPException(status_code=500, detail="Database error")
