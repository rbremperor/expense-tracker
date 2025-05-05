from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import openai
import os
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# FastAPI app
app = FastAPI()

# OpenAI API key setup
openai.api_key = os.getenv("OPENAI_API_KEY")

# Serve static files (index.html)
app.mount("/static", StaticFiles(directory="templates"), name="static")


# Expense model
class Expense(BaseModel):
    title: str
    category: str
    amount: float


# Temporary in-memory database
expenses_db = []


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
    expenses_db.append(parsed_data)
    return {"message": "Expense added successfully", "data": parsed_data}


async def parse_expense(description: str) -> Expense:
    lower_desc = description.lower()
    amount = 0
    title = description

    # Extract amount if it exists
    amount_match = re.search(r'(\d+\.?\d*)', description)
    if amount_match:
        amount = float(amount_match.group(1))
        title = description[:amount_match.start()].strip()

    # Simple category detection
    category = "Other"
    food_keywords = ["lunch", "dinner", "breakfast", "coffee", "banana", "apple", "groceries", "food", "restaurant"]
    transport_keywords = ["uber", "taxi", "train", "bus", "metro", "subway", "gas", "parking"]
    entertainment_keywords = ["movie", "netflix", "concert", "game", "hobby"]
    shopping_keywords = ["amazon", "shirt", "purchase", "buy", "mall"]
    bills_keywords = ["bill", "rent", "electric", "water", "internet", "phone"]

    if any(word in lower_desc for word in food_keywords):
        category = "Food"
    elif any(word in lower_desc for word in transport_keywords):
        category = "Transportation"
    elif any(word in lower_desc for word in entertainment_keywords):
        category = "Entertainment"
    elif any(word in lower_desc for word in shopping_keywords):
        category = "Shopping"
    elif any(word in lower_desc for word in bills_keywords):
        category = "Bills"

    if category != "Other":
        return Expense(title=title, category=category, amount=amount)

    # Call OpenAI for ambiguous cases
    prompt = f"""
    Categorize this expense: "{description}"

    Respond ONLY with this exact format:
    title|category|amount

    Where:
    - title: 1-3 word title
    - category: Food, Transportation, Entertainment, Shopping, Bills, or Other
    - amount: numeric value (0 if not specified)
    """

    try:
        response = openai.ChatCompletion.create(
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
            title, category, amount_str = content.split("|")
            try:
                amount = float(amount_str) if amount_str.replace('.', '', 1).isdigit() else amount
            except ValueError:
                pass

            valid_categories = ["Food", "Transportation", "Entertainment", "Shopping", "Bills", "Other"]
            if category not in valid_categories:
                category = "Other"

            return Expense(title=title, category=category, amount=amount)

    except Exception as e:
        print(f"OpenAI API error: {e}")

    return Expense(title=title, category=category, amount=amount)


@app.get("/get_expenses/")
async def get_expenses():
    return {"expenses": expenses_db}


@app.get("/get_expenses/{category}")
async def get_expenses_by_category(category: str):
    if category.lower() == "all":
        return {"expenses": expenses_db}
    filtered_expenses = [e for e in expenses_db if e.category.lower() == category.lower()]
    return {"expenses": filtered_expenses}