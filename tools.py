# tools.py
from datetime import datetime

def get_date():
    return datetime.now().strftime("%A, %d %B %Y")

def calculator(expression):
    allowed = "0123456789+-*/(). "
    if not all(ch in allowed for ch in expression):
        return "Invalid expression"
    return str(eval(expression))