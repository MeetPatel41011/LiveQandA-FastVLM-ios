import re
import numpy as np

# ─────────────────────────────────────────────
# Tool 1: Scalar Calculator
# ─────────────────────────────────────────────
def execute_calculator(equation: str) -> str:
    """Safe scalar calculator. e.g. "45 * 12", "100 / 4 + 3" """
    math_match = re.search(r'[\d\s\+\-\*\/\.\(\)]+', equation)
    if not math_match:
        return f"Error: No math expression found in '{equation}'"
    safe_equation = math_match.group(0).strip()
    if not safe_equation:
        return f"Error: Empty expression from '{equation}'"
    try:
        result = eval(safe_equation)
        return str(result)
    except Exception as e:
        return f"Error evaluating '{safe_equation}': {e}"


# ─────────────────────────────────────────────
# Tool 2: NumPy Matrix Multiplier
# ─────────────────────────────────────────────
def _parse_complex(s: str) -> complex:
    """Parse complex number strings like 1+i, 4-i, 2, i, 3."""
    s = s.strip().replace(' ', '')
    s = re.sub(r'(?<![0-9])i', '1j', s)
    s = re.sub(r'([0-9])i', r'\1j', s)
    try:
        return complex(eval(s))
    except Exception:
        try:
            return complex(s)
        except Exception:
            return complex(0)


def execute_matrix_multiply(problem_text: str) -> str:
    """
    Parses a matrix multiplication problem and computes C = A @ B using NumPy.
    Supports complex numbers using 'i' notation (1+i, 4-i, etc).
    """
    try:
        all_nums = re.findall(
            r'(?<![a-zA-Z])(?:[+\-]?\s*\d+(?:\.\d+)?\s*[+\-]\s*\d*\s*i|[+\-]?\s*\d+(?:\.\d+)?|[+\-]?\s*\d*\s*i)(?![a-zA-Z])',
            problem_text
        )
        filtered = [n for n in all_nums if n.strip() and n.strip() not in ('+', '-')]
        parsed = [_parse_complex(n) for n in filtered]
        if len(parsed) < 4:
            return f"Error: Could not extract enough numbers. Found: {parsed}"
        half = len(parsed) // 2
        a_nums = parsed[:half]
        b_nums = parsed[half:]
        size = int(len(a_nums) ** 0.5)
        if size * size != len(a_nums):
            return f"Error: Matrix is not square (found {len(a_nums)} elements per matrix)"
        A = np.array(a_nums, dtype=complex).reshape(size, size)
        B = np.array(b_nums, dtype=complex).reshape(size, size)
        C = A @ B
        def fmt(v):
            v = complex(v)
            r = round(v.real, 4)
            im = round(v.imag, 4)
            if im == 0:
                return str(int(r) if r == int(r) else r)
            sign = '+' if im >= 0 else '-'
            return f"{int(r) if r == int(r) else r} {sign} {abs(int(im) if im == int(im) else im)}i"
        rows = []
        for row in C:
            rows.append('[' + ',  '.join(fmt(v) for v in row) + ']')
        return 'C =\n  ' + '\n  '.join(rows)
    except Exception as e:
        return f"Error computing matrix multiplication: {e}"


# ─────────────────────────────────────────────
# Tool 3: Web Search (Powered by Tavily AI)
# ─────────────────────────────────────────────
def execute_web_search(query: str) -> str:
    """
    Perform a high-precision web search using Tavily AI.
    Replaces Wikipedia and DuckDuckGo for better accuracy.
    """
    import os
    import requests
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Error: TAVILY_API_KEY not found in environment."

    # Clean the query (remove any JSON artifacts if the LLM leaked them)
    if '{' in query:
        # Try to extract content between quotes if it looks like JSON
        match = re.search(r'"(?:text_in_image|query)":\s*"([^"]+)"', query)
        if match:
            query = match.group(1)
        else:
            # Fallback: just remove curly braces and extra quotes
            query = query.replace('{', '').replace('}', '').replace('"', '').strip()
    
    query = query.strip()
    if not query:
        return "Error: Cleaned search query is empty."

    print(f"DEBUG: Sending to Tavily -> '{query}'")
    try:
        # We use the raw requests approach to avoid dependency issues in Colab
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": api_key,
            "query": query,
            "search_depth": "smart",
            "include_answer": True,
            "max_results": 1
        }
        response = requests.post(url, json=payload, timeout=15)
        
        if response.status_code != 200:
            return f"Tavily API Error ({response.status_code}): {response.text}"
            
        data = response.json()
        
        # 1. Return the AI-generated answer if available (best)
        if data.get("answer"):
            return f"\U0001f310 Tavily AI: {data['answer']}"
            
        # 2. Otherwise return the top search result
        results = data.get("results", [])
        if results:
            content = results[0].get('content', '')
            return f"\U0001f310 Tavily Search: {content}"
            
        return f"No relevant information found for '{query}' on the web."
    except Exception as e:
        return f"Tavily Search Error: {str(e)}"


# ─────────────────────────────────────────────
# Tool Registry — Add new tools here
# ─────────────────────────────────────────────
AVAILABLE_TOOLS = {
    "calculator": execute_calculator,
    "matrix":     execute_matrix_multiply,
    "web_search": execute_web_search,
}