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
# Tool 3: Web Search (Multi-Source: Wikipedia + Tavily + DDG)
# ─────────────────────────────────────────────
def execute_web_search(query: str) -> str:
    """
    Advanced multi-source search.
    1. Wikipedia (Definitions)
    2. Tavily (High-quality AI Search)
    3. DuckDuckGo (Fallback/News)
    """
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    # Clean the query
    if '{' in query:
        query = re.sub(r'\{.*?\}', '', query).strip()
    
    is_definition = any(word in query.lower() for word in ["what is", "who is", "define", "meaning of"])
    clean_query = re.sub(r'^(what is|who is|where is|when was|tell me about|search for|how much is|current price of|define)\s+', '', query, flags=re.IGNORECASE)
    clean_query = clean_query.strip().rstrip('?').strip()
    
    if not clean_query:
        return "Error: Empty search query."

    # --- Step 1: Wikipedia Check (Zero-cost, High-quality definitions) ---
    if is_definition:
        try:
            import requests
            wiki_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{clean_query.replace(' ', '_')}"
            response = requests.get(wiki_url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if 'extract' in data:
                    return f"\U0001f4da Wikipedia: {data['extract']}"
        except Exception:
            pass

    # --- Step 2: Tavily Search (Best AI Results) ---
    tavily_key = os.getenv("TAVILY_API_KEY")
    if tavily_key:
        try:
            from tavily import TavilyClient
            tavily = TavilyClient(api_key=tavily_key)
            # Use 'search' for a comprehensive answer
            response = tavily.search(query=query, search_depth="basic", max_results=1)
            if response and response['results']:
                result = response['results'][0]
                content = result.get('content', '')
                return f"\U0001f310 Tavily Search: {content}"
        except Exception as e:
            print(f"DEBUG: Tavily error: {e}")

    # --- Step 3: DuckDuckGo Fallback ---
    try:
        import warnings
        from duckduckgo_search import DDGS
        warnings.filterwarnings("ignore", message="This package .* has been renamed to `ddgs`!")
        
        with DDGS() as ddgs:
            results = list(ddgs.text(clean_query, max_results=1))
            if results:
                return f"\U0001f50e Web Fallback: {results[0].get('body', '')}"
    except Exception:
        pass

    return f"Sorry, I couldn't find a specific answer for '{clean_query}'."


# ─────────────────────────────────────────────
# Tool Registry — Add new tools here
# ─────────────────────────────────────────────
AVAILABLE_TOOLS = {
    "calculator": execute_calculator,
    "matrix":     execute_matrix_multiply,
    "web_search": execute_web_search,
}