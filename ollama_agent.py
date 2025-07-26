import pandas as pd
import requests
import json
import os

# === 1. Load the CSV file ===
csv_path = input("Enter path to your CSV file (e.g., sample.csv): ").strip()
if not os.path.exists(csv_path):
    print("File not found!")
    exit()

df = pd.read_csv(csv_path)
print("\nCSV loaded successfully.")
print("Columns:", list(df.columns))
print("Rows:", len(df))
print("Preview:\n", df.head())

# === 2. Ask user for a question ===
question = input("\nAsk a question about the data:\n> ").strip()
if not question:
    print("Question cannot be empty.")
    exit()

# === 3. Build prompt to generate pandas code ===
df_preview = df.head(5).to_string(index=False)

prompt = f"""
You are a skilled Python data analyst.
Here is a preview of a pandas DataFrame:

{df_preview}

Based on this sample, write ONLY the pandas code (no explanations) that answers the following question:
\"\"\"{question}\"\"\"

Use the variable name `df`.
Only output code, no markdown, no comments, no explanation, and no quotation marks.
"""

# === 4. Call Ollama API (via HTTP POST) ===
def ask_ollama_code(prompt):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3", "prompt": prompt},
            stream=True
        )
        result = ""
        print("\nStreaming response from Ollama:")
        for line in response.iter_lines():
            if line:
                decoded = line.decode("utf-8")
                try:
                    data = json.loads(decoded)
                    token = data.get("response", "")
                    print("<-", token, end="", flush=True)
                    result += token
                except json.JSONDecodeError:
                    pass
        return result.strip()
    except Exception as e:
        return f"Error: {e}"

# === 5. Ask Ollama to generate pandas code ===
print("\n\nPrompt sent to model")

pandas_code = ask_ollama_code(prompt)
print("\n\nSuggested pandas code:\n", pandas_code)

# === 6. Try to execute generated code ===
print("\nExecuting code...")
try:
    local_vars = {'df': df.copy()}
    result = eval(pandas_code, {}, local_vars)

    print("\nExecution result:\n", result)

# === 7. Send result back to LLM for explanation ===
    explanation_prompt = f"""
    You are a helpful AI assistant.
    The user asked: \"{question}\"
    Here is the output from pandas:
    {result}

    Please summarize the result in 1–2 clear English sentences.
    """
    print("\nSending result to LLM for natural language explanation...")

    def ask_ollama_explanation(prompt):
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "llama3", "prompt": prompt},
                stream=True
            )
            result = ""
            for line in response.iter_lines():
                if line:
                    decoded = line.decode("utf-8")
                    try:
                        data = json.loads(decoded)
                        result += data.get("response", "")
                    except json.JSONDecodeError:
                        pass
            return result.strip()
        except Exception as e:
            return f"Error: {e}"

    answer = ask_ollama_explanation(explanation_prompt)
    print("\nNatural language answer:")
    print(answer)

except Exception as e:
    print("\nError executing code:", str(e))
