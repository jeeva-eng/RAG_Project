import os
import pandas as pd
from langchain_groq import ChatGroq


class CSVAgent:

    def __init__(self, csv_path: str):
        # Load CSV
        self.df = pd.read_csv(csv_path)

        # Initialize LLM
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.1-8b-instant",
            temperature=0
        )

    # ----------------------------
    # QUERY METHOD
    # ----------------------------
    def query(self, question: str):
        try:
            q = question.lower()

            # Detect zip-related queries
            if any(k in q for k in ["zip", "zipcode", "customer_zip_code_prefix"]):
                city = self.extract_city(q)

                if city:
                    df = self.df[
                        self.df["customer_city"].str.lower() == city
                    ]

                    if df.empty:
                        return f"No data found for city: {city}"

                    df = df[["customer_zip_code_prefix"]].drop_duplicates()
                    return df.to_string(index=False)

            # Fallback if unknown
            return "Query not supported yet. Try asking about zip codes or cities."

        except Exception as e:
            return f"CSV query failed: {e}"

    # ----------------------------
    # EXTRACT CITY HELPER
    # ----------------------------
    def extract_city(self, question: str):
        words = [w.lower().strip(",.?") for w in question.split()]

        ignore = [
            "zip", "zipcode", "zipcodes",
            "city", "for", "the", "give", "me",
            "list", "customer_zip_code_prefix"
        ]

        for w in words:
            if w not in ignore:
                return w

        return None

    # ----------------------------
    # GENERATE FOLLOW-UP SUGGESTIONS
    # ----------------------------
    def generate_suggestions(self, question: str, result: str):
        try:
            prompt = f"""
You are a data assistant.

User question:
{question}

Result:
{result}

Generate 4 smart follow-up questions.

Rules:
- Very short
- Actionable
- Based on the data
- No explanation
"""

            response = self.llm.invoke(prompt)
            text = response.content.strip()

            # Convert to list
            suggestions = [
                s.strip("- ").strip()
                for s in text.split("\n")
                if s.strip()
            ]

            return suggestions[:4]

        except Exception:
            # Fallback suggestions
            return [
                "Show top 10 rows",
                "Count total records",
                "Group by column",
                "Sort descending"
            ]