import os
import re
import pandas as pd
from langchain_groq import ChatGroq


class CSVAgent:

    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.1-8b-instant",
            temperature=0,
        )

    # ──────────────────────────────────────────────────
    # MAIN QUERY METHOD
    # ──────────────────────────────────────────────────
    def query(self, question: str) -> str:
        try:
            q = question.lower().strip()

            # ── 1. DUPLICATE queries ───────────────────
            if any(k in q for k in ["duplicate", "duplicates", "repeated"]):
                return self._handle_duplicates(q)

            # ── 2. UNIQUE / DISTINCT queries ──────────
            if any(k in q for k in ["unique", "distinct", "how many different"]):
                return self._handle_unique(q)

            # ── 3. MISSING / NULL queries ─────────────
            if any(k in q for k in ["missing", "null", "empty", "nan"]):
                return self._handle_missing(q)

            # ── 4. MOST / LEAST COMMON queries ────────
            if any(k in q for k in ["most common", "least common", "top", "frequent", "popular"]):
                return self._handle_frequency(q)

            # ── 5. COUNT queries ───────────────────────
            if any(k in q for k in ["count", "how many", "total number", "number of"]):
                return self._handle_count(q)

            # ── 6. AVERAGE / SUM queries ───────────────
            if any(k in q for k in ["average", "avg", "mean", "sum", "total"]):
                return self._handle_aggregation(q)

            # ── 7. ZIP / LIST queries ──────────────────
            if any(k in q for k in ["zip", "zipcode", "customer_zip_code_prefix", "list", "show"]):
                return self._handle_list(q)

            # ── 8. FALLBACK ────────────────────────────
            return self._llm_fallback(question)

        except Exception as e:
            return f"CSV query failed: {e}"

    # ──────────────────────────────────────────────────
    # HANDLER: DUPLICATES
    # ──────────────────────────────────────────────────
    def _handle_duplicates(self, q: str) -> str:
        col  = self._detect_column(q)
        city = self._extract_city(q)

        # Apply city filter if present
        df = self.df
        city_label = ""
        if city:
            df = df[df["customer_city"].str.lower() == city]
            if df.empty:
                return f"No data found for city: {city.title()}"
            city_label = f" — {city.title()}"

        if col:
            WIDE = "━" * 46
            THIN = "·" * 46

            total_rows  = len(df)
            dupe_count  = df[col].duplicated().sum()
            unique_vals = df[col].nunique()

            if dupe_count == 0:
                return (
                    f"{WIDE}\n"
                    f"  🔁 Duplicates in '{col}'{city_label}\n"
                    f"{THIN}\n"
                    f"  ✅ No duplicate values found\n"
                    f"  Total rows   : {total_rows}\n"
                    f"  Unique values: {unique_vals}\n"
                    f"{WIDE}"
                )

            # Show which values are duplicated and how many times
            top = (
                df[col]
                .value_counts()
                .where(lambda x: x > 1)
                .dropna()
                .astype(int)
                .head(15)
            )
            lines = [f"  {str(v):<12} → appears {c:>4} times" for v, c in top.items()]

            return (
                f"{WIDE}\n"
                f"  🔁 Duplicates in '{col}'{city_label}\n"
                f"{THIN}\n"
                f"  Total rows        : {total_rows}\n"
                f"  Duplicate rows    : {dupe_count}\n"
                f"  Unique values     : {unique_vals}\n"
                f"{THIN}\n"
                f"  Duplicated values:\n"
                + "\n".join(lines) + "\n"
                + f"{WIDE}"
            )

        # No column specified → check whole dataframe
        total_dupes = df.duplicated().sum()
        if total_dupes == 0:
            return "✅ No duplicate rows found in the dataset."
        return (
            f"🔁 Duplicate Rows Found\n"
            f"{'─' * 44}\n"
            f"  Total duplicate rows: {total_dupes}\n"
            f"  Total rows          : {len(self.df)}\n"
            f"  Unique rows         : {len(self.df) - total_dupes}"
        )

    # ──────────────────────────────────────────────────
    # HANDLER: UNIQUE / DISTINCT
    # ──────────────────────────────────────────────────
    def _handle_unique(self, q: str) -> str:
        col = self._detect_column(q)
        city = self._extract_city(q)

        if col:
            df = self.df
            if city:
                df = df[df["customer_city"].str.lower() == city]
                if df.empty:
                    return f"No data found for city: {city}"

            unique_count = df[col].nunique()
            sample = df[col].dropna().unique()[:10]
            sample_str = "  " + "\n  ".join(str(v) for v in sample)

            return (
                f"🔢 Unique values in '{col}'"
                + (f" — {city.title()}" if city else "") + "\n"
                f"{'─' * 44}\n"
                f"  Total unique values: {unique_count}\n"
                f"{'─' * 44}\n"
                f"  Sample:\n{sample_str}"
            )

        # No column — show unique counts for all columns
        lines = [f"  {c:<35} {self.df[c].nunique():>6} unique" for c in self.df.columns]
        return (
            f"🔢 Unique Value Counts — All Columns\n"
            f"{'─' * 44}\n"
            + "\n".join(lines)
        )

    # ──────────────────────────────────────────────────
    # HANDLER: MISSING / NULL
    # ──────────────────────────────────────────────────
    def _handle_missing(self, q: str) -> str:
        col = self._detect_column(q)

        if col:
            missing = self.df[col].isna().sum()
            pct = (missing / len(self.df)) * 100
            if missing == 0:
                return f"✅ No missing values in '{col}'."
            return (
                f"⚠️  Missing Values in '{col}'\n"
                f"{'─' * 44}\n"
                f"  Missing : {missing} rows\n"
                f"  Total   : {len(self.df)} rows\n"
                f"  Percent : {pct:.1f}%"
            )

        # All columns
        missing_info = self.df.isnull().sum()
        missing_info = missing_info[missing_info > 0]
        if missing_info.empty:
            return "✅ No missing values found in any column."

        lines = [
            f"  {c:<35} {v:>6} missing ({(v/len(self.df)*100):.1f}%)"
            for c, v in missing_info.items()
        ]
        return (
            f"⚠️  Missing Values Summary\n"
            f"{'─' * 44}\n"
            + "\n".join(lines)
        )

    # ──────────────────────────────────────────────────
    # HANDLER: MOST / LEAST COMMON
    # ──────────────────────────────────────────────────
    def _handle_frequency(self, q: str) -> str:
        col = self._detect_column(q)
        n = self._extract_number(q) or 10
        least = "least" in q

        if not col:
            col = "customer_city"  # sensible default

        counts = self.df[col].value_counts()
        if least:
            counts = counts.tail(n)
            label = f"🔻 Least Common — '{col}'"
        else:
            counts = counts.head(n)
            label = f"🔝 Most Common — '{col}'"

        lines = [f"  {str(v):<30} {c:>6}" for v, c in counts.items()]
        return (
            f"{label}\n"
            f"{'─' * 44}\n"
            + "\n".join(lines)
        )

    # ──────────────────────────────────────────────────
    # HANDLER: COUNT
    # ──────────────────────────────────────────────────
    def _handle_count(self, q: str) -> str:
        city = self._extract_city(q)
        col  = self._detect_column(q)

        if city:
            count = len(self.df[self.df["customer_city"].str.lower() == city])
            return (
                f"🔢 Customer Count — {city.title()}\n"
                f"{'─' * 44}\n"
                f"  Total customers: {count}"
            )

        if col:
            return (
                f"🔢 Count — '{col}'\n"
                f"{'─' * 44}\n"
                f"  Non-null rows : {self.df[col].count()}\n"
                f"  Total rows    : {len(self.df)}"
            )

        return (
            f"🔢 Dataset Size\n"
            f"{'─' * 44}\n"
            f"  Total rows    : {len(self.df)}\n"
            f"  Total columns : {len(self.df.columns)}\n"
            f"  Columns: {', '.join(self.df.columns.tolist())}"
        )

    # ──────────────────────────────────────────────────
    # HANDLER: AGGREGATION (avg, sum)
    # ──────────────────────────────────────────────────
    def _handle_aggregation(self, q: str) -> str:
        col = self._detect_column(q)
        numeric_cols = self.df.select_dtypes(include="number").columns.tolist()

        if col and col in numeric_cols:
            if any(k in q for k in ["average", "avg", "mean"]):
                val = self.df[col].mean()
                return f"📊 Average of '{col}': {val:.2f}"
            elif any(k in q for k in ["sum", "total"]):
                val = self.df[col].sum()
                return f"📊 Sum of '{col}': {val:,.2f}"

        if numeric_cols:
            lines = [
                f"  {c:<35} avg={self.df[c].mean():.2f}  sum={self.df[c].sum():,.0f}"
                for c in numeric_cols
            ]
            return (
                f"📊 Numeric Summary\n"
                f"{'─' * 44}\n"
                + "\n".join(lines)
            )

        return "No numeric columns found for aggregation."

    # ──────────────────────────────────────────────────
    # HANDLER: LIST / ZIP
    # ──────────────────────────────────────────────────
    def _handle_list(self, q: str) -> str:
        col  = self._detect_column(q)
        city = self._extract_city(q)

        # default column if none detected
        if not col:
            col = "customer_zip_code_prefix"

        df = self.df
        if city:
            df = df[df["customer_city"].str.lower() == city]
            if df.empty:
                return f"No data found for city: {city}"

        if col not in df.columns:
            return f"Column '{col}' not found in dataset."

        values = df[col].drop_duplicates().dropna().tolist()
        return " ".join(str(v) for v in values)  # let format_answer() handle grid

    # ──────────────────────────────────────────────────
    # LLM FALLBACK
    # ──────────────────────────────────────────────────
    def _llm_fallback(self, question: str) -> str:
        schema = ", ".join(self.df.columns.tolist())
        sample = self.df.head(3).to_string(index=False)
        prompt = f"""You are a data assistant. The user has a CSV dataset with columns: {schema}

Sample rows:
{sample}

Answer this question about the data clearly in 3-5 lines:
{question}"""
        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception:
            return "Query not supported. Try: duplicates, unique count, missing values, most common, count, list by city."

    # ──────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────
    def _detect_column(self, q: str) -> str:
        """Return the column name mentioned in the query, or empty string."""
        q_clean = q.lower()
        # check full column names first (longest match wins)
        for col in sorted(self.df.columns, key=len, reverse=True):
            if col.lower() in q_clean:
                return col
        # check partial word matches
        col_map = {
            "zip":      "customer_zip_code_prefix",
            "zipcode":  "customer_zip_code_prefix",
            "prefix":   "customer_zip_code_prefix",
            "city":     "customer_city",
            "state":    "customer_state",
            "id":       "customer_id",
            "unique_id":"customer_unique_id",
        }
        for keyword, col in col_map.items():
            if keyword in q_clean and col in self.df.columns:
                return col
        return ""

    def _extract_city(self, q: str) -> str:
        """
        Extract city name from query.
        Works with AND without prepositions:
          'list zip for osasco'              → 'osasco'
          'duplicates in zip osasco'         → 'osasco'
          'are there duplicates osasco'      → 'osasco'
        Strategy: check every word in the query against
        actual city names in the dataset.
        """
        cities = set(self.df["customer_city"].str.lower().unique())

        stopwords = {
            "me", "the", "a", "an", "all", "my", "this", "that",
            "any", "there", "each", "every", "some", "most",
            "list", "customers", "duplicates", "duplicate", "records",
            "values", "prefix", "count", "rows", "city", "zip", "state",
            "column", "table", "data", "results", "unique", "for", "in",
            "of", "from", "are", "is", "show", "give", "find", "get",
            "customer", "zip_code", "customer_zip_code_prefix",
        }

        # First try: word after preposition (most specific)
        match = re.search(r'\b(?:for|in|of|from)\s+(\w+)', q)
        if match:
            word = match.group(1).strip("?.,").lower()
            if word not in stopwords and word in cities:
                return word

        # Second try: any word in query that is a known city
        words = re.findall(r'[a-záéíóúãõâêîôûç]+', q)
        for word in words:
            if word not in stopwords and len(word) > 2 and word in cities:
                return word

        return ""

    def _extract_number(self, q: str) -> int:
        """Extract a number from query like 'top 5' or 'top ten'."""
        word_nums = {
            "one":1,"two":2,"three":3,"four":4,"five":5,
            "six":6,"seven":7,"eight":8,"nine":9,"ten":10
        }
        for w, n in word_nums.items():
            if w in q:
                return n
        match = re.search(r'\b(\d+)\b', q)
        return int(match.group(1)) if match else None

    # ──────────────────────────────────────────────────
    # SUGGESTIONS (kept for compatibility)
    # ──────────────────────────────────────────────────
    def generate_suggestions(self, question: str, result: str):
        try:
            prompt = f"""You are a data assistant.
User question: {question}
Result: {result[:300]}
Generate 4 smart follow-up questions. Very short. No explanation. Numbered 1-4."""
            response = self.llm.invoke(prompt)
            lines = [
                l.lstrip("0123456789.- ").strip()
                for l in response.content.strip().split("\n")
                if l.strip()
            ]
            return [l for l in lines if len(l) > 3][:4]
        except Exception:
            return ["Show top 10 rows", "Count total records", "Group by column", "Sort descending"]