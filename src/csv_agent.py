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

        df = self.df
        city_label = city.title() if city else None
        if city:
            df = df[df["customer_city"].str.lower() == city]
            if df.empty:
                return f"<p>No data found for city <strong>{city.title()}</strong>.</p>"

        if col:
            total_rows  = len(df)
            dupe_count  = int(df[col].duplicated().sum())
            unique_vals = df[col].nunique()
            city_phrase = f" in <strong>{city_label}</strong>" if city_label else ""

            if dupe_count == 0:
                return f"<p>✅ Great news! There are <strong>no duplicate values</strong> in the column <strong>{col}</strong>{city_phrase}. All <strong>{unique_vals}</strong> values are unique across <strong>{total_rows}</strong> rows.</p>"

            top = (
                df[col].value_counts()
                .where(lambda x: x > 1).dropna().astype(int).head(15)
            )
            dupe_rows = "".join(
                f"<tr><td>{v}</td><td>{c}</td><td>{c-1} extra</td></tr>"
                for v, c in top.items()
            )
            pct = round(dupe_count / total_rows * 100, 1)
            return f"""<!--HTML-->
<p>Here is the duplicate analysis for <strong>{col}</strong>{city_phrase}.</p>
<p>Out of <strong>{total_rows} total rows</strong>, there are <strong>{dupe_count} duplicate entries</strong> ({pct}% of the data). These duplicates are spread across <strong>{len(top)} distinct values</strong>. Removing them would leave <strong>{unique_vals} unique values</strong>.</p>
<p><strong>🔁 Top Duplicated Values</strong></p>
<table>
  <thead><tr><th>{col}</th><th>Count</th><th>Duplicates</th></tr></thead>
  <tbody>{dupe_rows}</tbody>
</table>
<p>Consider deduplicating this column if you need unique records for analysis or reporting.</p>"""

        total_dupes = int(self.df.duplicated().sum())
        if total_dupes == 0:
            return "<p>✅ The dataset has <strong>no duplicate rows</strong>. All records are unique.</p>"
        return f"<p>The dataset contains <strong>{total_dupes} fully duplicate rows</strong> out of <strong>{len(self.df)}</strong> total rows ({round(total_dupes/len(self.df)*100,1)}%). Consider removing them for cleaner analysis.</p>"
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
        col   = self._detect_column(q)
        n     = self._extract_number(q) or 5
        least = "least" in q

        if not col:
            col = "customer_city"

        counts = self.df[col].value_counts()

        if least:
            top    = counts.tail(n)
            intro  = f"The **{n} least common** values in **{col}** are:"
        else:
            top    = counts.head(n)
            intro  = f"The **{n} most common** values in **{col}** are:"

        parts = [f"**{v}** ({c} times)" for v, c in top.items()]
        result = ", ".join(parts[:-1]) + f", and {parts[-1]}" if len(parts) > 1 else parts[0]

        total_unique = self.df[col].nunique()
        return (
            f"{intro} {result}. "
            f"In total there are **{total_unique} unique values** in this column across the entire dataset."
        )

    # ──────────────────────────────────────────────────
    # HANDLER: COUNT
    # ──────────────────────────────────────────────────
    def _handle_count(self, q: str) -> str:
        city = self._extract_city(q)
        col  = self._detect_column(q)

        if city:
            df    = self.df[self.df["customer_city"].str.lower() == city]
            count = len(df)
            pct   = round(count / len(self.df) * 100, 2)
            state = df["customer_state"].value_counts().idxmax() if "customer_state" in df.columns and not df.empty else ""
            state_txt = f" in the state of <strong>{state}</strong>" if state else ""
            return f"""<!--HTML-->
<p>There are <strong>{count} customers</strong> registered in <strong>{city.title()}</strong>{state_txt}.</p>
<table>
  <thead><tr><th>City</th><th>Total Customers</th><th>% of Dataset</th></tr></thead>
  <tbody><tr><td>{city.title()}</td><td>{count}</td><td>{pct}%</td></tr></tbody>
</table>
<p>This means <strong>{city.title()}</strong> represents <strong>{pct}%</strong> of all <strong>{len(self.df)}</strong> customers in the dataset.</p>"""

        if col:
            non_null = int(self.df[col].count())
            total    = len(self.df)
            missing  = total - non_null
            miss_txt = f" There are also <strong>{missing} missing values</strong> that may need attention." if missing > 0 else " There are <strong>no missing values</strong> in this column."
            return f"<p>The column <strong>{col}</strong> has <strong>{non_null} non-null records</strong> out of <strong>{total} total rows</strong>.{miss_txt}</p>"

        total = len(self.df)
        cols  = self.df.columns.tolist()
        col_rows = "".join(
            f"<tr><td>{c}</td><td>{self.df[c].dtype}</td><td>{self.df[c].nunique()}</td><td>{self.df[c].isnull().sum()}</td></tr>"
            for c in cols
        )
        return f"""<!--HTML-->
<p>The dataset contains <strong>{total} rows</strong> and <strong>{len(cols)} columns</strong>. Here is a full summary of each column:</p>
<table>
  <thead><tr><th>Column</th><th>Type</th><th>Unique Values</th><th>Missing</th></tr></thead>
  <tbody>{col_rows}</tbody>
</table>"""
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

        if not col:
            col = "customer_zip_code_prefix"

        df = self.df
        city_label = city.title() if city else "the dataset"

        if city:
            df = df[df["customer_city"].str.lower() == city]
            if df.empty:
                return f"<p>I could not find any records for <strong>{city.title()}</strong>. Please check the city name and try again.</p>"

        if col not in df.columns:
            return f"<p>The column <strong>{col}</strong> was not found in the dataset.</p>"

        # ── stats ──────────────────────────────────────────────
        unique_zips  = sorted(df[col].drop_duplicates().dropna().tolist())
        total_cust   = len(df)
        total_ds     = len(self.df)
        pct_of_ds    = round(total_cust / total_ds * 100, 2)
        avg_per_zip  = round(total_cust / len(unique_zips), 1) if unique_zips else 0
        vc           = df[col].value_counts()
        top_zip      = vc.idxmax()
        top_count    = int(vc.max())
        low_zip      = vc.idxmin()
        low_count    = int(vc.min())
        median_count = round(float(vc.median()), 1)
        above_avg    = int((vc >= avg_per_zip).sum())

        top5_text = ", ".join(
            f"<strong>{z}</strong> ({c} customers)" for z, c in vc.head(5).items()
        )

        state_line = ""
        if "customer_state" in df.columns and not df.empty:
            states = df["customer_state"].value_counts()
            if len(states) == 1:
                state_line = f"All {total_cust} customers are registered under the state of <strong>{states.index[0]}</strong>."
            else:
                parts = [f"<strong>{s}</strong> ({c} customers, {round(c/total_cust*100,1)}%)" for s, c in states.items()]
                state_line = "Customers are spread across: " + ", ".join(parts) + "."

        # ── PARAGRAPH 1: city overview ─────────────────────────
        p1 = f"""<p>
Here is a complete analysis of ZIP code data for <strong>{city_label}</strong>.
The city has <strong>{total_cust} registered customers</strong>, which accounts for
<strong>{pct_of_ds}%</strong> of the entire dataset containing {total_ds:,} records.
These customers are distributed across <strong>{len(unique_zips)} unique ZIP code prefixes</strong>,
with an average of <strong>{avg_per_zip} customers per ZIP code</strong>.
{state_line}
</p>"""

        # ── PARAGRAPH 2: distribution analysis ────────────────
        p2 = f"""<p>
The ZIP code distribution in <strong>{city_label}</strong> is <em>not uniform</em>.
The most densely populated zone is ZIP <strong>{top_zip}</strong>,
which alone accounts for <strong>{top_count} customers</strong>
({round(top_count/total_cust*100,1)}% of the city total).
At the other extreme, ZIP <strong>{low_zip}</strong> has only
<strong>{low_count} customer(s)</strong>, showing that some postal zones
are sparsely covered. The median customer count per ZIP is
<strong>{median_count}</strong>, and <strong>{above_avg} out of {len(unique_zips)}</strong>
ZIP codes have at or above the average number of customers.
The top 5 most active zones are: {top5_text}.
</p>"""

        # ── TABLE 1: Top 10 with rank, count, share, bar ───────
        top10 = (
            df.groupby(col).size()
            .reset_index(name="Customers")
            .sort_values("Customers", ascending=False)
            .head(10)
            .reset_index(drop=True)
        )
        top10_rows = ""
        for i, row in top10.iterrows():
            rank    = i + 1
            z       = row[col]
            c       = int(row["Customers"])
            share   = round(c / total_cust * 100, 1)
            bar     = "&#9608;" * min(c, 10)
            medal   = ["🥇","🥈","🥉"][i] if i < 3 else f"#{rank}"
            top10_rows += f"<tr><td>{medal}</td><td><strong>{z}</strong></td><td>{c}</td><td>{share}%</td><td style='color:#4fffb0;letter-spacing:2px'>{bar}</td></tr>"

        table1_intro = f"""<p>
The table below ranks the <strong>Top 10 ZIP codes</strong> in <strong>{city_label}</strong>
by customer count. Together, these 10 ZIP codes account for
<strong>{sum(top10['Customers'])} customers</strong>
({round(sum(top10['Customers'])/total_cust*100,1)}% of the city total),
highlighting where the majority of the population is concentrated.
</p>"""

        table1 = f"""<table>
  <thead>
    <tr><th>Rank</th><th>ZIP Code</th><th>Customers</th><th>City Share</th><th>Activity</th></tr>
  </thead>
  <tbody>{top10_rows}</tbody>
</table>"""

        table1_after = f"""<p>
As seen above, the top 3 ZIP codes (<strong>{top10.iloc[0][col]}</strong>,
<strong>{top10.iloc[1][col]}</strong>, <strong>{top10.iloc[2][col]}</strong>)
together hold <strong>{sum(top10.head(3)['Customers'])} customers</strong>,
representing <strong>{round(sum(top10.head(3)['Customers'])/total_cust*100,1)}%</strong>
of all customers in <strong>{city_label}</strong>.
This concentration suggests these zones are likely commercial or
high-density residential areas.
</p>"""

        # ── PARAGRAPH before zip grid ──────────────────────────
        p3 = f"""<p>
The table below lists all <strong>{len(unique_zips)} unique ZIP code prefixes</strong>
active in <strong>{city_label}</strong>.
Each cell represents one postal zone with at least one registered customer.
ZIP codes with fewer customers may represent rural outskirts or newly
developed areas with lower population density.
</p>"""

        # ── TABLE 2: ZIP grid 6 per row ────────────────────────
        zip_trs = ""
        for i in range(0, len(unique_zips), 6):
            chunk = unique_zips[i:i+6]
            # pad row to always have 6 cells
            while len(chunk) < 6:
                chunk.append("")
            zip_trs += "<tr>" + "".join(
                f"<td>{z}</td>" if z else "<td style='background:transparent;border:none;'></td>"
                for z in chunk
            ) + "</tr>"

        table2 = f"""<table class="zip-grid">
  <tbody>{zip_trs}</tbody>
</table>"""

        # ── PARAGRAPH 4: closing summary ──────────────────────
        p4 = f"""<p>
In summary, <strong>{city_label}</strong> is a city with
<strong>{total_cust} customers</strong> distributed across
<strong>{len(unique_zips)} ZIP code prefixes</strong>.
The customer base is moderately concentrated, with the top 10 ZIP codes
covering <strong>{round(sum(top10['Customers'])/total_cust*100,1)}%</strong>
of the population and the remaining
<strong>{len(unique_zips)-10} ZIP codes</strong> sharing the rest.
This data is useful for optimising delivery routes, identifying
high-value service zones, and planning regional marketing campaigns.
</p>"""

        return "<!--HTML-->" + p1 + p2 + table1_intro + table1 + table1_after + p3 + table2 + p4
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