import sqlite3
import os
import pandas as pd
from config import DB_PATH
from utils.helpers import ensure_dir


class FactorDatabase:

    def __init__(self, db_path: str = DB_PATH):
        ensure_dir(os.path.dirname(db_path))
        self.conn = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS daily_market (
                ts_code     TEXT NOT NULL,
                trade_date  TEXT NOT NULL,
                open        REAL,
                high        REAL,
                low         REAL,
                close       REAL,
                pre_close   REAL,
                vol         REAL,
                amount      REAL,
                ret         REAL,
                PRIMARY KEY (ts_code, trade_date)
            );

            CREATE TABLE IF NOT EXISTS factor_values (
                factor_name TEXT NOT NULL,
                ts_code     TEXT NOT NULL,
                trade_date  TEXT NOT NULL,
                value       REAL,
                PRIMARY KEY (factor_name, ts_code, trade_date)
            );

            CREATE INDEX IF NOT EXISTS idx_factor_name_date
                ON factor_values(factor_name, trade_date);

            CREATE INDEX IF NOT EXISTS idx_market_date
                ON daily_market(trade_date);
        """)
        self.conn.commit()

    def save_market(self, df: pd.DataFrame):
        df.to_sql("daily_market", self.conn, if_exists="replace", index=False, method="multi")

    def load_market(self, start: str, end: str) -> pd.DataFrame:
        sql = """
            SELECT * FROM daily_market
            WHERE trade_date BETWEEN ? AND ?
            ORDER BY ts_code, trade_date
        """
        return pd.read_sql(self.conn, sql, params=(start, end))

    def save_factor(self, factor_name: str, df: pd.DataFrame):
        df = df[["ts_code", "trade_date", "value"]].copy()
        df["factor_name"] = factor_name
        self.conn.execute("DELETE FROM factor_values WHERE factor_name = ?", (factor_name,))
        df.to_sql("factor_values", self.conn, if_exists="append", index=False, method="multi")
        self.conn.commit()

    def load_factor(self, factor_name: str, start: str, end: str) -> pd.DataFrame:
        sql = """
            SELECT ts_code, trade_date, value
            FROM factor_values
            WHERE factor_name = ? AND trade_date BETWEEN ? AND ?
            ORDER BY ts_code, trade_date
        """
        return pd.read_sql(self.conn, sql, params=(factor_name, start, end))

    def load_all_factors(self, factor_names: list, date: str) -> pd.DataFrame:
        placeholders = ",".join(["?"] * len(factor_names))
        sql = f"""
            SELECT ts_code, factor_name, value
            FROM factor_values
            WHERE factor_name IN ({placeholders}) AND trade_date = ?
        """
        params = factor_names + [date]
        df = pd.read_sql(self.conn, sql, params=params)
        if df.empty:
            return pd.DataFrame()
        return df.pivot(index="ts_code", columns="factor_name", values="value")

    def list_factors(self) -> list:
        cur = self.conn.execute("SELECT DISTINCT factor_name FROM factor_values")
        return [r[0] for r in cur.fetchall()]

    def close(self):
        self.conn.close()
