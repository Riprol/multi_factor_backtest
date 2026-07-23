import sqlite3
import os
import pandas as pd
from config import DB_PATH
from utils.helpers import ensure_dir

CHUNK_SIZE = 5000


class FactorDatabase:

    def __init__(self, db_path: str = DB_PATH):
        ensure_dir(os.path.dirname(db_path))
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, timeout=30)
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

    # ---- 数据写入（使用 executemany 分批写入，避免 SQL 变量超限） ----

    def save_market(self, df: pd.DataFrame):
        self.conn.execute("DELETE FROM daily_market")
        cols = ["ts_code", "trade_date", "open", "high", "low", "close",
                "pre_close", "vol", "amount", "ret"]
        # 用 to_numpy 转 list 比 iterrows 快很多
        rows = [tuple(r) for r in df[cols].to_numpy().tolist()]
        sql = f"INSERT INTO daily_market ({','.join(cols)}) VALUES ({','.join(['?']*len(cols))})"
        for i in range(0, len(rows), CHUNK_SIZE):
            self.conn.executemany(sql, rows[i:i + CHUNK_SIZE])
        self.conn.commit()

    def save_factor(self, factor_name: str, df: pd.DataFrame):
        self.conn.execute("DELETE FROM factor_values WHERE factor_name = ?", (factor_name,))
        # 高效转换：用 to_numpy 代替 iterrows
        vals = df[["ts_code", "trade_date", "value"]].to_numpy()
        rows = [(factor_name, str(r[0]), str(r[1]), float(r[2])) for r in vals]
        sql = "INSERT INTO factor_values (factor_name,ts_code,trade_date,value) VALUES (?,?,?,?)"
        for i in range(0, len(rows), CHUNK_SIZE):
            self.conn.executemany(sql, rows[i:i + CHUNK_SIZE])
        self.conn.commit()

    # ---- 数据读取（使用 sqlite3 游标，避免 pandas 3.0 兼容问题） ----

    def load_market(self, start: str, end: str) -> pd.DataFrame:
        sql = """
            SELECT ts_code, trade_date, open, high, low, close,
                   pre_close, vol, amount, ret
            FROM daily_market
            WHERE trade_date BETWEEN ? AND ?
            ORDER BY ts_code, trade_date
        """
        cur = self.conn.execute(sql, (start, end))
        rows = cur.fetchall()
        return pd.DataFrame(rows, columns=[d[0] for d in cur.description])

    def load_factor(self, factor_name: str, start: str, end: str) -> pd.DataFrame:
        sql = """
            SELECT ts_code, trade_date, value
            FROM factor_values
            WHERE factor_name = ? AND trade_date BETWEEN ? AND ?
            ORDER BY ts_code, trade_date
        """
        cur = self.conn.execute(sql, (factor_name, start, end))
        rows = cur.fetchall()
        return pd.DataFrame(rows, columns=[d[0] for d in cur.description])

    def load_all_factors(self, factor_names: list, date: str) -> pd.DataFrame:
        if not factor_names:
            return pd.DataFrame()
        placeholders = ",".join(["?"] * len(factor_names))
        sql = f"""
            SELECT ts_code, factor_name, value
            FROM factor_values
            WHERE factor_name IN ({placeholders}) AND trade_date = ?
        """
        cur = self.conn.execute(sql, factor_names + [date])
        rows = cur.fetchall()
        df = pd.DataFrame(rows, columns=[d[0] for d in cur.description])
        if df.empty:
            return pd.DataFrame()
        return df.pivot(index="ts_code", columns="factor_name", values="value")

    def list_factors(self) -> list:
        cur = self.conn.execute("SELECT DISTINCT factor_name FROM factor_values")
        return [r[0] for r in cur.fetchall()]

    def close(self):
        self.conn.close()
