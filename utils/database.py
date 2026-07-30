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
            CREATE TABLE IF NOT EXISTS index_weights (
                index_code  TEXT NOT NULL,
                con_code    TEXT NOT NULL,
                trade_date  TEXT NOT NULL,
                weight      REAL,
                PRIMARY KEY (index_code, con_code, trade_date)
            );
            CREATE INDEX IF NOT EXISTS idx_factor_name_date
                ON factor_values(factor_name, trade_date);
            CREATE INDEX IF NOT EXISTS idx_factor_code_date
                ON factor_values(ts_code, trade_date);
            CREATE INDEX IF NOT EXISTS idx_market_date
                ON daily_market(trade_date);
            CREATE INDEX IF NOT EXISTS idx_iw_code_date
                ON index_weights(index_code, trade_date);
            CREATE TABLE IF NOT EXISTS download_log (
                trade_date TEXT PRIMARY KEY
            );
        """)
        self.conn.commit()

    # ---- 行情存取 ----

    def save_market_raw(self, df: pd.DataFrame):
        """仅 INSERT，不做日期检查。下载器专用。"""
        cols = ["ts_code", "trade_date", "open", "high", "low", "close",
                "pre_close", "vol", "amount", "ret"]
        sql = f"INSERT OR IGNORE INTO daily_market ({','.join(cols)}) VALUES ({','.join(['?']*len(cols))})"
        for start in range(0, len(df), CHUNK_SIZE):
            chunk = df[cols].iloc[start:start + CHUNK_SIZE]
            rows = [tuple(r) for r in chunk.to_numpy().tolist()]
            self.conn.executemany(sql, rows)
        self.conn.commit()

    def save_market(self, df: pd.DataFrame):
        existing_dates = self.get_existing_dates()
        new_dates = set(df["trade_date"].unique())
        to_insert = new_dates - existing_dates
        if not to_insert:
            print("  [数据库] 行情已存在，跳过写入")
            return
        df = df[df["trade_date"].isin(to_insert)]

        cols = ["ts_code", "trade_date", "open", "high", "low", "close",
                "pre_close", "vol", "amount", "ret"]
        sql = f"INSERT INTO daily_market ({','.join(cols)}) VALUES ({','.join(['?']*len(cols))})"
        for start in range(0, len(df), CHUNK_SIZE):
            chunk = df[cols].iloc[start:start + CHUNK_SIZE]
            rows = [tuple(r) for r in chunk.to_numpy().tolist()]
            self.conn.executemany(sql, rows)
        self.conn.commit()
        print(f"  [数据库] 新增 {len(to_insert)} 天, 共 {df.shape[0]} 行")

    def get_existing_dates(self) -> set:
        cur = self.conn.execute("SELECT DISTINCT trade_date FROM daily_market")
        return {r[0] for r in cur.fetchall()}

    def get_missing_dates(self, all_dates: list) -> list:
        existing = self.get_existing_dates()
        # 还检查 download_log（Tushare 返回空的日期也标记为已尝试）
        cur = self.conn.execute("SELECT trade_date FROM download_log")
        attempted = {r[0] for r in cur.fetchall()}
        return [d for d in all_dates if d not in existing and d not in attempted]

    def mark_dates_downloaded(self, dates: list):
        sql = "INSERT OR IGNORE INTO download_log (trade_date) VALUES (?)"
        self.conn.executemany(sql, [(d,) for d in dates])
        self.conn.commit()

    def load_market_by_dates(self, dates: list) -> pd.DataFrame:
        if not dates:
            return pd.DataFrame()
        placeholders = ",".join(["?"] * len(dates))
        sql = f"SELECT * FROM daily_market WHERE trade_date IN ({placeholders}) ORDER BY ts_code, trade_date"
        return pd.read_sql(sql, self.conn, params=dates)

    def save_factor(self, factor_name: str, df: pd.DataFrame):
        self.conn.execute("DELETE FROM factor_values WHERE factor_name = ?", (factor_name,))
        sql = "INSERT INTO factor_values (factor_name,ts_code,trade_date,value) VALUES (?,?,?,?)"
        sub = df[["ts_code", "trade_date", "value"]]
        for start in range(0, len(sub), CHUNK_SIZE):
            chunk = sub.iloc[start:start + CHUNK_SIZE]
            rows = [(factor_name, str(r[0]), str(r[1]), float(r[2]))
                    for r in chunk.to_numpy()]
            self.conn.executemany(sql, rows)
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

    def load_market_with_future(self, start: str, end: str) -> pd.DataFrame:
        """行情 + 未来收益。ic_analysis / layer_backtest 共用。"""
        df = self.load_market(start, end)
        df["future_ret"] = df.groupby("ts_code")["ret"].shift(-1)
        return df

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

    # ---- 指数成分股 ----

    def save_index_weights(self, df: pd.DataFrame, index_code: str):
        """保存指数权重到 index_weights 表（REPLACE 语义，按月覆盖）"""
        self.conn.execute(
            "DELETE FROM index_weights WHERE index_code = ?", (index_code,)
        )
        sql = "INSERT INTO index_weights (index_code,con_code,trade_date,weight) VALUES (?,?,?,?)"
        rows = [(index_code, str(r["con_code"]), str(r["trade_date"]), float(r["weight"]))
                for _, r in df.iterrows()]
        for i in range(0, len(rows), CHUNK_SIZE):
            self.conn.executemany(sql, rows[i:i + CHUNK_SIZE])
        self.conn.commit()
        print(f"  [数据库] 已存储 {index_code} 成分股 {len(rows)} 条")

    def get_index_stocks(self, index_code: str) -> pd.DataFrame:
        """返回 index_weights 全部数据：con_code, trade_date, weight"""
        sql = """
            SELECT con_code, trade_date, weight
            FROM index_weights
            WHERE index_code = ?
            ORDER BY trade_date, con_code
        """
        cur = self.conn.execute(sql, (index_code,))
        rows = cur.fetchall()
        return pd.DataFrame(rows, columns=[d[0] for d in cur.description])

    def close(self):
        self.conn.close()
