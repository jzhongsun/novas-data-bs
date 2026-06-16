import argparse
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import random
import sys
import time

import logging
from typing import Optional

import httpx
import pandas as pd
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_em_cookies():
    from playwright.sync_api import sync_playwright
    # 使用 sync_playwright() 代替之前的 async_playwright()
    with sync_playwright() as p:
        print("正在启动 [同步版] 深度伪装 Firefox 浏览器...")
        
        # 1. 启动轻量隐蔽的 Firefox 浏览器内核
        browser = p.firefox.launch(
            headless=True,  # 阿里云 DSW 云端环境必须为 True
            args=["--no-sandbox", "--disable-dev-shm-usage"]
        )
        
        # 2. 深度伪装浏览器上下文环境
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
            viewport={"width": 1920, "height": 1080},
            locale="zh-CN",
            timezone_id="Asia/Shanghai"
        )
        
        page = context.new_page()
        
        # 3. 注入防检测脚本（同步执行，抹除无头特征和 webdriver）
        page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            Object.defineProperty(navigator, 'languages', { get: () => ['zh-CN', 'zh'] });
            window.chrome = { runtime: {} };
        """)
        
        print("正在加载东方财富网页...")
        try:
            # 4. 同步加载网页，使用 domcontentloaded 避免被东方财富的长连接卡死
            page.goto("https://quote.eastmoney.com/zz/2.H11030.html#fullScreenChart", wait_until="domcontentloaded", timeout=15000)
            time.sleep(2)
            
            # 5. 模拟人机交互，强迫东方财富底层的混淆加密 JS 运转生成核心 Cookie
            print("正在模拟人类交互（鼠标无规律移动与轻微点击）...")
            for _ in range(5):
                x = random.randint(100, 800)
                y = random.randint(100, 600)
                page.mouse.move(x, y)  # 同步鼠标移动
                time.sleep(random.uniform(0.1, 0.3))
            
            # 模拟在网页空白处点击一次，激活点击事件监听器
            page.mouse.click(200, 200)
            
            # 模拟网页无规律滚动
            page.evaluate("window.scrollTo(0, 300);")
            time.sleep(1)
            page.evaluate("window.scrollTo(0, 600);")
            
            # 6. 关键等待：给异步混淆脚本计算并更新 Cookie 留出充足时间
            print("正在等待东方财富核心数据渲染与 Cookie 释放...")
            time.sleep(4)

            page.screenshot(path="eastmoney_page.png")
            
            # 7. 同步提取浏览器在内存中已经生成好的全部 Cookie
            cookies = context.cookies()
            print(f"【成功】当前获取到有效 Cookie 数量：{len(cookies)}")
            
            cookie_dict = {ck['name']: ck['value'] for ck in cookies}
            print("当前生成的 Cookie 所有键名：", list(cookie_dict.keys()))
            
            # 8. 将获取到的 Cookie 持久化保存到本地 JSON 文件中
            with open("eastmoney_cookies.json", "w", encoding="utf-8") as f:
                json.dump(cookies, f, ensure_ascii=False, indent=4)
                
            print("Cookie 已成功序列化至本地 eastmoney_cookies.json")
            
        except Exception as e:
            print(f"同步执行过程中发生异常: {e}")
        finally:
            browser.close()

        return cookies


@dataclass
class DailyKlineBar:
    """日线K线单根数据
    
    API 数据格式: "日期,开盘,收盘,最高,最低,成交量,成交额,振幅,涨跌幅,涨跌额,换手率"
    示例: "2026-01-21,99.91,102.50,103.00,99.00,12345,123456789.00,4.0,2.5,2.59,1.2"
    
    pre_close 通过计算得出: pre_close = close - change
    """
    date: str              # 日期 YYYY-MM-DD
    open: float            # 开盘价
    close: float           # 收盘价
    high: float            # 最高价
    low: float             # 最低价
    volume: float          # 成交量
    amount: float          # 成交额
    amplitude: float       # 振幅 %
    change_pct: float      # 涨跌幅 %
    change: float          # 涨跌额
    turnover_rate: float   # 换手率 %
    pre_close: float       # 昨收价 (= close - change)


@dataclass
class DailyKlineData:
    """日线K线数据"""
    code: str                    # 股票代码
    name: str                    # 股票名称
    market: int                  # 市场 (0: 深市, 1: 沪市)
    pre_close: float             # 昨收价（查询周期首日的前一日收盘价）
    bars: list[DailyKlineBar]    # 日线K线列表
    
    @property
    def total_bars(self) -> int:
        """K线总数"""
        return len(self.bars)
    
    def to_dataframe(self):
        """转换为 pandas DataFrame
        
        Returns:
            pd.DataFrame: 包含日线K线数据的 DataFrame，列包括:
                - date: 日期索引
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - pre_close: 昨收价 (= close - change)
                - volume: 成交量
                - amount: 成交额
                - amplitude: 振幅 %
                - change_pct: 涨跌幅 %
                - change: 涨跌额
                - turnover_rate: 换手率 %
                
            DataFrame.attrs 包含元数据:
                - code: 股票代码
                - name: 股票名称
                - market: 市场
                - pre_close: 查询周期首日的昨收价
        """
        import pandas as pd
        
        if not self.bars:
            return pd.DataFrame()
        
        records = []
        for bar in self.bars:
            records.append({
                "code": self.code,
                "name": self.name,
                "date": bar.date,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "pre_close": bar.pre_close,
                "volume": bar.volume,
                "amount": bar.amount,
                "amplitude": bar.amplitude,
                "change_pct": bar.change_pct,
                "change": bar.change,
                "turnover_rate": bar.turnover_rate,
            })
        
        df = pd.DataFrame(records)
        
        # 设置日期索引
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index(["date"])
        
        # 添加元数据作为 attrs
        df.attrs["code"] = self.code
        df.attrs["name"] = self.name
        df.attrs["market"] = self.market
        df.attrs["pre_close"] = self.pre_close
        
        return df
    
    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            "code": self.code,
            "name": self.name,
            "market": self.market,
            "pre_close": self.pre_close,
            "total_bars": self.total_bars,
            "bars": [
                {
                    "date": bar.date,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "pre_close": bar.pre_close,
                    "volume": bar.volume,
                    "amount": bar.amount,
                    "amplitude": bar.amplitude,
                    "change_pct": bar.change_pct,
                    "change": bar.change,
                    "turnover_rate": bar.turnover_rate,
                }
                for bar in self.bars
            ]
        }

def format_secid(code: str, market: Optional[int] = None) -> str:
    """
    格式化股票代码为 secid 格式
    
    Args:
        code: 股票代码，支持以下格式：
            - "0.301338" - 已经是 secid 格式
            - "301338" - 纯数字代码（自动判断市场）
            - "sz301338" / "sh600519" - 带市场前缀
        market: 市场代码（可选）
            - 0: 深市（含创业板）
            - 1: 沪市（含科创板）
            
    Returns:
        secid 格式的代码，如 "0.301338"
    """
    code = code.strip()
    
    # 已经是 secid 格式
    if "." in code:
        return code
    
    # 带市场前缀
    if code.lower().startswith("sz"):
        return f"0.{code[2:]}"
    elif code.lower().startswith("sh"):
        return f"1.{code[2:]}"
    
    # 如果指定了市场
    if market is not None:
        return f"{market}.{code}"
    
    # 自动判断市场
    # 6开头：沪市主板
    # 688开头：科创板（沪市）
    # 0开头：深市主板
    # 3开头：创业板（深市）
    if code.startswith("6"):
        return f"1.{code}"
    elif code.startswith("0") or code.startswith("3"):
        return f"0.{code}"
    else:
        # 默认深市
        return f"0.{code}"

def get_daily_kline_sync(
    code: str,
    start_date: str,
    end_date: str,
    period: str = "daily",
    adjust: str = "",
    COOKIE: str = "",
) -> Optional[DailyKlineData]:
    """
    获取股票日线K线数据（同步）

    Args:
        code: 股票代码（支持多种格式，见 format_secid）
        start_date: 开始日期，格式 YYYY-MM-DD
        end_date: 结束日期，格式 YYYY-MM-DD
        period: K线周期，支持 "daily", "weekly", "monthly", "hourly", "30m", "15m", "5m"
        adjust: 复权类型，"" 不复权, "qfq" 前复权, "hfq" 后复权

    Returns:
        DailyKlineData 对象，失败返回 None
    """
    secid = format_secid(code)

    # 日期格式转换
    begin_date = datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y%m%d") if start_date and start_date != "0" else "0"
    end_date_fmt = datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y%m%d")

    # 复权类型映射
    adjust_dict = {"qfq": "1", "hfq": "2", "": "0"}
    adjust_code = adjust_dict.get(adjust, "0")

    # K线周期映射
    period_dict = {
        "daily": "101", "weekly": "102", "monthly": "103",
        "hourly": "60", "30m": "30", "15m": "15", "5m": "5"
    }
    period_code = period_dict.get(period, "101")

    params = {
        "secid": secid,
        "ut": "fa5fd1943c7b386f172d6893dbfba10b",
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": period_code,
        "fqt": adjust_code,
        "beg": begin_date,
        "end": end_date_fmt,
        "lmt": 1000000,
        "_": int(datetime.now().timestamp() * 1000),
    }
    DAILY_KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    DEFAULT_HEADERS = {
        "Accept": "*/*",
        "Accept-Language": "zh-CN,zh;q=0.9,zh-TW;q=0.8",
        "Connection": "keep-alive",
        "Referer": "https://quote.eastmoney.com/",
        "Sec-Fetch-Dest": "script",
        "Sec-Fetch-Mode": "no-cors",
        "Sec-Fetch-Site": "same-site",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36",
        "sec-ch-ua": '"Chromium";v="148", "Google Chrome";v="148", "Not/A)Brand";v="99"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"macOS"',
    }
    
    try:
        headers = DEFAULT_HEADERS.copy()
        headers["Cookie"] = COOKIE
        with httpx.Client(timeout=10, follow_redirects=True) as client:
            print(f"请求日线K线数据，参数: {params}")
            response = client.get(
                DAILY_KLINE_URL,
                params=params,
                headers=headers
            )
            response.raise_for_status()
            data = response.json()

            if data.get("rc") != 0 or not data.get("data"):
                return None

            return self._parse_daily_kline(data["data"])

    except Exception as e:
        print(f"获取日线K线数据失败: {e}")
        return None

def _parse_daily_kline(self, data: dict) -> DailyKlineData:
    """解析日线K线数据
    
    API 返回数据格式:
    {
        "code": "301338",
        "market": 0,
        "name": "凯格精机",
        "klines": [
            "2026-01-21,99.91,102.50,103.00,99.00,12345,123456789.00,4.0,2.5,2.59,1.2",
            ...
        ]
    }
    
    klines 每行格式: "日期,开盘,收盘,最高,最低,成交量,成交额,振幅,涨跌幅,涨跌额,换手率"
    """
    bars = []
    klines = data.get("klines", [])
    
    for kline in klines:
        parts = kline.split(",")
        if len(parts) >= 11:
            close = self._safe_float(parts[2])
            change = self._safe_float(parts[9])
            bars.append(DailyKlineBar(
                date=parts[0],                           # 日期
                open=self._safe_float(parts[1]),         # 开盘价
                close=close,                             # 收盘价
                high=self._safe_float(parts[3]),         # 最高价
                low=self._safe_float(parts[4]),          # 最低价
                volume=self._safe_float(parts[5]),       # 成交量
                amount=self._safe_float(parts[6]),       # 成交额
                amplitude=self._safe_float(parts[7]),    # 振幅 %
                change_pct=self._safe_float(parts[8]),   # 涨跌幅 %
                change=change,                           # 涨跌额
                turnover_rate=self._safe_float(parts[10]),  # 换手率 %
                pre_close=round(close - change, 4),      # 昨收价
            ))
    
    return DailyKlineData(
        code=data.get("code", ""),
        name=data.get("name", ""),
        market=data.get("market", 0),
        pre_close=self._safe_float(data.get("preKPrice", 0)),
        bars=bars,
    )

def _read_any(path: str) -> pd.DataFrame:
    """Read parquet or CSV, auto-detect by extension."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if p.suffix == ".parquet":
        return pd.read_parquet(path, engine="pyarrow")
    return pd.read_csv(path)

def download_index_data(
    index_path: str | None = None,
    start_date: str = "0",
    end_date: str = "2050-01-01",
    output: str | None = None,
    sleep_secs: float = 5.0,
) -> pd.DataFrame:
    """从东方财富下载中基协行业指数日线数据（parquet）。

    Args:
        index_path: 指数方案 CSV/parquet（含 指数代码 列）。
        start_date: "0"=最早。
        end_date: 结束日期。
        output: 输出 parquet 路径。
        sleep_secs: 请求间隔秒数。
    """
    import time as _time
    DEFAULT_INDEX_CSV = "中基协基金估值行业分类指数编制方案.csv"
    DEFAULT_AMAC_INDEX_OUTPUT = Path("./output") / "amac_industry_daily.parquet"

    path = index_path or str(DEFAULT_INDEX_CSV)
    out = output or str(DEFAULT_AMAC_INDEX_OUTPUT)
    df_index = _read_any(path)

    logger.info("下载 %d 个 AMAC 行业指数日线", len(df_index))
    logger.info("  日期范围: %s ~ %s", start_date, end_date)

    cookies = fetch_em_cookies()
    cs = [f"{ck['name']}={ck['value']}" for ck in cookies]
    cs = "; ".join(cs)
    logger.info("已获取 Cookie: %s", cs)

    frames: list[pd.DataFrame] = []

    for idx, row in df_index.iterrows():
        index_code = row["指数代码"]
        index_name = row.get("指数名称", "")
        logger.info("[%d/%d] %s %s", idx + 1, len(df_index), index_code, index_name)

        secid = f"2.{index_code}"
        data = get_daily_kline_sync(secid, start_date, end_date, COOKIE="Cookie: " + cs)
        if data is not None:
            df = data.to_dataframe()
            if not df.empty:
                frames.append(df)
            else:
                logger.warning("  数据为空")
        else:
            logger.warning("  获取失败")

        if idx + 1 < len(df_index):
            _time.sleep(sleep_secs)

    if not frames:
        logger.error("未获取到任何指数数据")
        return pd.DataFrame()

    merged = pd.concat(frames)
    logger.info("合并完成: rows=%d, codes=%d", len(merged), merged["code"].nunique())

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out, index=True, engine="pyarrow")
    logger.info("已保存: %s", out)

    return merged

def cmd_download(args: argparse.Namespace) -> None:
    df = download_index_data(
        index_path=args.index_file,
        start_date=args.start_date or "0",
        end_date=args.end_date or "2050-01-01",
        output=args.output,
        sleep_secs=args.sleep,
    )

    if args.upload and not df.empty:
        if not args.repo_id:
            logger.error("--repo-id is required when --upload is set")
            sys.exit(1)
        from baostock_data import upload_to_modelscope
        upload_to_modelscope(
            local_dir=str(Path(args.output or str(DEFAULT_AMAC_INDEX_OUTPUT)).parent),
            repo_id=args.repo_id,
        )

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AMAC 行业分类工具 — 从 CAPCO PDF 提取行业分类并映射中基协指数",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python amac.py run                                      # 使用默认路径一键执行
  python amac.py run --pdf a.pdf --output-dir ./out       # 自定义 PDF 和输出目录
  python amac.py extract --pdf a.pdf --output stocks.csv  # 仅提取 PDF
  python amac.py map --stocks stocks.csv --index-file index.csv  # 仅映射
  python amac.py download                                 # 下载 AMAC 行业指数日线
  python amac.py download --start-date 2020-01-01         # 指定日期
        """,
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="输出详细日志"
    )

    subparsers = parser.add_subparsers(dest="command", help="可用子命令")

    # ---- download ----
    dl_parser = subparsers.add_parser("download", help="下载 AMAC 行业指数日线数据")
    dl_parser.add_argument("--index-file", help="指数方案 CSV 路径")
    dl_parser.add_argument("--start-date", default="0", help="开始日期 (YYYY-MM-DD)，0=最早")
    dl_parser.add_argument("--end-date", default="2050-01-01", help="结束日期 (YYYY-MM-DD)")
    dl_parser.add_argument("--output", help="输出 parquet 路径")
    dl_parser.add_argument("--sleep", type=float, default=5.0, help="请求间隔秒数 (默认 5)")
    dl_parser.add_argument("--upload", action="store_true", help="上传到 ModelScope")
    dl_parser.add_argument("--repo-id", default="", help="ModelScope repo (required if --upload)")
    dl_parser.set_defaults(func=cmd_download)


    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
