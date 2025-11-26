# core/data_handlers.py
import pandas as pd
import openpyxl
import akshare as ak
from abc import ABC, abstractmethod

class BaseDataHandler(ABC):
    """
    数据处理器抽象基类(Abstract Base Class, ABC)。
    所有处理器（API、Excel等）都必须实现 get_data() 方法。
    """
    @abstractmethod     # 强制子类实现get_data()方法
    def get_data(self) -> pd.DataFrame:
        """
        获取数据并将其标准化为带pd.DatetimeIndex 索引的 DataFrame。
        """
        raise NotImplementedError

class ExcelHandler(BaseDataHandler):
    """
    适配从Excel文件读取OHLCV数据，针对数据格式：
    B1单元格为代码，A2:F2为'日期','开盘价','最高价','最低价','收盘价','成交量'。
    数据从第三行开始。
    """
    def __init__(self, file_path: str, sheet_name: str | int = 0):
        """
        初始化Excel处理器。

        Parameters
        ----------
        file_path: str
            Excel文件的路径
        sheet_name: str
            要读取的工作表名称或索引，默认为0（第一个工作表）。
        """
        self.file_path = file_path
        self.sheet_name = sheet_name
        # 定义Excel列名到标准列名的映射：A2:F2 -> date, open, high, low, close, volume
        self.COLUMN_MAP = {
            '日期': 'date',
            '开盘价': 'open',
            '最高价': 'high',
            '最低价': 'low',
            '收盘价': 'close',
            '成交量': 'volume'
        }

    def get_data(self) -> pd.DataFrame:
        """
        读取Excel文件，清洗数据，并返回标准化的DataFrame。
        """
        try:
            # 1. 读取Excel，指定第2行(索引1)为表头
            raw_df = pd.read_excel(
                self.file_path,
                sheet_name=self.sheet_name,
                header=1, 
                engine='openpyxl'
            )
        except Exception as e:
            print(f"读取Excel失败: {e}")
            return pd.DataFrame()

        # 2. 检查必要的中文列名是否存在
        # 注意：raw_df 此时的列名是中文（'日期', '开盘价'...），必须先查这些 key 是否存在
        required_cn_cols = set(self.COLUMN_MAP.keys())
        if not required_cn_cols.issubset(raw_df.columns):
            missing = required_cn_cols - set(raw_df.columns)
            print(f"错误: 表中缺少必要的列: {missing}")
            return pd.DataFrame()

        # 3. 重命名列 (中文 -> 英文) 及初步清洗
        # 使用 copy() 避免 SettingWithCopyWarning
        df = raw_df.rename(columns=self.COLUMN_MAP).copy()

        # 4. 格式化日期索引
        try:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            # 删除日期转换失败的行（如空行或标题行残留）
            df = df.dropna(subset=['date'])
            df = df.set_index('date').sort_index()
        except Exception as e:
            print(f"日期列处理错误: {e}")
            return pd.DataFrame()

        # 5. 格式化数值列
        target_cols = ['open', 'high', 'low', 'close', 'volume']
        # 使用 apply 批量将这些列转换为数字，无法转换的变成 NaN
        df[target_cols] = df[target_cols].apply(pd.to_numeric, errors='coerce')

        # 6. 删除包含 NaN 的行并只保留目标列
        df = df[target_cols].dropna()

        # 输出成功信息
        print(f"Excel数据加载成功: {self.file_path}")
        print(f"  时间范围: {df.index[0].date()} 至 {df.index[-1].date()} | 行数: {len(df)}")
        
        return df



class MacroExcelHandler(BaseDataHandler):
    """
    【新增】适配从Excel加载的宏观数据 (如 CPI, VIX) 。
    这是一个更通用的版本，用于非OHLC数据。
    """
    def __init__(self, file_path: str, sheet_name: str | int, 
                 date_col: str, data_cols: list[str]):
        """
        :param file_path: Excel 文件路径
        :param sheet_name: 表格名称
        :param date_col: 日期列的列名 (e.g., '月份')
        :param data_cols: 您希望加载的数据列名 (e.g., ['CPI', 'VIX'])
        """
        self.file_path = file_path
        self.sheet_name = sheet_name
        self.date_col = date_col
        self.data_cols = data_cols

    def get_data(self) -> pd.DataFrame:
        try:
            raw_df = pd.read_excel(
                self.file_path,
                sheet_name=self.sheet_name,
                engine='openpyxl'
            ) # [10, 11, 12, 16]
        except Exception as e:
            print(f"读取宏观Excel时发生错误: {e}")
            return pd.DataFrame()

        cols_to_load = [self.date_col] + self.data_cols
        if not all(col in raw_df.columns for col in cols_to_load):
            print(f"错误: 宏观数据表 '{self.sheet_name}' 中缺少列。")
            return pd.DataFrame()

        std_df = raw_df[cols_to_load].copy()
        std_df[self.date_col] = pd.to_datetime(std_df[self.date_col])
        std_df = std_df.set_index(self.date_col).sort_index()
        
        # 处理月度/季度数据可能产生的重复索引（如果需要）
        std_df = std_df[~std_df.index.duplicated(keep='last')]
        
        print(f"宏观数据 '{self.data_cols}' 加载成功。")
        return std_df
    

class iFindExcelHandler(BaseDataHandler):
    """
    适配从同花顺iFind导出的Excel文件 。
    """
    def __init__(self, file_path: str, sheet_name: str | int = 0, header_row: int = 0):
        self.file_path = file_path
        self.sheet_name = sheet_name
        self.header_row = header_row
        
        # 核心：定义iFind列名到标准列名的映射
        self.COLUMN_MAP = {
            '日期': 'date',
            '股票代码': 'code',
            '开盘价': 'open',
            '最高价': 'high',
            '最低价': 'low',
            '收盘价': 'close',
            '成交量': 'volume'
            # (您可以根据需要添加其他iFind列)
        }

    def get_data(self) -> pd.DataFrame:
        """
        读取Excel [10, 11, 12, 13, 14, 15, 16, 17, 18]，清洗，并返回标准化的DataFrame。
        """
        try:
            raw_df = pd.read_excel(
                self.file_path,
                sheet_name=self.sheet_name,
                header=self.header_row,
                engine='openpyxl'
            )
        except FileNotFoundError:
            print(f"错误: 文件未找到 {self.file_path}")
            return pd.DataFrame()
        except Exception as e:
            print(f"读取Excel时发生错误: {e}")
            return pd.DataFrame()

        std_df = raw_df.rename(columns=self.COLUMN_MAP)
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        
        if 'date' not in std_df.columns:
            print("错误: 缺少 'date' 列。")
            return pd.DataFrame()

        std_df['date'] = pd.to_datetime(std_df['date'])
        std_df = std_df.set_index('date').sort_index()

        # 仅保留存在的标准列
        available_cols = [col for col in required_cols if col in std_df.columns]
        std_df = std_df[available_cols]

        print(f"iFind Excel数据加载成功: {self.file_path} (Sheet: {self.sheet_name})")
        return std_df

class SinaDataHandler(BaseDataHandler):
    """
    适配新浪财经数据源 (通过akshare) [19, 20, 21, 22, 23, 24, 25, 1, 26, 27]。
    """
    def __init__(self, symbol: str, start_date: str, end_date: str, adjust: str = "qfq"):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.adjust = adjust
        self.COLUMN_MAP = {
            '日期': 'date', '开盘': 'open', '最高': 'high',
            '最低': 'low', '收盘': 'close', '成交量': 'volume'
        }

    def get_data(self) -> pd.DataFrame:
        try:
            raw_df = ak.stock_zh_a_hist(
                symbol=self.symbol, period="daily",
                start_date=self.start_date, end_date=self.end_date,
                adjust=self.adjust
            ) # [22, 23, 24, 25]
        except Exception as e:
            print(f"akshare/Sina 数据获取失败: {e}")
            return pd.DataFrame()

        std_df = raw_df.rename(columns=self.COLUMN_MAP)
        std_df['date'] = pd.to_datetime(std_df['date'])
        std_df = std_df.set_index('date').sort_index()
        
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        std_df = std_df[required_cols].astype(float)
        
        print(f"Sina/akshare数据加载成功: {self.symbol}")
        return std_df
