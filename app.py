import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
from datetime import datetime

# ============ 页面配置 ============
st.set_page_config(page_title="📈 ATAS 自动化交易分析报告生成器",
                   layout="wide",
                   initial_sidebar_state="expanded")

# ============ 多语言支持 ============
LANG = {
    '中文': '📈 ATAS 自动化交易分析报告生成器',
    'English': '📈 ATAS Automated Trading Report Generator'
}
lang = st.sidebar.selectbox('语言 / Language', list(LANG.keys()))
st.title(LANG[lang])

# ============ 侧边栏 ============
st.sidebar.header('📁 上传与快照管理')
uploaded = st.sidebar.file_uploader(
    '上传 ATAS 导出数据 (.xlsx)',
    type='xlsx',
    accept_multiple_files=True
)
market_file = st.sidebar.file_uploader(
    '上传市场快照 CSV (Symbol,Time,MarketPrice)',
    type='csv'
)
sent_file = st.sidebar.file_uploader(
    '上传舆情数据 CSV (Symbol,Date,SentimentScore)',
    type='csv'
)
max_snapshots = st.sidebar.number_input(
    '保留最近快照份数',
    min_value=1,
    value=10
)

SNAP_DIR = 'snapshots'
os.makedirs(SNAP_DIR, exist_ok=True)

# ============ 数据加载 ============
@st.cache_data
def load_and_clean(files):
    dfs = []
    for f in files:
        # 读取“日志”工作表
        df_log = pd.read_excel(f, sheet_name='日志')
        df_log['上传文件'] = f.name
        dfs.append(df_log)
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    # 只保留有 PnL 的记录
    df = df.dropna(subset=['PnL'])
    # 重命名列
    df = df.rename(columns={
        '平仓时间': '时间',
        '平仓价': '价格',
        '平仓量': '数量',
        'PnL': '盈亏'
    })
    # 转换类型
    df['时间'] = pd.to_datetime(df['时间'], errors='coerce')
    df['数量'] = df['数量'].abs()
    df['方向'] = df['开仓量'].apply(lambda x: 'Buy' if x > 0 else 'Sell')
    for col in ['价格', '数量', '盈亏']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['时间', '价格']).sort_values('时间').reset_index(drop=True)
    return df

if uploaded:
    df = load_and_clean(uploaded)

    # 保存快照并清理旧文件
    snap_file = f"atas_snapshot_{len(uploaded)}files_{datetime.now():%Y%m%d_%H%M%S}.csv"
    df.to_csv(os.path.join(SNAP_DIR, snap_file), index=False)
    snaps = sorted(os.listdir(SNAP_DIR))
    if len(snaps) > max_snapshots:
        for old in snaps[:-max_snapshots]:
            os.remove(os.path.join(SNAP_DIR, old))
    st.sidebar.success(f"已加载 {len(df)} 条交易，生成快照：{snap_file}")
    st.sidebar.write({f.name: len(df[df['上传文件']==f.name]) for f in uploaded})

    # 视图和风险预警
    view = st.sidebar.selectbox('视图分组', ['总体', '按账户', '按品种'])
    st.sidebar.header('⚠️ 风险阈值预警')
    max_loss = st.sidebar.number_input('单笔最大亏损', value=-100.0)
    max_trades = st.sidebar.number_input('日内最大交易次数', value=50)
    today_count = df[df['时间'].dt.date == datetime.today().date()].shape[0]
    if df['盈亏'].min() < max_loss:
        st.warning(f"⚠️ 存在单笔盈亏低于阈值({max_loss})！")
    if today_count > max_trades:
        st.warning(f"⚠️ 今日交易次数超过阈值({max_trades})！")

    # 核心指标计算
    df['累计盈亏'] = df['盈亏'].cumsum()
    df['日期'] = df['时间'].dt.date
    df['小时'] = df['时间'].dt.hour
    period_days = max((df['时间'].max() - df['时间'].min()).days, 1)
    total_pl = df['盈亏'].sum()
    ann_return = total_pl / period_days * 252
    downside_dev = df[df['盈亏'] < 0]['盈亏'].std()
    var95 = -df['盈亏'].quantile(0.05)
    cvar95 = -df[df['盈亏'] <= df['盈亏'].quantile(0.05)]['盈亏'].mean()
    sharpe = df['盈亏'].mean() / df['盈亏'].std() * np.sqrt(252)
    winrate = (df['盈亏'] > 0).mean()
    profit_factor = df[df['盈亏'] > 0]['盈亏'].mean() / (-df[df['盈亏'] < 0]['盈亏'].mean())
    mdd = (df['累计盈亏'] - df['累计盈亏'].cummax()).min()

    # 交互式图表示例（更多见完整脚本）
    st.subheader('📈 累计盈亏趋势')
    fig = px.line(df, x='时间', y='累计盈亏', color='账户' if view == '按账户' else None)
    st.plotly_chart(fig, use_container_width=True)

    # …（其余图表与指标同之前版本）…

else:
    st.info('👆 请在侧边栏上传 ATAS 导出 .xlsx 文件进行分析')
