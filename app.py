import streamlit as st
import pandas as pd
import numpy as np
import os
import io
from datetime import datetime, timedelta
import plotly.express as px

# ============ PDF 依赖检测 ============
try:
    from fpdf import FPDF
    pdf_available = True
except ImportError:
    FPDF = None
    pdf_available = False

# ============ 页面配置 ============
st.set_page_config(page_title="📈 ATAS 自动化交易分析报告生成器", layout="wide", initial_sidebar_state="expanded")

# ============ 多语言支持 ============
LANG = {'中文': '📈 ATAS 自动化交易分析报告生成器', 'English': '📈 Automated Trading Report Generator'}
lang = st.sidebar.selectbox('语言 / Language', list(LANG.keys()))
st.title(LANG[lang])

if lang == '中文' and not pdf_available:
    st.sidebar.warning('未检测到 PDF 导出库，PDF 导出功能已禁用，请在 requirements.txt 中添加 `fpdf2`')

# ============ 配置参数 ============
st.sidebar.header('⚙️ 参数设置')
cache_days = st.sidebar.number_input('缓存天数 (天)', min_value=1, value=7)
max_snapshots = st.sidebar.number_input('保留快照份数', min_value=1, value=10)
drawdown_lookback = st.sidebar.number_input('回撤回溯期 (天)', min_value=1, value=30)

# ============ 上传与快照管理 ============
st.sidebar.header('📁 上传与快照管理')
uploaded = st.sidebar.file_uploader('上传 ATAS 导出数据 (.xlsx)', type='xlsx', accept_multiple_files=True)
market_file = st.sidebar.file_uploader('上传市场快照 CSV (Symbol,Time,MarketPrice)', type='csv')
sent_file = st.sidebar.file_uploader('上传舆情数据 CSV (Symbol,Date,SentimentScore)', type='csv')

SNAP_DIR = 'snapshots'
os.makedirs(SNAP_DIR, exist_ok=True)

# ============ 数据加载 ============
@st.cache_data(ttl=cache_days * 86400)
def load_and_clean(files):
    dfs = []
    for f in files:
        df_log = pd.read_excel(f, sheet_name='日志')
        df_log['上传文件'] = f.name
        dfs.append(df_log)
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna(subset=['PnL'])
    df = df.rename(columns={'平仓时间': '时间', '平仓价': '价格', '平仓量': '数量', 'PnL': '盈亏'})
    df['时间'] = pd.to_datetime(df['时间'], errors='coerce')
    df['数量'] = df['数量'].abs()
    df['方向'] = df['开仓量'].apply(lambda x: 'Buy' if x > 0 else 'Sell')
    for col in ['价格', '数量', '盈亏']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.dropna(subset=['时间', '价格']).sort_values('时间').reset_index(drop=True)

# ============ 主流程 ============
if uploaded:
    df = load_and_clean(uploaded)

    # 保存快照并清理
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    snap_file = f"atas_snapshot_{len(uploaded)}files_{now}.csv"
    df.to_csv(os.path.join(SNAP_DIR, snap_file), index=False)
    snaps = sorted(os.listdir(SNAP_DIR))
    if len(snaps) > max_snapshots:
        for old in snaps[:-max_snapshots]:
            os.remove(os.path.join(SNAP_DIR, old))
    st.sidebar.success(f"已加载 {len(df)} 条交易，快照：{snap_file}")

    # 视图 & 风险阈值预警设置
    view = st.sidebar.selectbox('视图分组', ['总体', '按账户', '按品种'])
    st.sidebar.header('⚠️ 风险阈值预警设置')
    max_loss = st.sidebar.number_input('单笔最大亏损阈值', value=-100.0)
    max_trades = st.sidebar.number_input('日内最大交易次数阈值', value=50)

    # ====== 指标与表格准备 ======
    df['累计盈亏'] = df['盈亏'].cumsum()
    df['日期'] = df['时间'].dt.date
    df['小时'] = df['时间'].dt.hour
    daily = df.groupby('日期')['盈亏'].sum().reset_index()
    hourly = df.groupby('小时')['盈亏'].mean().reset_index()
    df_sorted = df.sort_values(['账户', '品种', '时间'])
    df_sorted['持仓时长'] = df_sorted.groupby(['账户', '品种'])['时间'].diff().dt.total_seconds() / 60
    holding = df_sorted[['账户', '品种', '持仓时长']]
    returns = df['盈亏'].values
    sims, n = 500, len(returns)
    final = [np.random.choice(returns, n, replace=True).cumsum()[-1] for _ in range(sims)]
    monte_df = pd.DataFrame({'Monte Carlo Final': final})
    # 滑点
    if market_file:
        mp = pd.read_csv(market_file)
        mp['Time'] = pd.to_datetime(mp['Time'], errors='coerce')
        mp = mp.rename(columns={'MarketPrice': '市场价格', 'Symbol': '品种'})
        merge = df.merge(mp, left_on=['品种', '时间'], right_on=['品种', 'Time'], how='left')
        merge['滑点'] = merge['价格'] - merge['市场价格']
        slippage = merge[['时间', '品种', '价格', '市场价格', '滑点']]
    else:
        slippage = pd.DataFrame()
    # 舆情
    if sent_file:
        df_sent = pd.read_csv(sent_file)
        df_sent['Date'] = pd.to_datetime(df_sent['Date'], errors='coerce').dt.date
        sentiment = df_sent
    else:
        sentiment = pd.DataFrame()
    # 汇总指标
    total_pl = df['盈亏'].sum()
    ann_return = total_pl / max((df['时间'].max() - df['时间'].min()).days, 1) * 252
    downside_dev = df[df['盈亏'] < 0]['盈亏'].std()
    var95 = -df['盈亏'].quantile(0.05)
    cvar95 = -df[df['盈亏'] <= df['盈亏'].quantile(0.05)]['盈亏'].mean()
    sharpe = df['盈亏'].mean() / df['盈亏'].std() * np.sqrt(252)
    winrate = (df['盈亏'] > 0).mean()
    profit_factor = df[df['盈亏'] > 0]['盈亏'].mean() / (-df[df['盈亏'] < 0]['盈亏'].mean())
    # 回撤回溯期计算
    lookback_date = datetime.now() - timedelta(days=drawdown_lookback)
    df_lookback = df[df['时间'] >= lookback_date]
    mdd = (df_lookback['累计盈亏'] - df_lookback['累计盈亏'].cummax()).min()
    summary = pd.DataFrame({
        'Metric': ['Total P&L', 'Annual Return', 'Sharpe', 'Win Rate', 'Profit Factor', f'Max Drawdown ({drawdown_lookback}d)', 'VaR95', 'CVaR95', 'Downside Std'],
        'Value': [total_pl, ann_return, sharpe, winrate, profit_factor, mdd, var95, cvar95, downside_dev]
    })

    # ====== 风险阈值预警展示 ======
    st.subheader('⚠️ 风险阈值预警')
    if df['盈亏'].min() < max_loss:
        st.error(f"存在单笔盈亏低于阈值 ({max_loss}) 的记录！")
    if df[df['时间'].dt.date == datetime.today().date()].shape[0] > max_trades:
        st.error(f"今日交易次数超过阈值 ({max_trades}) 次！")

    # ====== 可视化展示 ======
    st.subheader('📈 累计盈亏趋势')
    if view == '按账户':
        st.plotly_chart(px.line(df, x='时间', y='累计盈亏', color='账户'), use_container_width=True)
    elif view == '按品种':
        st.plotly_chart(px.line(df, x='时间', y='累计盈亏', color='品种'), use_container_width=True)
    else:
        st.plotly_chart(px.line(df, x='时间', y='累计盈亏'), use_container_width=True)

    st.subheader('📊 日/小时盈亏')
    st.plotly_chart(px.bar(daily, x='日期', y='盈亏', title='每日盈亏'), use_container_width=True)
    st.plotly_chart(px.bar(hourly, x='小时', y='盈亏', title='每小时平均盈亏'), use_container_width=True)

    st.subheader('⏳ 持仓时长分布（分钟）')
    st.plotly_chart(px.box(holding, x='账户', y='持仓时长', title='按账户持仓时长'), use_container_width=True)
    st.plotly_chart(px.box(holding, x='品种', y='持仓时长', title='按品种持仓时长'), use_container_width=True)

    st.subheader('🎲 Monte Carlo 模拟')
    st.plotly_chart(px.histogram(monte_df, x='Monte Carlo Final', nbins=40, title='Monte Carlo 累积盈亏'), use_container_width=True)

    st.subheader('🕳️ 滑点与成交率分析')
    if not slippage.empty:
        st.plotly_chart(px.histogram(slippage, x='滑点', nbins=50, title='滑点分布'), use_container_width=True)
    else:
        st.info('请上传市场快照以启用滑点分析')

    st.subheader('📣 社交舆情热力图')
    if not sentiment.empty:
        heat = sentiment.pivot_table(index='Symbol', columns='Date', values='SentimentScore', aggfunc='mean')
        st.plotly_chart(px.imshow(heat, aspect='auto', title='舆情热力图'), use_container_width=True)
    else:
        st.info('请上传舆情数据以启用热力图')

    st.subheader('📌 核心统计指标')
    st.metric('夏普比率', f"{sharpe:.2f}")
    st.metric('胜率', f"{winrate:.2%}")
    st.metric('盈亏比', f"{profit_factor:.2f}")
    st.metric('年化收益率', f"{ann_return:.2f}")
    st.metric('下行风险', f"{downside_dev:.2f}")
    st.metric('VaR95', f"{var95:.2f}")
    st.metric('CVaR95', f"{cvar95:.2f}")
    st.metric(f'最大回撤 ({drawdown_lookback}天)', f"{mdd:.2f}")

    # ====== 导出功能 ======
    if pdf_available and st.button('📄 导出PDF报告'):
        pdf = FPDF()
        def add_table_page(title, df_table):
            pdf.add_page()
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, title, ln=True)
            pdf.set_font('Arial', '', 8)
            for i, row in df_table.iterrows():
                pdf.cell(0, 6, str(row.to_dict()), ln=True)
        add_table_page('Trades', df)
        add_table_page('Daily P&L', daily)
        add_table_page('Hourly P&L', hourly)
        add_table_page('Holding Time', holding)
        add_table_page('Monte Carlo', monte_df)
        if not slippage.empty:
            add_table_page('Slippage', slippage)
        if not sentiment.empty:
            add_table_page('Sentiment', sentiment)
        add_table_page('Summary', summary)
        pdf_output = io.BytesIO()
        pdf.output(pdf_output)
        st.download_button('下载PDF报告', data=pdf_output.getvalue(), file_name=f'ATS_Report_{now}.pdf', mime='application/pdf')

    if st.button('📥 导出Excel报告'):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Trades')
            daily.to_excel(writer, index=False, sheet_name='Daily P&L')
            hourly.to_excel(writer, index=False, sheet_name='Hourly P&L')
            holding.to_excel(writer, index=False, sheet_name='Holding Time')
            monte_df.to_excel(writer, index=False, sheet_name='Monte Carlo')
            if not slippage.empty:
                slippage.to_excel(writer, index=False, sheet_name='Slippage')
            if not sentiment.empty:
                sentiment.to_excel(writer, index=False, sheet_name='Sentiment')
            summary.to_excel(writer, index=False, sheet_name='Summary')
        st.download_button('下载Excel报告', data=output.getvalue(), file_name=f'ATS_Report_{now}.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
else:
    st.info('👆 请在侧边栏上传 .xlsx 文件进行分析')
