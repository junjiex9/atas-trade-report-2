import streamlit as st
import pandas as pd
import numpy as np
import os
import io
from datetime import datetime
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

# ============ 侧边栏 ============
st.sidebar.header('📁 上传与快照管理')
uploaded = st.sidebar.file_uploader('上传 ATAS 导出数据 (.xlsx)', type='xlsx', accept_multiple_files=True)
market_file = st.sidebar.file_uploader('上传市场快照 CSV (Symbol,Time,MarketPrice)', type='csv')
sent_file = st.sidebar.file_uploader('上传舆情数据 CSV (Symbol,Date,SentimentScore)', type='csv')
max_snapshots = st.sidebar.number_input('保留最近快照份数', min_value=1, value=10)

SNAP_DIR = 'snapshots'
os.makedirs(SNAP_DIR, exist_ok=True)

# ============ 数据加载 ============
@st.cache_data
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
    df = df.dropna(subset=['时间', '价格']).sort_values('时间').reset_index(drop=True)
    return df

# ============ 主流程 ============
if uploaded:
    df = load_and_clean(uploaded)

    # 快照管理
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    snap_file = f"atas_snapshot_{len(uploaded)}files_{now}.csv"
    df.to_csv(os.path.join(SNAP_DIR, snap_file), index=False)
    snaps = sorted(os.listdir(SNAP_DIR))
    if len(snaps) > max_snapshots:
        for old in snaps[:-max_snapshots]:
            os.remove(os.path.join(SNAP_DIR, old))
    st.sidebar.success(f"已加载 {len(df)} 条交易，快照：{snap_file}")
    st.sidebar.write({f.name: len(df[df['上传文件']==f.name]) for f in uploaded})

    # 视图 & 风险预警
    view = st.sidebar.selectbox('视图分组', ['总体', '按账户', '按品种'])
    st.sidebar.header('⚠️ 风险阈值预警')
    max_loss = st.sidebar.number_input('单笔最大亏损', value=-100.0)
    max_trades = st.sidebar.number_input('日内最大交易次数', value=50)
    today_count = df[df['时间'].dt.date == datetime.today().date()].shape[0]
    if df['盈亏'].min() < max_loss:
        st.warning(f"⚠️ 存在单笔盈亏低于阈值({max_loss})！")
    if today_count > max_trades:
        st.warning(f"⚠️ 今日交易次数超过阈值({max_trades})！")

    # ====== 指标计算 ======
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

    # ====== 可视化 ======
    st.subheader('📈 累计盈亏趋势')
    if view == '按账户':
        fig = px.line(df, x='时间', y='累计盈亏', color='账户')
    elif view == '按品种':
        fig = px.line(df, x='时间', y='累计盈亏', color='品种')
    else:
        fig = px.line(df, x='时间', y='累计盈亏')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader('📊 日/小时盈亏')
    daily = df.groupby('日期')['盈亏'].sum().reset_index()
    hourly = df.groupby('小时')['盈亏'].mean().reset_index()
    st.plotly_chart(px.bar(daily, x='日期', y='盈亏', title='每日盈亏'))
    st.plotly_chart(px.bar(hourly, x='小时', y='盈亏', title='每小时平均盈亏'))

    st.subheader('⏳ 持仓时长分布（分钟）')
    df_sorted = df.sort_values(['账户', '品种', '时间'])
    df_sorted['持仓时长'] = df_sorted.groupby(['账户','品种'])['时间'].diff().dt.total_seconds()/60
    st.plotly_chart(px.box(df_sorted, x='账户', y='持仓时长', title='按账户持仓时长'))
    st.plotly_chart(px.box(df_sorted, x='品种', y='持仓时长', title='按品种持仓时长'))

    st.subheader('🎲 Monte Carlo 模拟')
    returns = df['盈亏'].values
    sims, n = 500, len(returns)
    final = [np.random.choice(returns, n, replace=True).cumsum()[-1] for _ in range(sims)]
    st.plotly_chart(px.histogram(final, nbins=40, title='Monte Carlo 累积盈亏'))

    st.subheader('🕳️ 滑点与成交率分析')
    if market_file:
        mp = pd.read_csv(market_file)
        mp['Time'] = pd.to_datetime(mp['Time'], errors='coerce')
        mp = mp.rename(columns={'MarketPrice':'市场价格','Symbol':'品种'})
        merge = df.merge(mp, left_on=['品种','时间'], right_on=['品种','Time'], how='left')
        merge['滑点'] = merge['价格'] - merge['市场价格']
        st.plotly_chart(px.histogram(merge, x='滑点', nbins=50, title='滑点分布'))
    else:
        st.info('请在侧边栏上传市场快照 CSV 以启用滑点分析')

    st.subheader('📣 社交舆情热力图')
    if sent_file:
        df_sent = pd.read_csv(sent_file)
        df_sent['Date'] = pd.to_datetime(df_sent['Date'], errors='coerce').dt.date
        heat = df_sent.pivot_table(index='Symbol', columns='Date', values='SentimentScore', aggfunc='mean')
        st.plotly_chart(px.imshow(heat, aspect='auto', title='舆情热力图'))
    else:
        st.info('请在侧边栏上传舆情 CSV 以启用热力图')

    st.subheader('📌 核心统计指标')
    st.metric('夏普比率', f"{sharpe:.2f}")
    st.metric('胜率', f"{winrate:.2%}")
    st.metric('盈亏比', f"{profit_factor:.2f}")
    st.metric('年化收益率', f"{ann_return:.2f}")
    st.metric('下行风险', f"{downside_dev:.2f}")
    st.metric('VaR95', f"{var95:.2f}")
    st.metric('CVaR95', f"{cvar95:.2f}")
    st.metric('最大回撤', f"{mdd:.2f}")

    # ====== 导出报告 ======
    if pdf_available:
        if st.button('📄 导出PDF报告'):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 10, 'ATAS 交易分析报告', ln=True)
            pdf.ln(10)
            pdf.set_font('Arial', '', 12)
            pdf.cell(0, 8, f"总盈亏: {total_pl:.2f}", ln=True)
            pdf.cell(0, 8, f"夏普比率: {sharpe:.2f}", ln=True)
            pdf_output = io.BytesIO()
            pdf.output(pdf_output)
            st.download_button('下载PDF报告', data=pdf_output.getvalue(), file_name=f'ATS_Report_{now}.pdf', mime='application/pdf')
    else:
        if lang == '中文':
            st.info('PDF 导出功能已禁用，请安装 fpdf2')

    # 导出 Excel
    if st.button('📥 导出Excel报告'):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Trades')
            daily.to_excel(writer, index=False, sheet_name='DailyPL')
            hourly.to_excel(writer, index=False, sheet_name='HourlyPL')
        st.download_button('下载Excel报告', data=output.getvalue(), file_name=f'ATS_Report_{now}.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
else:
    st.info('👆 请在侧边栏上传 .xlsx 文件进行分析')
