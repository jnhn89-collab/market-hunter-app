import google.generativeai as genai
import yfinance as yf
from finvizfinance.screener.overview import Overview
from finvizfinance.quote import Quote
import json
import random
import matplotlib
matplotlib.use('Agg') # 서버 전용 (창 안 띄움)
import matplotlib.pyplot as plt
import io
import base64
import requests
import xml.etree.ElementTree as ET
import os
import pandas as pd
import re
import time
import datetime # 날짜 에러 해결

# ==========================================
# 🔑 API 키 (환경변수)
# ==========================================
API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_FILE = os.path.join(BASE_DIR, 'history.json')
TODAY_FILE = os.path.join(BASE_DIR, 'today.json')
CHART_TECH_FILE = os.path.join(BASE_DIR, 'chart_tech.png')
CHART_FUND_FILE = os.path.join(BASE_DIR, 'chart_fund.png')

# ---------------------------------------------------------
# 🛠️ 유틸리티 함수들 (Base64 이미지 변환 포함)
# ---------------------------------------------------------

def get_chart_base64(ticker, df_ta, title):
    """기술적 분석 차트 -> Base64 변환"""
    try:
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [3, 1]})
        
        subset = df_ta.iloc[-90:]
        ax1.plot(subset.index, subset['Close'], color='#00ff88', label='Price')
        ax1.plot(subset.index, subset['SMA20'], color='yellow', linestyle='--', label='20 SMA')
        ax1.set_title(title, fontsize=14, color='white', fontweight='bold')
        ax1.grid(True, linestyle=':', alpha=0.3)
        
        ax2.plot(subset.index, subset['RSI'], color='cyan', label='RSI')
        ax2.axhline(70, color='red', linestyle=':', alpha=0.5)
        ax2.axhline(30, color='green', linestyle=':', alpha=0.5)
        ax2.set_title("RSI (14)", fontsize=10)
        ax2.grid(True, linestyle=':', alpha=0.3)
        
        plt.tight_layout()
        
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        b64_data = base64.b64encode(img.getvalue()).decode()
        plt.close()
        return f"data:image/png;base64,{b64_data}"
    except: return None

def get_fund_chart_base64(ticker, fund_data):
    """펀더멘털 스코어카드 -> Base64 변환"""
    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.axis('off')
        
        def clean(v): return "N/A" if v in [None, 'N/A', '-'] else str(v)
        
        text = f"""
        {ticker} Fundamental Scorecard
        -----------------------------------
        
        [Valuation]
        PER: {clean(fund_data.get('PER'))}
        PBR: {clean(fund_data.get('PBR'))}
        
        [Profitability]
        ROE: {clean(fund_data.get('ROE'))}
        Margin: {clean(fund_data.get('ProfitMargin'))}
        
        [Growth]
        Sales Growth: {clean(fund_data.get('RevenueGrowth'))}
        """
        ax.text(0.1, 0.5, text, fontsize=15, color='white', fontfamily='monospace', va='center')
        
        img = io.BytesIO()
        plt.savefig(img, format='png', facecolor='#1e1e1e', bbox_inches='tight')
        img.seek(0)
        b64_data = base64.b64encode(img.getvalue()).decode()
        plt.close()
        return f"data:image/png;base64,{b64_data}"
    except: return None

def get_technical_data(ticker, hist):
    try:
        df = hist.copy()
        close = df['Close']
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['SMA20'] = close.rolling(20).mean()
        
        curr = df.iloc[-1]
        report = f"[Tech] Price:${curr['Close']:.2f}, RSI:{curr['RSI']:.2f}, SMA20:${curr['SMA20']:.2f}"
        return report, df
    except: return "TA Error", hist

def get_fundamental_data(ticker):
    data = {}
    try:
        t = yf.Ticker(ticker)
        i = t.info
        if i.get('trailingPE') or i.get('priceToBook'):
            data = {
                "PER": i.get('trailingPE', 'N/A'),
                "PBR": i.get('priceToBook', 'N/A'),
                "ROE": i.get('returnOnEquity', 'N/A'),
                "ProfitMargin": i.get('profitMargin', 'N/A'),
                "RevenueGrowth": i.get('revenueGrowth', 'N/A')
            }
    except: pass
    
    if not data or data.get('PER') == 'N/A':
        try:
            q = Quote(); q.set_ticker(ticker); fund = q.ticker_fundament()
            data = {
                "PER": fund.get('P/E', 'N/A'), "PBR": fund.get('P/B', 'N/A'),
                "ROE": fund.get('ROE', 'N/A'), "ProfitMargin": fund.get('Profit Margin', 'N/A'),
                "RevenueGrowth": fund.get('Sales Q/Q', 'N/A')
            }
        except: pass

    def fmt(v, is_pct=False):
        if v in [None, 'N/A', '-']: return "N/A"
        if isinstance(v, str) and '%' in v: return v
        try: return f"{float(v)*100:.2f}%" if is_pct else f"{float(v):.2f}"
        except: return str(v)

    final_data = {
        "PER": fmt(data.get("PER")), "PBR": fmt(data.get("PBR")),
        "ROE": fmt(data.get("ROE"), True), "ProfitMargin": fmt(data.get("ProfitMargin"), True),
        "RevenueGrowth": fmt(data.get("RevenueGrowth"), True)
    }
    return f"[Fund] PER:{final_data['PER']}, PBR:{final_data['PBR']}", final_data

def get_news_robust(ticker):
    news = []
    try:
        url = f"https://news.google.com/rss/search?q={ticker}+stock+when:7d&hl=en-US&gl=US&ceid=US:en"
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
        root = ET.fromstring(res.content)
        for item in root.findall('.//item')[:3]: news.append(f"- {item.find('title').text}")
    except: pass
    
    if not news:
        try:
            q = Quote(); q.set_ticker(ticker); news_df = q.ticker_news_outer()
            for i, r in news_df.head(3).iterrows(): news.append(f"- {r['Title']}")
        except: pass
        
    return "\n".join(news) if news else "No News"

def get_full_name(ticker):
    try: return yf.Ticker(ticker).info.get('longName', ticker)
    except: return ticker

# --- [메인 실행 함수] ---
def run_hunter(mode="ROCKET"):
    print(f"🚀 [Hunter] 분석 시작: Mode={mode}")
    
    # 🚨 [복구완료] 사용자님이 원하시던 8종 전략 그대로 적용
    STRATEGY_BOOK = {
        "ROCKET": [
            {
                "name": "Top Gainers (Large Cap)",
                "desc": "시총 100억불 이상 대형주 중 오늘 가장 강한 놈",
                "filters": {'Market Cap.': '+Large (over $10bln)'},
                "signal": "Top Gainers"
            },
            {
                "name": "Oversold Bounce",
                "desc": "과매도(RSI<30) 상태에서 기술적 반등이 나오는 놈",
                "filters": {'RSI (14)': 'Oversold (30)'}, 
                "signal": "Top Gainers"
            },
            {
                "name": "Earnings Surprise",
                "desc": "이번주 실적 발표 이슈가 있는 놈",
                "filters": {'Earnings Date': 'This Week'},
                "signal": "Top Gainers"
            },
            {
                "name": "Bullish Breakout", 
                "desc": "상승 삼각형 패턴을 완성하고 위로 쏘는 놈",
                "filters": {'Pattern': 'Triangle Ascending (Strong)'},
                "signal": "Top Gainers"
            }
        ],
        "SEED": [
            {
                "name": "Analyst Strong Buy",
                "desc": "월가 형님들이 '강력 매수' 외치는 놈",
                "filters": {'Analyst Recom.': 'Strong Buy (1)'},
                "signal": "New High"
            },
            {
                "name": "Trend Support Buy", 
                "desc": "상승 추세선 지지를 받고 다시 튀어 오를 준비하는 놈",
                "filters": {'Pattern': 'TL Support (Strong)'},
                "signal": "New High"
            },
            {
                "name": "Heavy Volume Buying",
                "desc": "평소보다 거래량이 2배 이상 터지며 매집 들어온 놈",
                "filters": {'Relative Volume': 'Over 2'},
                "signal": "New High"
            },
            {
                "name": "Channel Up",
                "desc": "상승 채널을 그리며 우상향하는 놈",
                "filters": {'Pattern': 'Channel Up (Strong)'},
                "signal": "Most Active"
            }
        ]
    }
    
    strategy = random.choice(STRATEGY_BOOK[mode])
    target_ticker = None

    # 1. 스크리닝
    try:
        foverview = Overview()
        foverview.set_filter(signal=strategy['signal'], filters_dict=strategy['filters'])
        df = foverview.screener_view()
        if df.empty: raise Exception("Empty")
        target_ticker = df.iloc[:10].sample(n=1).iloc[0]['Ticker']
        print(f"👉 타겟 포착: {target_ticker} ({strategy['name']})")
    except:
        target_ticker = random.choice(["NVDA", "PLTR", "TSLA", "AMD", "SOFI", "COIN"])
        print(f"⚠️ 백업 타겟 사용: {target_ticker}")

    # 2. 데이터 수집
    company_name = get_full_name(target_ticker)
    stock = yf.Ticker(target_ticker)
    hist = stock.history(period="6mo")
    
    ta_text, df_ta = get_technical_data(target_ticker, hist)
    fund_text, fund_dict = get_fundamental_data(target_ticker)
    news_text = get_news_robust(target_ticker)
    
    # 3. 차트 생성 (메모리 Base64)
    chart_tech = get_chart_base64(target_ticker, df_ta, f"{target_ticker} Tech")
    chart_fund = get_fund_chart_base64(target_ticker, fund_dict)

    # 4. AI 분석
    print("🧠 Gemini 분석 중...")
    
    # [수정 1] 프롬프트에 tag, summary 필드를 명확히 요구
    prompt = f"""
    Role: Hedge Fund Manager
    Ticker: {target_ticker} ({company_name})
    Strategy: {strategy['name']}
    Data: {ta_text} \n {fund_text}
    News: {news_text}
    
    Mission: Create a 5-step analysis quiz.
    
    [Output JSON Structure]
    {{
        "intro": {{ "summary": "Korean 1-line company description" }},
        "quiz_cards": [
            {{ 
                "title": "Round 1: 기술적 분석", 
                "tag": "Technical", 
                "summary": "차트의 핵심 포인트 (예: RSI 과매수)", 
                "description": "상세 분석 내용 (한국어)", 
                "quiz": "O/X 퀴즈", 
                "answer": "O", 
                "comment": "해설" 
            }},
            {{ 
                "title": "Round 2: 펀더멘털", 
                "tag": "Fundamental", 
                "summary": "재무 건전성 요약", 
                "description": "상세 분석 내용 (한국어)", 
                "quiz": "O/X 퀴즈", 
                "answer": "O", 
                "comment": "해설" 
            }},
            {{ 
                "title": "Round 3: 뉴스/재료", 
                "tag": "Catalyst", 
                "summary": "뉴스 한 줄 요약", 
                "description": "상세 분석 내용 (한국어)", 
                "quiz": "O/X 퀴즈", 
                "answer": "O", 
                "comment": "해설" 
            }},
            {{ 
                "title": "Round 4: 시장 심리", 
                "tag": "Sentiment", 
                "summary": "현재 시장의 분위기", 
                "description": "상세 분석 내용 (한국어)", 
                "quiz": "O/X 퀴즈", 
                "answer": "O", 
                "comment": "해설" 
            }},
            {{ 
                "title": "Final: 최종 결론", 
                "tag": "Strategy", 
                "summary": "매수/매도/관망", 
                "description": "최종 전략 (한국어)", 
                "quiz": "O/X 퀴즈", 
                "answer": "O", 
                "comment": "해설" 
            }}
        ]
    }}
    """
    
    try:
        safety_settings = [{"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}]
        res = model.generate_content(prompt, safety_settings=safety_settings)
        match = re.search(r'\{.*\}', res.text, re.DOTALL)
        ai_data = json.loads(match.group(0))
    except Exception as e:
        print(f"❌ AI 분석 실패: {e}")
        ai_data = {"intro": {"summary": "분석 실패"}, "quiz_cards": []}

    # [수정 2] 데이터 조립 시 이미지와 타입 강제 주입
    # AI가 이미지를 직접 다루지 못하므로, 파이썬이 순서에 맞춰 이미지를 끼워 넣어줍니다.
    
    generated_cards = ai_data.get('quiz_cards', [])
    processed_cards = []

    for idx, card in enumerate(generated_cards):
        # 기본 데이터 복사
        new_card = card.copy()
        
        # Round 1 (인덱스 0) -> Tech 차트
        if idx == 0: 
            new_card['type'] = 'CONTEXT'
            new_card['image'] = chart_tech
            
        # Round 2 (인덱스 1) -> Fund 차트
        elif idx == 1: 
            new_card['type'] = 'FUNDAMENTAL'
            new_card['image'] = chart_fund
            
        # 나머지 -> 이미지 없음
        else:
            new_card['type'] = 'NORMAL'
            new_card['image'] = None
            
        processed_cards.append(new_card)

    # 최종 결과 리스트 (프로필 카드 + 처리된 퀴즈 카드들)
    # ---------------------------------------------------------
    # 5. 결과 조립 및 저장
    # ---------------------------------------------------------
    final_cards = [
        {
            "type": "PROFILE",
            "title": target_ticker,
            "tag": f"{mode} | {strategy['name']}",
            "summary": company_name,
            "description": ai_data.get('intro', {}).get('summary', 'Ready'),
            "quiz": "Start Analysis? (O)", "answer": "O", "comment": "Let's Go!",
            "image": None,
            "website": f"https://finance.yahoo.com/quote/{target_ticker}"
        }
    ] + ai_data.get('quiz_cards', [])

    # 이미지 매핑 (Base64 데이터 연결)
    if len(final_cards) > 1: final_cards[1]['image'] = chart_tech
    if len(final_cards) > 2: final_cards[2]['image'] = chart_fund

    # 최종 결과 데이터
    result_data = {
        "date": datetime.datetime.now().strftime("%Y-%m-%d"),
        "ticker": target_ticker,
        "company_name": company_name,
        "cards": final_cards,
        "conclusion": ai_data.get('conclusion', '')
    }

    # 🚨 [History 저장 로직] 리턴하기 전에 저장 수행!
    try:
        new_record = {
            "date": result_data["date"], 
            "ticker": target_ticker, 
            "company_name": company_name,
            "mode": mode, 
            "price": hist['Close'].iloc[-1] if not hist.empty else 0, 
            "cards": final_cards
        }

        history = []
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, 'r', encoding='utf-8') as f: 
                    history = json.load(f)
            except: pass

        # 중복 제거 (같은 날짜, 같은 종목이면 기존 것 삭제하고 최신으로 갱신)
        history = [h for h in history if not (h['date'] == new_record['date'] and h['ticker'] == target_ticker)]
        
        # 최신순으로 맨 앞에 추가
        history.insert(0, new_record)

        # 파일에 쓰기
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        print(f"⚠️ 히스토리 저장 실패: {e}")

    # 저장 후 결과 반환
    return result_data