# Visual Summary: 从珠峰到泥潭 (Premium Edition)

## Slide 1: Cover (The Hook)
```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1080 1440">
  <defs>
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;700;900&amp;display=swap');
      .title { font-family: 'Noto Sans SC', sans-serif; font-weight: 900; font-size: 130px; fill: #FFFFFF; }
      .sub { font-family: 'Noto Sans SC', sans-serif; font-weight: 400; font-size: 50px; fill: #CCCCCC; }
      .tag { font-family: 'Noto Sans SC', sans-serif; font-weight: 700; font-size: 40px; fill: #1D1D1F; }
    </style>
    <linearGradient id="grad_dark" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#141E30;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#243B55;stop-opacity:1" />
    </linearGradient>
    <linearGradient id="grad_gold" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#F2994A;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#F2C94C;stop-opacity:1" />
    </linearGradient>
    <filter id="shadow">
      <feDropShadow dx="0" dy="10" stdDeviation="10" flood-opacity="0.5"/>
    </filter>
  </defs>
  
  <rect width="100%" height="100%" fill="url(#grad_dark)"/>
  
  <!-- Decorative Circle -->
  <circle cx="900" cy="200" r="300" fill="url(#grad_gold)" opacity="0.1"/>
  <circle cx="100" cy="1200" r="400" fill="#CC0000" opacity="0.1"/>
  
  <!-- Tag -->
  <rect x="100" y="150" width="300" height="80" rx="40" fill="url(#grad_gold)"/>
  <text x="250" y="205" text-anchor="middle" class="tag">深度商业解析</text>
  
  <text x="100" y="500" class="title" filter="url(#shadow)">
    <tspan x="100" dy="0">从珠峰</tspan>
    <tspan x="100" dy="160" fill="url(#grad_gold)">到泥潭</tspan>
  </text>
  
  <line x1="100" y1="700" x2="300" y2="700" stroke="white" stroke-width="10" opacity="0.5"/>
  
  <text x="100" y="850" class="sub">
    <tspan x="100" dy="0">万科“实用主义”的致命异化</tspan>
    <tspan x="100" dy="80">——当生存成为唯一的信仰，</tspan>
    <tspan x="100" dy="80">企业将付出怎样的代价？</tspan>
  </text>
</svg>
```

## Slide 2: 时代的错位 (Context)
```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1080 1440">
  <defs>
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;700;900&amp;display=swap');
      .head { font-family: 'Noto Sans SC', sans-serif; font-weight: 900; font-size: 90px; fill: #1D1D1F; }
      .card-title { font-family: 'Noto Sans SC', sans-serif; font-weight: 700; font-size: 50px; fill: #CC0000; }
      .card-body { font-family: 'Noto Sans SC', sans-serif; font-weight: 400; font-size: 40px; fill: #555555; }
    </style>
    <linearGradient id="bg_grad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#FDFBFB;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#EBEDEE;stop-opacity:1" />
    </linearGradient>
    <filter id="card-shadow">
      <feDropShadow dx="0" dy="15" stdDeviation="20" flood-opacity="0.15"/>
    </filter>
  </defs>
  
  <rect width="100%" height="100%" fill="url(#bg_grad)"/>
  
  <text x="540" y="150" text-anchor="middle" class="head">两个时代的错位</text>
  
  <!-- Card 1 -->
  <rect x="80" y="250" width="920" height="480" rx="30" fill="white" filter="url(#card-shadow)"/>
  <rect x="80" y="250" width="20" height="480" rx="10" fill="#243B55"/> <!-- Left Border -->
  
  <text x="150" y="340" class="card-title">2003 王石时代</text>
  <text x="150" y="430" class="card-body">
    <tspan x="150" dy="0">背景：增量红利，遍地黄金。</tspan>
    <tspan x="150" dy="65">表现：登珠峰，谈情怀，讲体面。</tspan>
    <tspan x="150" dy="65" font-weight="700">本质：时代的红利足以覆盖</tspan>
    <tspan x="150" dy="65" font-weight="700">理想主义的成本。</tspan>
  </text>
  
  <!-- Card 2 -->
  <rect x="80" y="800" width="920" height="480" rx="30" fill="white" filter="url(#card-shadow)"/>
  <rect x="80" y="800" width="20" height="480" rx="10" fill="#CC0000"/> <!-- Left Border -->
  
  <text x="150" y="890" class="card-title" fill="#CC0000">2018 郁亮时代</text>
  <text x="150" y="980" class="card-body">
    <tspan x="150" dy="0">背景：存量博弈，零和游戏。</tspan>
    <tspan x="150" dy="65">表现：喊出“活下去”，动作变形。</tspan>
    <tspan x="150" dy="65" font-weight="700">本质：为了生存，将企业转向</tspan>
    <tspan x="150" dy="65" font-weight="700">“极致的实用主义”。</tspan>
  </text>
</svg>
```

## Slide 3: 制度的异化 (Mechanism)
```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1080 1440">
  <defs>
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;700;900&amp;display=swap');
      .title { font-family: 'Noto Sans SC', sans-serif; font-weight: 900; font-size: 100px; fill: #FFFFFF; }
      .num { font-family: 'Noto Sans SC', sans-serif; font-weight: 900; font-size: 120px; fill: rgba(255,255,255,0.1); }
      .h3 { font-family: 'Noto Sans SC', sans-serif; font-weight: 700; font-size: 55px; fill: #F2C94C; }
      .p { font-family: 'Noto Sans SC', sans-serif; font-weight: 400; font-size: 42px; fill: #E0E0E0; }
    </style>
    <linearGradient id="grad_red" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#8E0E00;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#1F1C18;stop-opacity:1" />
    </linearGradient>
  </defs>
  
  <rect width="100%" height="100%" fill="url(#grad_red)"/>
  
  <!-- Background Pattern -->
  <path d="M0,0 L1080,1440" stroke="white" stroke-width="2" opacity="0.05"/>
  <path d="M1080,0 L0,1440" stroke="white" stroke-width="2" opacity="0.05"/>
  
  <text x="100" y="200" class="title">合规的崩塌</text>
  
  <!-- Point 1 -->
  <text x="80" y="400" class="num">01</text>
  <text x="250" y="380" class="h3">跟投机制变味</text>
  <text x="100" y="480" class="p">
    <tspan x="100" dy="0">初衷是“利益共享”，</tspan>
    <tspan x="100" dy="65">却异化为“投名状”。</tspan>
    <tspan x="100" dy="65">员工被迫掏空积蓄为项目输血，</tspan>
    <tspan x="100" dy="65" fill="#FF6B6B" font-weight="700">“不跟投就是不忠诚”。</tspan>
  </text>
  
  <!-- Point 2 -->
  <text x="80" y="900" class="num">02</text>
  <text x="250" y="880" class="h3">手段吞噬目的</text>
  <text x="100" y="980" class="p">
    <tspan x="100" dy="0">为了维持报表好看，</tspan>
    <tspan x="100" dy="65">一线不得不突破底线。</tspan>
    <tspan x="100" dy="65">合规文化在“活下去”的KPI下，</tspan>
    <tspan x="100" dy="65" fill="#FF6B6B" font-weight="700">被悄无声息地瓦解。</tspan>
  </text>
</svg>
```

## Slide 4: 悲剧的根源 (Quote)
```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1080 1440">
  <defs>
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;700;900&amp;display=swap');
      .quote-mark { font-family: serif; font-size: 300px; fill: #F2994A; opacity: 0.5; }
      .quote-text { font-family: 'Noto Sans SC', sans-serif; font-weight: 900; font-size: 80px; fill: #333333; }
      .highlight { fill: #CC0000; text-decoration: underline; }
    </style>
    <filter id="blur-bg">
      <feGaussianBlur stdDeviation="50" />
    </filter>
  </defs>
  
  <rect width="100%" height="100%" fill="#FFFFFF"/>
  <circle cx="200" cy="200" r="300" fill="#F2C94C" opacity="0.1" filter="url(#blur-bg)"/>
  <circle cx="900" cy="1200" r="400" fill="#243B55" opacity="0.1" filter="url(#blur-bg)"/>
  
  <text x="100" y="350" class="quote-mark">“</text>
  
  <text x="150" y="550" class="quote-text">
    <tspan x="150" dy="0">当生存成为</tspan>
    <tspan x="150" dy="120" class="highlight">唯一的信仰</tspan>
    <tspan x="150" dy="120">手段就变得</tspan>
    <tspan x="150" dy="120">不再重要。</tspan>
  </text>
  
  <line x1="150" y1="950" x2="930" y2="950" stroke="#CCCCCC" stroke-width="2"/>
  
  <rect x="100" y="1050" width="880" height="250" rx="20" fill="#F5F5F7"/>
  <text x="150" y="1150" font-family="'Noto Sans SC', sans-serif" font-size="40" fill="#555555">
    <tspan x="150" dy="0">万科的教训警示我们：</tspan>
    <tspan x="150" dy="70" font-weight="700" fill="#1D1D1F">没有法治作为压舱石，</tspan>
    <tspan x="150" dy="70" font-weight="700" fill="#1D1D1F">极致的实用主义终将触礁。</tspan>
  </text>
</svg>
```

## Slide 5: Outro (Reflection)
```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1080 1440">
  <defs>
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;700;900&amp;display=swap');
      .cta { font-family: 'Noto Sans SC', sans-serif; font-weight: 900; font-size: 90px; fill: #FFFFFF; }
      .small { font-family: 'Noto Sans SC', sans-serif; font-weight: 400; font-size: 35px; fill: #AAAAAA; }
    </style>
    <linearGradient id="grad_night" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#000000;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#434343;stop-opacity:1" />
    </linearGradient>
  </defs>
  
  <rect width="100%" height="100%" fill="url(#grad_night)"/>
  
  <!-- Question Mark Art -->
  <circle cx="540" cy="500" r="180" fill="none" stroke="#F2C94C" stroke-width="10"/>
  <text x="540" y="620" text-anchor="middle" font-size="250" font-family="'Noto Sans SC', sans-serif" font-weight="900" fill="#F2C94C">?</text>
  
  <text x="540" y="900" text-anchor="middle" class="cta">
    <tspan x="540" dy="0">我们是否还记得</tspan>
    <tspan x="540" dy="140">当初为何出发？</tspan>
  </text>
  
  <rect x="340" y="1150" width="400" height="100" rx="50" stroke="white" stroke-width="2" fill="none"/>
  <text x="540" y="1215" text-anchor="middle" font-size="40" fill="white" font-family="'Noto Sans SC', sans-serif">
    评论区聊聊你的看法
  </text>
  
  <text x="540" y="1350" text-anchor="middle" class="small">商业 | 深度 | 思考</text>
</svg>
```
