import os, sys
sys.stdout.reconfigure(encoding="utf-8")
html = []
html.append("<!DOCTYPE html>")
html.append('<html lang="zh-CN">')
html.append("<head>")
html.append('<meta charset="UTF-8">')
html.append("<title>test</title>")
html.append('<style>body{background:#000;color:#fff;font-family:sans-serif;padding:40px;}</style>')
html.append("</head><body>")
html.append("<h1>QlibWorks Test</h1>")
html.append("<p>If you see this, the generator works.</p>")
html.append("</body></html>")
with open("docs/design/out_test.html","w",encoding="utf-8") as f:
    f.write("\n".join(html))
print("Written OK")
