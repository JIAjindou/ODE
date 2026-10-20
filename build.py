"""Build index.html (a complete document for GitHub Pages) from page.html (the page content, also used for previews)."""
src = open('page.html').read()
i = src.index('<nav class="top"')
head, body = src[:i], src[i:]
html = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        + head + '\n</head>\n<body>\n' + body + '\n</body>\n</html>\n')
open('index.html', 'w').write(html)
print('index.html', len(html))
