from kocrawl.spell import SpellCrawler


utterance = '점심 때 뭐 할까?'
print(utterance)
# 맞춤법 검사기
spellCheck = SpellCrawler()
utterance = spellCheck.request(utterance)

print(utterance)

