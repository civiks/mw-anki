.PHONY: all sync enrich prune viz open anki

# fetch latest saved words, enrich, build deck + viz
all: sync viz

# fetch new saved words from MW + enrich (writes words.full.json, words.json, deck)
sync:
	uv run scrape.py

# re-enrich existing words without refetching the saved-word list
enrich:
	uv run scrape.py --enrich-only

# re-derive lean words.json + deck from words.full.json (no API calls)
prune:
	uv run scrape.py --prune

# rebuild the semantic viz
viz:
	uv run embed.py

open: viz
	open viz.html

# rebuild just the anki deck from words.json
anki:
	uv run python3 -c "import json, scrape; scrape.build_deck(json.load(open('words.json')))"
