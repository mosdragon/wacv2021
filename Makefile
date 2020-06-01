DEST="gs://kdd2020hdvisai/static/"

.PHONY: all
all: serve

.PHONY: serve
serve:
	@echo "Serving on port 8080"
	@cd site && python3 -m http.server --cgi 8080


.PHONY: deploy
deploy:
	@gsutil -m cp -r site/*.html site/img site/css site/js ${DEST}
