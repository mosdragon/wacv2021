DEST="gs://kdd2020hdvisai/static/"

.PHONY: all
all: serve

.PHONY: serve
serve:
	@echo "Serving on port 8080"
	@python3 -m http.server --cgi 8080


.PHONY: deploy
deploy:
	@gsutil -m cp -r *.html css js img ${DEST}
