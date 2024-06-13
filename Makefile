build:
	@python3 ./aicomm.py

run: build
	# @./bin/wserv

test:
	@go test -v ./...
