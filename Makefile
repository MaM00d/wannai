build:
	@python ./aicomm.py

run: build
	# @./bin/wserv

test:
	@go test -v ./...
