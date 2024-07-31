# makefile

build: docker-compose build web

run: docker-compose up -d

test: pytest --cov=payment_gateway_cc tests

install-test: pip install -r .\requirements-test.txt
