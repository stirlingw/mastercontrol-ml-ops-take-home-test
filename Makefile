#!make

# Options
PROJECT=$(shell basename $(PWD))
TIMESTAMP=$(shell date "+%Y-%m-%d_%H.%M.%S")

# development
build:
	docker build --compress -t $(PROJECT).base -f Dockerfile .

run:
	docker run -d --name $(PROJECT).container -p 8000:8000 $(PROJECT).base:latest

restart-dev: 
	docker ps -aq | xargs docker stop | xargs docker rm
	make run

restart-dev-full:
	docker system prune --all
	docker ps -aq | xargs docker stop | xargs docker rm
	make run

# test
pytest:
    docker exec -it $(PROJECT).container pytest --ignore=tests/ --cov=app tests/ --cov-config=.coveragerc

