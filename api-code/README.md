# Lab 1: Containerizing a Basic API

## Overview 

This project is a FastAPI application that uses Poetry to manage project dependencies, pytest to 
test its functionalities, and is containerized with Docker to provide the following endpoints:

- `/generate-recipe`: A POST endpoint returning recipe information in a JSON format.
- `/transcribe-audio`: A POST endpoint that takes a query parameter `audio` and returns a JSON message with a string input of the .
- `/`: A root endpoint that returns a “Not Found” response.
- `/docs`: An endpoint that provides a browsable documentation while the API is running.
- `/openapi.json`: Returns a JSON object that meets the OpenAPI specification version 3+.

## Table of Contents

### Running the Application with Poetry
- [Prerequisites](#poetry-prerequisites)
- [How to Build the Application with Poetry](#how-to-build-the-application-with-poetry)
- [How to Run the Application with Poetry](#how-to-run-the-application-with-poetry)
- [How to Test the Application with Poetry](#how-to-test-the-application-with-poetry)

### Running the Application with Docker
- [Prerequisites](#docker-prerequisites)
- [How to Build the Application with Docker](#how-to-build-the-application-with-docker)
- [How to Run the Application with Docker](#how-to-run-the-application-with-docker)
- [How to Test the Application with Docker](#how-to-test-the-application-with-docker)

## Running the Application with Poetry

### Poetry Prerequisites

Before proceeding, ensure that you have Poetry installed on your machine. If not, you can install it 
by following the instructions at [Poetry Installation 
Guide](https://python-poetry.org/docs/#installing-with-the-official-installer). Make sure to add 
Poetry to your PATH.

### How to Build the Application with Poetry

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/UCB-W255/lab1-containerizing-a-basic-api-ashoksun01.git

2. **Navigate to Project Directory:**
   ```bash
   cd lab1-containerizing-a-basic-api-ashoksun01/lab1

3. **Install Project Dependencies:**
   ```bash
   poetry install

### How to Run the Application with Poetry

1. **Run Uvicorn Server:**
   ```bash
   poetry run uvicorn src.main:app --reload

2. **Access API:**
   Open your browser and go to http://127.0.0.1:8000 to interact with the FastAPI application.

### How to Test the Application with Poetry

1. **Run Pytest:**
   ```bash
   poetry run pytest

## Running the Application with Docker

### Docker Prerequisites

Before proceeding, ensure that you have Docker installed on your machine. If not, you can install it 
by following the instructions at [Docker Installation Guide](https://docs.docker.com/engine/install/).

### How to Build the Application with Docker

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/UCB-W255/lab1-containerizing-a-basic-api-ashoksun01.git
2. **Navigate to Project Directory:**
   ```bash
   cd lab1-containerizing-a-basic-api-ashoksun01/lab1
3. **Build Docker Image:**
   ```bash
   docker build -t <docker-image> .

   docker build -t fastapi-app .

### How to Run the Application with Docker

1. **Run Docker Container:**
   ```bash 
   docker run -p <host-port>:<container-post> <docker_image>

   docker run -p 8080:8000 fastapi-app
2. **Access API:**
   Open your browser and go to http://localhost:host-port to interact with the FastAPI application 
(ex. http://localhost:8080).

### How to Test the Application with Docker

1. **Run Pytest:**
   ```bash
   docker run <docker-image> pytest

   docker run fastapi-app pytest
