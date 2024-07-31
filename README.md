É necessario instalar o Docker e o Python.
Caso não seja Windows, o Docker Compose é requerido também.
No Windows Entre na pasta do projeto com o Powershell e execute o comando "docker-compose build web" e depois o comando "docker-compose up -d", para subir o projeto na porta 8000.
Se quiser executar os teste no Windows, "pip install -r .\requirements-test.txt", depois "python -m pytest --cov=payment_gateway_cc /tests".

Em outros SOs, está tudo no make.
